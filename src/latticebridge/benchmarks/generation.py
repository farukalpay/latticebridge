from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import time
from collections import Counter

import torch

from latticebridge.constraints.phrase_automaton import SurfaceProductAutomaton
from latticebridge.metrics.text import rouge_l_f1, token_f1
from latticebridge.models.prefix_lm import PrefixLanguageModel
from latticebridge.models.tokenizer import LatticeTokenizer


@dataclass
class BenchmarkTask:
    dataset_name: str
    example_id: str
    source_text: str
    target_text: str
    references: list[str]
    required_phrases: list[str]


@dataclass
class GenerationResult:
    method: str
    dataset_name: str
    example_id: str
    text: str
    success: bool
    coverage: float
    rouge_l: float
    token_f1: float
    log_score: float
    runtime_seconds: float
    metadata: dict[str, float | int | str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_benchmark_tasks(
    records: list,
    tokenizer: LatticeTokenizer,
    *,
    max_anchors: int = 3,
    min_anchors: int = 2,
) -> list[BenchmarkTask]:
    phrase_document_frequency: Counter[str] = Counter()
    document_count = 0
    for record in records:
        normalized_candidates = {
            phrase.lower().strip()
            for phrase in record.candidate_phrases
            if phrase.lower().strip()
        }
        if not normalized_candidates:
            continue
        phrase_document_frequency.update(normalized_candidates)
        document_count += 1

    def information_score(normalized_phrase: str) -> float:
        document_frequency = phrase_document_frequency.get(normalized_phrase, 0)
        return math.log((document_count + 1.0) / (document_frequency + 1.0))

    tasks: list[BenchmarkTask] = []
    for record in records:
        reference_texts = record.references or [record.target_text]
        normalized_refs = [text.lower() for text in reference_texts]
        feasible: list[tuple[float, str, str]] = []
        seen: set[str] = set()
        for phrase in record.candidate_phrases:
            normalized_phrase = phrase.lower().strip()
            if normalized_phrase and normalized_phrase not in seen and any(normalized_phrase in ref for ref in normalized_refs):
                feasible.append((information_score(normalized_phrase), normalized_phrase, phrase))
                seen.add(normalized_phrase)
        if len(feasible) < min_anchors:
            continue
        feasible.sort(key=lambda item: (-item[0], item[1]))
        ordered_phrases = [phrase for _, _, phrase in feasible]
        required_phrases = ordered_phrases if max_anchors <= 0 else ordered_phrases[:max_anchors]
        tasks.append(
            BenchmarkTask(
                dataset_name=record.dataset_name,
                example_id=record.example_id,
                source_text=record.source_text,
                target_text=record.target_text,
                references=reference_texts,
                required_phrases=required_phrases,
            )
        )
    return tasks


def _prefix_ids(tokenizer: LatticeTokenizer, source_text: str) -> list[int]:
    bos_id = tokenizer.id_for("<bos>")
    src_id = tokenizer.id_for("<src>")
    tgt_id = tokenizer.id_for("<tgt>")
    return [bos_id, src_id] + tokenizer.encode(source_text) + [tgt_id]


def _decode_generated(tokenizer: LatticeTokenizer, token_ids: list[int]) -> str:
    eos_id = tokenizer.id_for("<eos>")
    if eos_id in token_ids:
        token_ids = token_ids[: token_ids.index(eos_id)]
    return tokenizer.decode(token_ids)


def _coverage(tokenizer: LatticeTokenizer, text: str, required_phrases: list[str]) -> float:
    if not required_phrases:
        return 1.0
    normalized_text = text.lower()
    hit_count = 0
    for phrase in required_phrases:
        if phrase.lower().strip() in normalized_text:
            hit_count += 1
    return hit_count / len(required_phrases)


def _candidate_result(
    method: str,
    task: BenchmarkTask,
    tokenizer: LatticeTokenizer,
    token_ids: list[int],
    log_score: float,
    runtime_seconds: float,
    metadata: dict[str, float | int | str] | None = None,
) -> GenerationResult:
    text = _decode_generated(tokenizer, token_ids)
    coverage = _coverage(tokenizer, text, task.required_phrases)
    references = task.references or [task.target_text]
    return GenerationResult(
        method=method,
        dataset_name=task.dataset_name,
        example_id=task.example_id,
        text=text,
        success=coverage >= 0.999,
        coverage=coverage,
        rouge_l=rouge_l_f1(text, references),
        token_f1=token_f1(text, references),
        log_score=log_score,
        runtime_seconds=runtime_seconds,
        metadata=metadata or {},
    )


def _selection_key(result: GenerationResult) -> tuple[int, float, float]:
    return (int(result.success), result.log_score, result.rouge_l)


def greedy_decode(
    model: PrefixLanguageModel,
    tokenizer: LatticeTokenizer,
    task: BenchmarkTask,
    *,
    max_new_tokens: int,
    device: torch.device,
) -> GenerationResult:
    start = time.perf_counter()
    prefix_ids = _prefix_ids(tokenizer, task.source_text)
    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    last_logits, hidden = model.warm_start(prefix_tensor)
    eos_id = tokenizer.id_for("<eos>")

    generated: list[int] = []
    total_logprob = 0.0
    for _ in range(max_new_tokens):
        log_probs = torch.log_softmax(last_logits, dim=-1)
        token_id = int(torch.argmax(log_probs, dim=-1).item())
        total_logprob += float(log_probs[0, token_id].item())
        generated.append(token_id)
        if token_id == eos_id:
            break
        last_logits, hidden = model.step(torch.tensor([token_id], device=device), hidden)
    return _candidate_result(
        "greedy",
        task,
        tokenizer,
        generated,
        total_logprob,
        time.perf_counter() - start,
    )


def ancestral_best_of_k(
    model: PrefixLanguageModel,
    tokenizer: LatticeTokenizer,
    task: BenchmarkTask,
    *,
    max_new_tokens: int,
    num_samples: int,
    temperature: float,
    device: torch.device,
) -> GenerationResult:
    start = time.perf_counter()
    prefix_ids = _prefix_ids(tokenizer, task.source_text)
    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device).repeat(num_samples, 1)
    last_logits, hidden = model.warm_start(prefix_tensor)
    eos_id = tokenizer.id_for("<eos>")

    generated = [[] for _ in range(num_samples)]
    log_scores = torch.zeros(num_samples, device=device)
    done = torch.zeros(num_samples, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        active_logits = last_logits / max(temperature, 1e-4)
        log_probs = torch.log_softmax(active_logits, dim=-1)
        probs = torch.exp(log_probs)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        sampled = torch.where(done, torch.full_like(sampled, eos_id), sampled)
        sample_scores = log_probs.gather(1, sampled.unsqueeze(-1)).squeeze(-1)
        log_scores = torch.where(done, log_scores, log_scores + sample_scores)
        for row in range(num_samples):
            if not bool(done[row].item()):
                generated[row].append(int(sampled[row].item()))
        hidden_prev = hidden
        next_logits, next_hidden = model.step(sampled, hidden)
        if done.any():
            next_hidden[:, done, :] = hidden_prev[:, done, :]
            next_logits[done] = -1.0e9
            next_logits[done, eos_id] = 0.0
        hidden = next_hidden
        last_logits = next_logits
        done = done | (sampled == eos_id)

    candidates = [
        _candidate_result(
            "ancestral_best_of_k",
            task,
            tokenizer,
            generated[idx],
            float(log_scores[idx].item()),
            0.0,
            metadata={"num_samples": num_samples},
        )
        for idx in range(num_samples)
    ]
    chosen = max(candidates, key=_selection_key)
    chosen.runtime_seconds = time.perf_counter() - start
    return chosen


def beam_filter_decode(
    model: PrefixLanguageModel,
    tokenizer: LatticeTokenizer,
    task: BenchmarkTask,
    *,
    max_new_tokens: int,
    beam_size: int,
    device: torch.device,
) -> GenerationResult:
    start = time.perf_counter()
    eos_id = tokenizer.id_for("<eos>")
    prefix_ids = _prefix_ids(tokenizer, task.source_text)
    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    last_logits, hidden = model.warm_start(prefix_tensor)

    beams = [([], 0.0, hidden, last_logits, False)]
    for _ in range(max_new_tokens):
        next_beams = []
        candidate_tokens: list[int] = []
        candidate_scores: list[float] = []
        candidate_prefixes: list[list[int]] = []
        candidate_done: list[bool] = []
        candidate_hidden: list[torch.Tensor] = []
        for tokens, score, beam_hidden, beam_logits, done in beams:
            if done:
                next_beams.append((tokens, score, beam_hidden, beam_logits, done))
                continue
            log_probs = torch.log_softmax(beam_logits, dim=-1)
            top_scores, top_ids = torch.topk(log_probs[0], k=min(beam_size, log_probs.shape[-1]))
            for token_score, token_id in zip(top_scores.tolist(), top_ids.tolist()):
                candidate_tokens.append(int(token_id))
                candidate_scores.append(score + float(token_score))
                candidate_prefixes.append(tokens + [int(token_id)])
                candidate_done.append(int(token_id) == eos_id)
                candidate_hidden.append(beam_hidden)

        if candidate_tokens:
            token_tensor = torch.tensor(candidate_tokens, dtype=torch.long, device=device)
            hidden_tensor = torch.cat(candidate_hidden, dim=1)
            batched_logits, batched_hidden = model.step(token_tensor, hidden_tensor)
            for idx, token_id in enumerate(candidate_tokens):
                next_beams.append(
                    (
                        candidate_prefixes[idx],
                        candidate_scores[idx],
                        batched_hidden[:, idx : idx + 1, :].contiguous(),
                        batched_logits[idx : idx + 1].contiguous(),
                        token_id == eos_id,
                    )
                )
        beams = sorted(next_beams, key=lambda item: item[1], reverse=True)[:beam_size]
        if all(done for _, _, _, _, done in beams):
            break

    candidates = [
        _candidate_result(
            "beam_filter",
            task,
            tokenizer,
            tokens,
            score,
            0.0,
            metadata={"beam_size": beam_size},
        )
        for tokens, score, _, _, _ in beams
    ]
    chosen = max(candidates, key=_selection_key)
    chosen.runtime_seconds = time.perf_counter() - start
    return chosen


def twisted_smc_decode(
    model: PrefixLanguageModel,
    tokenizer: LatticeTokenizer,
    task: BenchmarkTask,
    *,
    max_new_tokens: int,
    particles: int,
    lambda_weight: float,
    twist_scale: float,
    ess_threshold: float,
    split_interval: int,
    elite_fraction: float,
    temperature: float,
    device: torch.device,
) -> GenerationResult:
    if particles <= 0:
        raise ValueError("particles must be positive")
    if not 0.0 < ess_threshold <= 1.0:
        raise ValueError("ess_threshold must be in (0, 1]")
    if not 0.0 < elite_fraction <= 1.0:
        raise ValueError("elite_fraction must be in (0, 1]")
    start = time.perf_counter()
    eos_id = tokenizer.id_for("<eos>")
    prefix_ids = _prefix_ids(tokenizer, task.source_text)
    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device).repeat(particles, 1)
    last_logits, hidden = model.warm_start(prefix_tensor)

    token_surfaces = [tokenizer.token_surface(token_id) for token_id in range(tokenizer.vocab_size)]
    automaton = SurfaceProductAutomaton.from_phrases(task.required_phrases, token_surfaces)
    transitions = automaton.transitions.to(device)
    distances = automaton.distances.to(device)
    accepting = automaton.accepting.to(device)

    states = torch.zeros(particles, dtype=torch.long, device=device)
    done = torch.zeros(particles, dtype=torch.bool, device=device)
    log_weights = torch.zeros(particles, dtype=torch.float32, device=device)
    trajectories = torch.full((particles, max_new_tokens), eos_id, dtype=torch.long, device=device)
    resamples = 0
    mean_ess = 0.0
    steps_run = 0

    def normalized(logw: torch.Tensor) -> torch.Tensor:
        anchor = torch.max(logw)
        weights = torch.exp(logw - anchor)
        return weights / torch.clamp(weights.sum(), min=1e-8)

    for step in range(max_new_tokens):
        steps_run = step + 1
        current_distance = distances.index_select(0, states)
        next_states = transitions.index_select(0, states)
        next_distance = distances.index_select(0, next_states.reshape(-1)).reshape(particles, -1)
        progress = current_distance.unsqueeze(-1) - next_distance
        guidance = twist_scale * progress

        scaled_logits = last_logits / max(temperature, 1e-4)
        base_logprob = torch.log_softmax(scaled_logits, dim=-1)
        proposal_logprob = torch.log_softmax(scaled_logits + guidance, dim=-1)

        sampled = torch.full((particles,), eos_id, dtype=torch.long, device=device)
        if (~done).any():
            active = ~done
            sampled_active = torch.multinomial(torch.exp(proposal_logprob[active]), num_samples=1).squeeze(-1)
            sampled[active] = sampled_active

        sampled_next_state = next_states.gather(1, sampled.unsqueeze(-1)).squeeze(-1)
        delta = progress.gather(1, sampled.unsqueeze(-1)).squeeze(-1)
        base_lp = base_logprob.gather(1, sampled.unsqueeze(-1)).squeeze(-1)
        prop_lp = proposal_logprob.gather(1, sampled.unsqueeze(-1)).squeeze(-1)
        log_weights = torch.where(done, log_weights, log_weights + base_lp + lambda_weight * delta - prop_lp)

        states = torch.where(done, states, sampled_next_state)
        trajectories[:, step] = sampled

        hidden_prev = hidden
        next_logits, next_hidden = model.step(sampled, hidden)
        if done.any():
            next_hidden[:, done, :] = hidden_prev[:, done, :]
            next_logits[done] = -1.0e9
            next_logits[done, eos_id] = 0.0
        hidden = next_hidden
        last_logits = next_logits
        done = done | (sampled == eos_id)

        probs = normalized(log_weights)
        ess = float(1.0 / torch.sum(probs * probs).item())
        mean_ess += ess
        if ess < ess_threshold * particles:
            indices = torch.multinomial(probs, num_samples=particles, replacement=True)
            states = states[indices]
            done = done[indices]
            log_weights = torch.zeros_like(log_weights)
            trajectories = trajectories[indices]
            hidden = hidden[:, indices, :]
            last_logits = last_logits[indices]
            resamples += 1

        if split_interval > 0 and (step + 1) % split_interval == 0 and step + 1 < max_new_tokens:
            score = log_weights - lambda_weight * distances.index_select(0, states)
            elite_k = max(1, int(math.ceil(particles * elite_fraction)))
            _, elite_idx = torch.topk(score, k=elite_k)
            pick = elite_idx[torch.randint(0, elite_k, (particles,), device=device)]
            states = states[pick]
            done = done[pick]
            trajectories = trajectories[pick]
            hidden = hidden[:, pick, :]
            last_logits = last_logits[pick]
            log_weights = torch.zeros_like(log_weights)

        if done.all():
            break

    probs = normalized(log_weights)
    accepting_mask = accepting.index_select(0, states)
    candidate_indices = torch.where(accepting_mask)[0]
    if len(candidate_indices) == 0:
        candidate_indices = torch.arange(particles, device=device)
        candidate_distances = distances.index_select(0, states).index_select(0, candidate_indices)
        best_distance = torch.min(candidate_distances)
        candidate_indices = candidate_indices[candidate_distances == best_distance]
    chosen_idx = int(candidate_indices[torch.argmax(log_weights.index_select(0, candidate_indices))].item())

    metadata = {
        "particles": particles,
        "lambda_weight": lambda_weight,
        "twist_scale": twist_scale,
        "resamples": resamples,
        "mean_ess": mean_ess / max(1, steps_run),
        "acceptance_mass": float(probs[accepting_mask].sum().item()),
        "steps_run": steps_run,
    }
    result = _candidate_result(
        "twisted_smc",
        task,
        tokenizer,
        trajectories[chosen_idx].tolist(),
        float(log_weights[chosen_idx].item()),
        time.perf_counter() - start,
        metadata=metadata,
    )
    return result
