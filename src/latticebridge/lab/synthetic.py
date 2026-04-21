from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass

import torch


TOKENS = {
    "PAD": 0,
    "ALPHA": 1,
    "BETA": 2,
    "GAMMA": 3,
    "DELTA": 4,
    "EPSILON": 5,
    "THETA": 6,
    "IOTA": 7,
    "KAPPA": 8,
    "LAMBDA": 9,
    "MU": 10,
    "NU": 11,
    "XI": 12,
    "OMICRON": 13,
    "PI": 14,
    "RHO": 15,
    "SIGMA": 16,
    "TAU": 17,
    "UPSILON": 18,
    "PHI": 19
}
TOKEN_COUNT = len(TOKENS)


@dataclass
class Scenario:
    name: str
    required_phrases: list[list[int]]
    forbidden_phrases: list[list[int]]
    preferred_tokens: dict[int, float]
    draft_bias: dict[int, float]
    lambda_max: float


class PhraseDFA:
    def __init__(self, phrase: list[int], mode: str):
        self.phrase = phrase
        self.mode = mode
        self.length = len(phrase)
        self.pi = self._prefix_function(phrase)
        self.state_count = self.length + 1
        self.transitions = torch.zeros((self.state_count, TOKEN_COUNT), dtype=torch.long)
        for state in range(self.state_count):
            for token_id in range(TOKEN_COUNT):
                self.transitions[state, token_id] = self._advance(state, token_id)

    def _advance(self, state: int, token_id: int) -> int:
        if state == self.length:
            return self.length
        cursor = state
        while cursor > 0 and token_id != self.phrase[cursor]:
            cursor = self.pi[cursor - 1]
        if token_id == self.phrase[cursor]:
            cursor += 1
        return min(cursor, self.length)

    @staticmethod
    def _prefix_function(seq: list[int]) -> list[int]:
        table = [0] * len(seq)
        cursor = 0
        for idx in range(1, len(seq)):
            while cursor > 0 and seq[idx] != seq[cursor]:
                cursor = table[cursor - 1]
            if seq[idx] == seq[cursor]:
                cursor += 1
            table[idx] = cursor
        return table


class ProductAutomaton:
    def __init__(self, required: list[list[int]], forbidden: list[list[int]]):
        dfas = [PhraseDFA(phrase, "required") for phrase in required] + [PhraseDFA(phrase, "forbidden") for phrase in forbidden]
        radices = []
        state_count = 1
        for dfa in dfas:
            radices.append(state_count)
            state_count *= dfa.state_count
        self.transitions = torch.zeros((state_count, TOKEN_COUNT), dtype=torch.long)
        self.accepting = torch.zeros((state_count,), dtype=torch.bool)
        self.distance = torch.zeros((state_count,), dtype=torch.float32)
        for state in range(state_count):
            tuple_state = []
            remaining = state
            accept = True
            distance = 0.0
            for idx, dfa in enumerate(dfas):
                local_state = remaining % dfa.state_count
                remaining //= dfa.state_count
                tuple_state.append(local_state)
                if idx < len(required):
                    accept = accept and (local_state == dfa.length)
                    distance += float(max(0, dfa.length - local_state))
                else:
                    accept = accept and (local_state != dfa.length)
                    if local_state == dfa.length:
                        distance += 1.0
            self.accepting[state] = accept
            self.distance[state] = distance
            for token_id in range(TOKEN_COUNT):
                next_state = 0
                for idx, dfa in enumerate(dfas):
                    next_state += int(dfa.transitions[tuple_state[idx], token_id]) * radices[idx]
                self.transitions[state, token_id] = next_state


def build_scenario(name: str) -> Scenario:
    if name == "conflict_probe":
        return Scenario(
            name=name,
            required_phrases=[[TOKENS["GAMMA"]], [TOKENS["THETA"], TOKENS["IOTA"], TOKENS["KAPPA"]], [TOKENS["DELTA"]]],
            forbidden_phrases=[[TOKENS["LAMBDA"]], [TOKENS["MU"]], [TOKENS["NU"]]],
            preferred_tokens={TOKENS["GAMMA"]: 1.0, TOKENS["DELTA"]: 1.0, TOKENS["THETA"]: 0.7, TOKENS["IOTA"]: 0.7, TOKENS["KAPPA"]: 0.7},
            draft_bias={TOKENS["ALPHA"]: 0.6, TOKENS["GAMMA"]: 1.1, TOKENS["DELTA"]: 0.8, TOKENS["MU"]: 0.4},
            lambda_max=3.0,
        )
    if name == "structured_inference":
        return Scenario(
            name=name,
            required_phrases=[[TOKENS["DELTA"]], [TOKENS["GAMMA"]], [TOKENS["THETA"], TOKENS["IOTA"], TOKENS["KAPPA"]], [TOKENS["PI"]]],
            forbidden_phrases=[[TOKENS["LAMBDA"]], [TOKENS["ALPHA"]]],
            preferred_tokens={TOKENS["DELTA"]: 1.0, TOKENS["GAMMA"]: 1.0, TOKENS["PI"]: 1.0, TOKENS["RHO"]: 0.8, TOKENS["SIGMA"]: 0.7},
            draft_bias={TOKENS["DELTA"]: 1.0, TOKENS["GAMMA"]: 1.1, TOKENS["PI"]: 0.8, TOKENS["SIGMA"]: 0.6},
            lambda_max=2.5,
        )
    if name == "scale_probe":
        return Scenario(
            name=name,
            required_phrases=[[TOKENS["DELTA"]], [TOKENS["GAMMA"]], [TOKENS["THETA"], TOKENS["IOTA"], TOKENS["KAPPA"]], [TOKENS["PI"]], [TOKENS["RHO"]]],
            forbidden_phrases=[[TOKENS["LAMBDA"]], [TOKENS["MU"]], [TOKENS["NU"]], [TOKENS["ALPHA"]]],
            preferred_tokens={TOKENS["DELTA"]: 1.0, TOKENS["GAMMA"]: 1.0, TOKENS["PI"]: 1.0, TOKENS["RHO"]: 1.0, TOKENS["SIGMA"]: 0.7},
            draft_bias={TOKENS["DELTA"]: 1.1, TOKENS["GAMMA"]: 1.0, TOKENS["PI"]: 1.0, TOKENS["RHO"]: 1.0, TOKENS["SIGMA"]: 0.7},
            lambda_max=3.5,
        )
    raise ValueError(name)


def resolve_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name in {"mps", "cuda"}:
        print(f"warning: {name} unavailable, falling back to cpu")
    return torch.device("cpu")


def make_scores(args, scenario: Scenario, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    replica = torch.randn((args.replicas, args.rank), generator=generator)
    time_codes = torch.randn((args.seq_len, args.rank), generator=generator)
    token_codes = torch.randn((TOKEN_COUNT, args.rank), generator=generator)
    scores = torch.einsum("rk,tk,ck->rtc", replica, time_codes, token_codes) / math.sqrt(args.rank)
    scores += 0.2 * torch.randn((args.replicas, args.seq_len, TOKEN_COUNT), generator=generator)
    pref = torch.zeros((TOKEN_COUNT,), dtype=torch.float32)
    draft = torch.zeros((TOKEN_COUNT,), dtype=torch.float32)
    for token_id, weight in scenario.preferred_tokens.items():
        pref[token_id] = weight
    for token_id, weight in scenario.draft_bias.items():
        draft[token_id] = weight
    frac = torch.linspace(0.0, 1.0, args.seq_len)
    scores += pref.view(1, 1, TOKEN_COUNT)
    scores += args.draft_scale * (1.0 - frac).view(1, args.seq_len, 1) * draft.view(1, 1, TOKEN_COUNT)
    return torch.tanh(scores / 2.5) * 4.0


def run(args) -> dict[str, object]:
    start = time.perf_counter()
    scenario = build_scenario(args.scenario)
    device = resolve_device(args.device)
    automaton = ProductAutomaton(scenario.required_phrases, scenario.forbidden_phrases)
    transitions = automaton.transitions.to(device)
    accepting = automaton.accepting.to(device)
    distances = automaton.distance.to(device)
    scores = make_scores(args, scenario, device).to(device)

    state = torch.zeros((args.replicas, args.particles), dtype=torch.long, device=device)
    logw = torch.zeros((args.replicas, args.particles), dtype=torch.float32, device=device)
    resamples = torch.zeros((args.replicas,), dtype=torch.float32, device=device)
    mean_ess = []

    lambdas = torch.linspace(0.0, scenario.lambda_max if args.lambda_max <= 0 else args.lambda_max, args.replicas, device=device)
    for step in range(args.seq_len):
        step_scores = scores[:, step, :]
        next_states = transitions.index_select(0, state.reshape(-1)).reshape(args.replicas, args.particles, TOKEN_COUNT)
        current_dist = distances.index_select(0, state.reshape(-1)).reshape(args.replicas, args.particles)
        next_dist = distances.index_select(0, next_states.reshape(-1)).reshape(args.replicas, args.particles, TOKEN_COUNT)
        progress = current_dist.unsqueeze(-1) - next_dist
        proposal_logits = step_scores.unsqueeze(1) + args.twist_scale * progress - lambdas.view(args.replicas, 1, 1) * torch.clamp(next_dist - current_dist.unsqueeze(-1), min=0.0)
        proposal_logprob = torch.log_softmax(proposal_logits, dim=-1)
        sampled = torch.multinomial(torch.exp(proposal_logprob).reshape(args.replicas * args.particles, TOKEN_COUNT), num_samples=1).reshape(args.replicas, args.particles)
        sampled_logprob = proposal_logprob.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        sampled_score = step_scores.unsqueeze(1).expand(-1, args.particles, -1).gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        sampled_progress = progress.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        logw += sampled_score + lambdas.view(args.replicas, 1) * sampled_progress - sampled_logprob
        state = next_states.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        ess_values = []
        for replica_idx in range(args.replicas):
            anchor = torch.max(logw[replica_idx])
            probs = torch.exp(logw[replica_idx] - anchor)
            probs = probs / torch.clamp(probs.sum(), min=1e-8)
            ess = float(1.0 / torch.sum(probs * probs).item())
            ess_values.append(ess)
            if ess < args.ess_threshold * args.particles:
                indices = torch.multinomial(probs, num_samples=args.particles, replacement=True)
                state[replica_idx] = state[replica_idx, indices]
                logw[replica_idx].zero_()
                resamples[replica_idx] += 1.0
        mean_ess.append(sum(ess_values) / max(1, len(ess_values)))

        if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
            alive = float(accepting.index_select(0, state.reshape(-1)).reshape(args.replicas, args.particles).float().mean().item())
            print(f"[lab] t={step+1:3d}/{args.seq_len} alive={alive:0.4f} ess={mean_ess[-1]:0.1f}")

    hardest = int(torch.argmax(lambdas).item())
    anchor = torch.max(logw[hardest])
    probs = torch.exp(logw[hardest] - anchor)
    probs = probs / torch.clamp(probs.sum(), min=1e-8)
    accept = accepting.index_select(0, state[hardest])
    exact = float(torch.sum(probs * accept.float()).item())
    relaxed = float(torch.sum(probs * torch.exp(-0.7 * distances.index_select(0, state[hardest]))).item())
    return {
        "scenario": args.scenario,
        "exact_accept_estimate": exact,
        "relaxed_accept_estimate": relaxed,
        "mean_ess": sum(mean_ess) / max(1, len(mean_ess)),
        "resamples_mean": float(resamples.mean().item()),
        "elapsed": time.perf_counter() - start,
        "combined_states": int(transitions.shape[0]),
        "backend": device.type,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LatticeBridge synthetic rare-event lab")
    parser.add_argument("--scenario", default="conflict_probe", choices=["conflict_probe", "structured_inference", "scale_probe"])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--replicas", type=int, default=12)
    parser.add_argument("--particles", type=int, default=8192)
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--draft-scale", type=float, default=0.9)
    parser.add_argument("--twist-scale", type=float, default=1.4)
    parser.add_argument("--lambda-max", type=float, default=0.0)
    parser.add_argument("--ess-threshold", type=float, default=0.45)
    parser.add_argument("--log-interval", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    result = run(args)
    print("LatticeBridge Synthetic Lab")
    print("---------------------------")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}={value:.6f}")
        else:
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
