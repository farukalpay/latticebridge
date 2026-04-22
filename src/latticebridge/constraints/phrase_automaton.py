from __future__ import annotations

from dataclasses import dataclass

import torch


def _prefix_function(seq: list[int]) -> list[int]:
    pi = [0] * len(seq)
    j = 0
    for i in range(1, len(seq)):
        while j > 0 and seq[i] != seq[j]:
            j = pi[j - 1]
        if seq[i] == seq[j]:
            j += 1
        pi[i] = j
    return pi


class PhraseDFA:
    def __init__(self, phrase: list[int], vocab_size: int):
        self.phrase = phrase
        self.vocab_size = vocab_size
        self.length = len(phrase)
        self.pi = _prefix_function(phrase)
        self.state_count = self.length + 1
        self.transitions = torch.zeros((self.state_count, vocab_size), dtype=torch.long)
        for state in range(self.state_count):
            for token_id in range(vocab_size):
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


class SurfacePhraseDFA:
    def __init__(self, phrase: str):
        normalized = phrase.strip().lower()
        self.phrase = normalized
        self.symbols = list(normalized)
        self.length = len(self.symbols)
        self.pi = _prefix_function([ord(ch) for ch in self.symbols])
        self.state_count = self.length + 1

    def advance_text(self, state: int, text: str) -> int:
        if state == self.length:
            return self.length
        cursor = state
        for ch in text.lower():
            if cursor == self.length:
                return self.length
            while cursor > 0 and ch != self.symbols[cursor]:
                cursor = self.pi[cursor - 1]
            if ch == self.symbols[cursor]:
                cursor += 1
        return min(cursor, self.length)

    def transition_table(self, token_surfaces: list[str]) -> torch.Tensor:
        table = torch.zeros((self.state_count, len(token_surfaces)), dtype=torch.long)
        for state in range(self.state_count):
            for token_id, token_surface in enumerate(token_surfaces):
                table[state, token_id] = self.advance_text(state, token_surface)
        return table


@dataclass
class ProductAutomaton:
    transitions: torch.Tensor
    distances: torch.Tensor
    accepting: torch.Tensor

    @classmethod
    def from_phrases(cls, phrases: list[list[int]], vocab_size: int) -> "ProductAutomaton":
        if not phrases:
            transitions = torch.arange(vocab_size, dtype=torch.long).unsqueeze(0).repeat(1, 1)
            return cls(
                transitions=transitions,
                distances=torch.zeros(1, dtype=torch.float32),
                accepting=torch.ones(1, dtype=torch.bool),
            )

        dfas = [PhraseDFA(phrase, vocab_size) for phrase in phrases if phrase]
        radices: list[int] = []
        state_total = 1
        for dfa in dfas:
            radices.append(state_total)
            state_total *= dfa.state_count

        transitions = torch.zeros((state_total, vocab_size), dtype=torch.long)
        distances = torch.zeros((state_total,), dtype=torch.float32)
        accepting = torch.zeros((state_total,), dtype=torch.bool)

        for state in range(state_total):
            tuple_state: list[int] = []
            remaining = state
            accepted = True
            distance = 0
            for dfa in dfas:
                local_state = remaining % dfa.state_count
                remaining //= dfa.state_count
                tuple_state.append(local_state)
                accepted = accepted and (local_state == dfa.length)
                distance += max(0, dfa.length - local_state)
            accepting[state] = accepted
            distances[state] = float(distance)
            for token_id in range(vocab_size):
                next_state = 0
                for idx, dfa in enumerate(dfas):
                    next_state += int(dfa.transitions[tuple_state[idx], token_id]) * radices[idx]
                transitions[state, token_id] = next_state
        return cls(transitions=transitions, distances=distances, accepting=accepting)

    @property
    def state_count(self) -> int:
        return int(self.transitions.shape[0])


@dataclass
class SurfaceProductAutomaton:
    transitions: torch.Tensor
    distances: torch.Tensor
    accepting: torch.Tensor
    radices: tuple[int, ...]
    state_counts: tuple[int, ...]
    phrases: tuple[str, ...]
    lengths: tuple[int, ...]

    @classmethod
    def from_phrases(cls, phrases: list[str], token_surfaces: list[str]) -> "SurfaceProductAutomaton":
        surface_phrases = [SurfacePhraseDFA(phrase) for phrase in phrases if phrase.strip()]
        if not surface_phrases:
            return cls(
                transitions=torch.zeros((1, len(token_surfaces)), dtype=torch.long),
                distances=torch.zeros((1,), dtype=torch.float32),
                accepting=torch.ones((1,), dtype=torch.bool),
                radices=(),
                state_counts=(),
                phrases=(),
                lengths=(),
            )

        radices: list[int] = []
        state_total = 1
        state_counts: list[int] = []
        for dfa in surface_phrases:
            radices.append(state_total)
            state_total *= dfa.state_count
            state_counts.append(dfa.state_count)

        component_transitions = [dfa.transition_table(token_surfaces) for dfa in surface_phrases]

        state_ids = torch.arange(state_total, dtype=torch.long)
        transitions = torch.zeros((state_total, len(token_surfaces)), dtype=torch.long)
        distances = torch.zeros((state_total,), dtype=torch.float32)
        accepting = torch.ones((state_total,), dtype=torch.bool)

        local_states: list[torch.Tensor] = []
        for idx, dfa in enumerate(surface_phrases):
            local_state = torch.div(state_ids, radices[idx], rounding_mode="floor") % dfa.state_count
            local_states.append(local_state)
            accepting &= local_state == dfa.length
            distances += torch.clamp(torch.tensor(dfa.length, dtype=torch.long) - local_state, min=0).float()

        for idx, table in enumerate(component_transitions):
            transitions += table.index_select(0, local_states[idx]) * radices[idx]
        return cls(
            transitions=transitions,
            distances=distances,
            accepting=accepting,
            radices=tuple(radices),
            state_counts=tuple(state_counts),
            phrases=tuple(dfa.phrase for dfa in surface_phrases),
            lengths=tuple(dfa.length for dfa in surface_phrases),
        )

    def local_states(self, state_ids: torch.Tensor) -> list[torch.Tensor]:
        if not self.radices:
            return []
        locals_: list[torch.Tensor] = []
        for radix, state_count in zip(self.radices, self.state_counts):
            local_state = torch.div(state_ids, radix, rounding_mode="floor") % state_count
            locals_.append(local_state)
        return locals_
