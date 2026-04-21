from __future__ import annotations

from collections import Counter


def _tokenize(text: str) -> list[str]:
    return [token for token in text.strip().split() if token]


def rouge_l_f1(candidate: str, references: list[str]) -> float:
    cand = _tokenize(candidate)
    if not cand or not references:
        return 0.0

    def lcs_length(a: list[str], b: list[str]) -> int:
        table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i, token_a in enumerate(a, start=1):
            for j, token_b in enumerate(b, start=1):
                if token_a == token_b:
                    table[i][j] = table[i - 1][j - 1] + 1
                else:
                    table[i][j] = max(table[i - 1][j], table[i][j - 1])
        return table[-1][-1]

    best = 0.0
    for reference in references:
        ref = _tokenize(reference)
        if not ref:
            continue
        lcs = lcs_length(cand, ref)
        precision = lcs / max(1, len(cand))
        recall = lcs / max(1, len(ref))
        if precision + recall == 0:
            continue
        best = max(best, 2.0 * precision * recall / (precision + recall))
    return best


def token_f1(candidate: str, references: list[str]) -> float:
    cand = Counter(_tokenize(candidate))
    if not cand or not references:
        return 0.0
    best = 0.0
    for reference in references:
        ref = Counter(_tokenize(reference))
        overlap = sum((cand & ref).values())
        precision = overlap / max(1, sum(cand.values()))
        recall = overlap / max(1, sum(ref.values()))
        if precision + recall == 0:
            continue
        best = max(best, 2.0 * precision * recall / (precision + recall))
    return best
