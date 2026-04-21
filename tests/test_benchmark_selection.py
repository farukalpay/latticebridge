from latticebridge.benchmarks.generation import build_benchmark_tasks
from latticebridge.constraints.phrase_automaton import SurfaceProductAutomaton
from latticebridge.data.records import DatasetRecord


def test_anchor_selection_uses_empirical_information_not_adapter_order() -> None:
    records = [
        DatasetRecord(
            dataset_name="toy",
            split="validation",
            example_id="0",
            source_text="x0",
            target_text="common medium rare",
            candidate_phrases=["common", "rare", "medium"],
            references=["common medium rare"],
        ),
        DatasetRecord(
            dataset_name="toy",
            split="validation",
            example_id="1",
            source_text="x1",
            target_text="common medium",
            candidate_phrases=["common", "medium"],
            references=["common medium"],
        ),
        DatasetRecord(
            dataset_name="toy",
            split="validation",
            example_id="2",
            source_text="x2",
            target_text="common filler",
            candidate_phrases=["common", "filler"],
            references=["common filler"],
        ),
    ]

    tasks = build_benchmark_tasks(records, tokenizer=None, max_anchors=2, min_anchors=2)

    assert tasks[0].required_phrases == ["rare", "medium"]


def test_surface_automaton_accepts_multi_token_surface_match() -> None:
    automaton = SurfaceProductAutomaton.from_phrases(
        ["alpha beta"],
        ["al", "pha", " ", "beta", "gamma"],
    )

    state = 0
    for token_id in (0, 1, 2, 3):
        state = int(automaton.transitions[state, token_id].item())

    assert bool(automaton.accepting[state].item()) is True
    assert float(automaton.distances[state].item()) == 0.0
