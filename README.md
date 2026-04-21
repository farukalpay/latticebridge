# LatticeBridge

![LatticeBridge banner](assets/latticebridge-banner.svg)

LatticeBridge is a compact implementation of **rare-event inference for structured sequence generation**. It targets the regime where a conditional autoregressive model can produce fluent text, but assigns low probability to continuations that satisfy several input-derived anchors simultaneously. The implementation combines:

- a compact prefix language model trained for structured-to-text tasks
- **surface automata** compiled directly from instance-provided anchors
- **twisted sequential Monte Carlo** with resampling and multilevel splitting
- an MPS-compatible synthetic scale lab for controlled rare-event workloads

Constraint handling is instance-driven. The decoder uses source-provided anchors and does not depend on global topic inventories or label maps.

Repository: <https://github.com/farukalpay/latticebridge>

## Method

LatticeBridge treats constrained structured decoding as a sequential rare-event inference problem:

1. serialize the structured source into a prefix
2. compile input-derived anchors into a surface-level product automaton
3. decode with a twisted SMC bridge that rewards progress toward acceptance
4. report coverage, overlap quality, runtime, and particle diagnostics

## Current Results

The validation benchmark in `results/benchmark_core_validation/validation_summary.json` covers 996 attainable constraint tasks: 500 CommonGen examples and 496 E2E NLG examples. All four decoders use the same compact prefix model and the same input-derived surface anchors. The benchmark runs on Apple Metal (`mps`) with beam size 6, best-of-16 ancestral sampling, 96 Twisted SMC particles, and a 64-token generation cap.

### CommonGen

| Method | Success | Coverage | ROUGE-L | Runtime (s) |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 0.000 | 0.049 | **0.252** | 0.018 |
| Beam Filter | 0.000 | 0.041 | 0.223 | 0.058 |
| Best-of-16 Ancestral | 0.000 | 0.023 | 0.179 | 0.605 |
| Twisted SMC | **0.668** | **0.869** | 0.220 | 0.348 |

### E2E NLG

| Method | Success | Coverage | ROUGE-L | Runtime (s) |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 0.111 | 0.662 | 0.460 | 0.059 |
| Beam Filter | 0.139 | 0.676 | **0.476** | 0.212 |
| Best-of-16 Ancestral | 0.407 | 0.709 | 0.381 | 0.686 |
| Twisted SMC | **0.427** | **0.782** | 0.226 | 0.314 |

Interpretation:

- Twisted SMC materially improves exact anchor satisfaction and average coverage under the same checkpoint.
- On CommonGen, it is the only tested method with nonzero exact success across the 500-task validation subset.
- On E2E NLG, it gives the highest coverage and success while running faster than best-of-16 ancestral sampling.
- ROUGE-L should be read next to coverage; the bridge changes the operating point rather than optimizing a single metric.

Generated figures:

- ![CommonGen benchmark](results/figures/common_gen_benchmark.png)
- ![E2E benchmark](results/figures/e2e_nlg_benchmark.png)
- ![Coverage-runtime frontier](results/figures/coverage_runtime_frontier.png)
- ![Training curve](results/figures/training_curve.png)
- ![MPS scale probe](results/figures/mps_scale_probe.png)

## MPS Scale Probes

`latticebridge_synthetic_lab.py` runs a neutral synthetic rare-event systems lab. The following MPS runs were executed:

| Scenario | Exact Accept | Relaxed Accept | Mean ESS | Runtime (s) |
| --- | ---: | ---: | ---: | ---: |
| `conflict_probe`, `96/256/12/8192` | 0.0386 | 0.3419 | 5189.0 | 3.49 |
| `scale_probe`, `128/320/16/16384` | 0.0122 | 0.2410 | 10348.7 | 19.38 |
| `scale_probe`, `160/384/24/24576` | 0.000854 | 0.1690 | 15843.5 | 49.43 |

The heavy run shows barrier growth: runtime and ESS remain tractable on MPS while exact satisfaction decreases by nearly two orders of magnitude.

## Quick Start

### 1. Prepare the benchmark corpora

```bash
PYTHONPATH=src python3 -m latticebridge.cli prepare \
  --manifest data/manifests/core_benchmark.json \
  --cache-root data/cache \
  --processed-root data/cache/processed_core \
  --tokenizer-corpus data/cache/tokenizer_corpus_core.txt \
  --dotenv .env
```

`core_benchmark.json` stages the CommonGen and E2E corpora used by the reported validation benchmark. `datasets.json` also defines a WikiBio adapter for table-to-text experiments. Public downloads do not require a token, but `HF_TOKEN` can be supplied through `.env` for Hugging Face-hosted assets.

### 2. Train the prefix language model

```bash
PYTHONPATH=src python3 -m latticebridge.cli train \
  --processed-root data/cache/processed_core \
  --tokenizer-corpus data/cache/tokenizer_corpus_core.txt \
  --tokenizer-path artifacts/tokenizer_core.json \
  --checkpoint-dir artifacts/checkpoints_core \
  --device mps \
  --epochs 4 \
  --batch-size 48 \
  --max-tokens 144
```

### 3. Run the validation benchmark

```bash
PYTHONPATH=src python3 -m latticebridge.cli benchmark \
  --processed-root data/cache/processed_core \
  --tokenizer-path artifacts/tokenizer_core.json \
  --checkpoint artifacts/checkpoints_core/best.pt \
  --output-dir results/benchmark_core_validation \
  --split validation \
  --per-dataset-limit 500 \
  --max-new-tokens 64 \
  --max-anchors 3 \
  --min-anchors 2 \
  --beam-size 6 \
  --num-samples 16 \
  --particles 96 \
  --lambda-weight 2.0 \
  --twist-scale 2.0 \
  --ess-threshold 0.5 \
  --split-interval 12 \
  --elite-fraction 0.2 \
  --sample-temperature 0.95 \
  --smc-temperature 0.9 \
  --seed 13 \
  --device mps \
  --log-interval 50
```

### 4. Render summary figures

```bash
PYTHONPATH=src python3 -m latticebridge.cli figures \
  --summary-path results/benchmark_core_validation/validation_summary.json \
  --output-dir results/figures \
  --train-report artifacts/checkpoints_core/train_report.json
```

### 5. Run the stress probes

```bash
python3 latticebridge_synthetic_lab.py \
  --scenario conflict_probe \
  --seq-len 96 \
  --rank 256 \
  --replicas 12 \
  --particles 8192 \
  --device mps \
  --log-interval 16
```

```bash
python3 latticebridge_synthetic_lab.py \
  --scenario scale_probe \
  --seq-len 128 \
  --rank 320 \
  --replicas 16 \
  --particles 16384 \
  --device mps \
  --log-interval 16
```

```bash
python3 latticebridge_synthetic_lab.py \
  --scenario scale_probe \
  --seq-len 160 \
  --rank 384 \
  --replicas 24 \
  --particles 24576 \
  --device mps \
  --log-interval 16
```

## Repository Layout

```text
assets/                     banner and README visuals
data/manifests/             benchmark and dataset manifests
paper/                      manuscript source and bibliography
results/                    benchmark outputs, figures, and scale probe logs
src/latticebridge/          library code
  benchmarks/               generation runners and benchmark tasks
  constraints/              token and surface automata
  data/                     downloaders and dataset adapters
  experiments/              prepare/train/benchmark/figure commands
  lab/                      synthetic rare-event scale probes
  models/                   tokenizer and prefix LM
```

## Design Notes

- **Constraint representation:** phrases are compiled into surface automata, not keyword classes.
- **Control signal:** Twisted SMC uses distance-to-acceptance progress, resampling, and splitting. There is no hand-authored topic logic.
- **Separation of concerns:** dataset adapters are thin and schema-aware; the inference library is reusable and dataset-agnostic.
- **Apple Metal posture:** all reported training and stress runs target `mps`.

## Validation Artifacts

- `results/benchmark_core_validation/validation_summary.json` stores the aggregate benchmark metrics.
- `results/benchmark_core_validation/validation_tasks.json` stores benchmark task counts and anchor counts.
- `results/benchmark_core_validation/validation_config.json` stores the decoding configuration used for the reported run.
- `results/figures/` stores the rendered benchmark and training figures used in the paper.

## Paper

The manuscript source lives under `paper/`. The compiled PDF is available at [`paper/latticebridge.pdf`](paper/latticebridge.pdf) and currently renders to 22 pages with 25 bibliography entries.

To rebuild it:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error latticebridge.tex
```
