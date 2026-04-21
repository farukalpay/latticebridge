from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

from latticebridge.experiments.evaluate import run_benchmarks
from latticebridge.experiments.figures import render_summary_figures
from latticebridge.experiments.prepare import prepare_workspace
from latticebridge.experiments.train import load_trained_model, train_model
from latticebridge.lab.synthetic import main as synthetic_main
from latticebridge.models.tokenizer import LatticeTokenizer
from latticebridge.utils.env import resolve_hf_token
from latticebridge.utils.seed import seed_everything


def _device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LatticeBridge CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--manifest", default="data/manifests/datasets.json")
    prep.add_argument("--cache-root", default="data/cache")
    prep.add_argument("--processed-root", default="data/cache/processed")
    prep.add_argument("--tokenizer-corpus", default="data/cache/tokenizer_corpus.txt")
    prep.add_argument("--dotenv", default="")

    train = sub.add_parser("train")
    train.add_argument("--processed-root", default="data/cache/processed")
    train.add_argument("--tokenizer-corpus", default="data/cache/tokenizer_corpus.txt")
    train.add_argument("--tokenizer-path", default="artifacts/tokenizer.json")
    train.add_argument("--checkpoint-dir", default="artifacts/checkpoints")
    train.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    train.add_argument("--epochs", type=int, default=6)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--max-tokens", type=int, default=160)
    train.add_argument("--seed", type=int, default=13)

    bench = sub.add_parser("benchmark")
    bench.add_argument("--processed-root", default="data/cache/processed")
    bench.add_argument("--tokenizer-path", default="artifacts/tokenizer.json")
    bench.add_argument("--checkpoint", default="artifacts/checkpoints/best.pt")
    bench.add_argument("--output-dir", default="results/benchmark")
    bench.add_argument("--split", default="test", choices=["validation", "test"])
    bench.add_argument("--per-dataset-limit", type=int, default=120)
    bench.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    bench.add_argument("--seed", type=int, default=13)
    bench.add_argument("--max-new-tokens", type=int, default=72)
    bench.add_argument("--max-anchors", type=int, default=3, help="Use 0 to keep all attested adapter phrases.")
    bench.add_argument("--min-anchors", type=int, default=2)
    bench.add_argument("--beam-size", type=int, default=6)
    bench.add_argument("--num-samples", type=int, default=16)
    bench.add_argument("--particles", type=int, default=96)
    bench.add_argument("--lambda-weight", type=float, default=2.0)
    bench.add_argument("--twist-scale", type=float, default=2.0)
    bench.add_argument("--sample-temperature", type=float, default=0.95)
    bench.add_argument("--smc-temperature", type=float, default=0.9)
    bench.add_argument("--ess-threshold", type=float, default=0.5)
    bench.add_argument("--split-interval", type=int, default=12)
    bench.add_argument("--elite-fraction", type=float, default=0.2)

    figs = sub.add_parser("figures")
    figs.add_argument("--summary-path", default="results/benchmark/test_summary.json")
    figs.add_argument("--output-dir", default="results/figures")
    figs.add_argument("--train-report", default="")

    sub.add_parser("synthetic-lab")
    return parser


def main() -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args()

    if args.command == "synthetic-lab":
        sys.argv = [sys.argv[0], *unknown]
        synthetic_main()
        return

    if args.command == "prepare":
        summary = prepare_workspace(
            manifest_path=Path(args.manifest),
            cache_root=Path(args.cache_root),
            processed_root=Path(args.processed_root),
            tokenizer_corpus_path=Path(args.tokenizer_corpus),
            hf_token=resolve_hf_token(args.dotenv or None),
        )
        print(summary)
        return

    if args.command == "train":
        seed_everything(args.seed)
        tokenizer_path = Path(args.tokenizer_path)
        if not tokenizer_path.exists():
            LatticeTokenizer.train([args.tokenizer_corpus], tokenizer_path)
        report = train_model(
            processed_root=Path(args.processed_root),
            tokenizer_path=tokenizer_path,
            checkpoint_dir=Path(args.checkpoint_dir),
            device=_device(args.device),
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
        )
        print(report)
        return

    if args.command == "benchmark":
        seed_everything(args.seed)
        device = _device(args.device)
        model = load_trained_model(Path(args.checkpoint), device)
        result = run_benchmarks(
            processed_root=Path(args.processed_root),
            tokenizer_path=Path(args.tokenizer_path),
            model=model,
            output_dir=Path(args.output_dir),
            device=device,
            split=args.split,
            per_dataset_limit=args.per_dataset_limit,
            max_new_tokens=args.max_new_tokens,
            max_anchors=args.max_anchors,
            min_anchors=args.min_anchors,
            beam_size=args.beam_size,
            num_samples=args.num_samples,
            particles=args.particles,
            lambda_weight=args.lambda_weight,
            twist_scale=args.twist_scale,
            sample_temperature=args.sample_temperature,
            smc_temperature=args.smc_temperature,
            ess_threshold=args.ess_threshold,
            split_interval=args.split_interval,
            elite_fraction=args.elite_fraction,
            random_seed=args.seed,
        )
        print(result["summary"])
        return

    if args.command == "figures":
        train_report_path = Path(args.train_report) if args.train_report else None
        figure_paths = render_summary_figures(
            Path(args.summary_path),
            Path(args.output_dir),
            train_report_path=train_report_path,
        )
        print({"figures": figure_paths})


if __name__ == "__main__":
    main()
