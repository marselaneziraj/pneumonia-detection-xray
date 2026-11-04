"""Command-line interface: `pneumonia {prepare,train,evaluate,gradcam,report,make-synthetic}`."""

from __future__ import annotations

import argparse
import sys

DEFAULT_CONFIG = "configs/densenet121.yaml"
DEFAULT_EVAL_CONFIGS = [
    "configs/densenet121.yaml",
    "configs/resnet18.yaml",
    "configs/resnet18_scratch.yaml",
]


def _overrides(args) -> dict:
    o: dict = {}
    if getattr(args, "data_dir", None):
        o.setdefault("data", {})["data_dir"] = args.data_dir
    return o


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pneumonia", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare", help="index, audit, patient-grouped split, manifests, cache")
    sp.add_argument("--config", default=DEFAULT_CONFIG)
    sp.add_argument("--data-dir", help="override data.data_dir")
    sp.add_argument("--skip-audit", action="store_true", help="skip decoding every image")
    sp.add_argument("--skip-cache", action="store_true")

    st = sub.add_parser("train", help="two-phase fine-tuning")
    st.add_argument("--config", default=DEFAULT_CONFIG)
    st.add_argument("--data-dir")
    st.add_argument("--limit-train-batches", type=int, help="smoke mode: cap batches per epoch")
    st.add_argument("--limit-val-batches", type=int)
    st.add_argument("--device", help="auto | cpu | cuda | mps")

    se = sub.add_parser("evaluate", help="score val + held-out test, baselines, CIs, figures")
    se.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_EVAL_CONFIGS,
        help="first config is the primary (shipped) model",
    )
    se.add_argument("--device")
    se.add_argument(
        "--rescore",
        action="store_true",
        help="re-score test even if cached predictions match the checkpoint",
    )

    sg = sub.add_parser("gradcam", help="Grad-CAM panel + explanation audit sheet")
    sg.add_argument("--config", default=DEFAULT_CONFIG)
    sg.add_argument("--per-case", type=int, default=3)
    sg.add_argument("--device")

    sr = sub.add_parser("report", help="inject tables from results/ into README/report/model card")
    sr.add_argument("--results-dir", default="results")
    sr.add_argument("--splits-dir", default="splits")

    sm = sub.add_parser("make-synthetic", help="write a tiny fake dataset for smoke tests")
    sm.add_argument("--out", default="data_synthetic")
    sm.add_argument("--n-train", type=int, default=40)
    sm.add_argument("--n-test", type=int, default=12)
    sm.add_argument("--size", type=int, default=128)

    args = p.parse_args(argv)

    if args.cmd == "make-synthetic":
        from .synthetic import make_synthetic_tree

        root = make_synthetic_tree(
            args.out, {"train": args.n_train, "val": 4, "test": args.n_test}, size=args.size
        )
        print(f"synthetic tree written to {root} (NOT real data)")
        return 0
    if args.cmd == "report":
        from .report import build_report

        build_report(args.results_dir, args.splits_dir)
        return 0

    from .config import load_config
    from .indexing import DatasetNotFoundError

    try:
        if args.cmd == "prepare":
            from .prepare import prepare

            prepare(load_config(args.config, _overrides(args)), args.skip_audit, args.skip_cache)
        elif args.cmd == "train":
            from .train import train

            train(
                load_config(args.config, _overrides(args)),
                args.limit_train_batches,
                args.limit_val_batches,
                args.device,
            )
        elif args.cmd == "evaluate":
            from .evaluate import evaluate

            evaluate(args.configs, args.device, args.rescore)
        elif args.cmd == "gradcam":
            from .gradcam import run_gradcam

            run_gradcam(load_config(args.config), args.per_case, args.device)
    except DatasetNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
