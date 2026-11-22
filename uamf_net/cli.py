"""Command-line entry points; the synthetic demo needs only Python's standard library."""

import argparse
import json

from .config import MODALITIES, load_config, write_json
from .data import read_manifest


def main():
    parser = argparse.ArgumentParser(description="UAMF-Net reference implementation")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "validate-manifest", "train", "evaluate", "tune"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/reference.json")
        if name in ("validate-manifest", "train", "evaluate"):
            command.add_argument("--manifest", required=True)
            command.add_argument("--data-root", required=True)
        if name in ("train", "evaluate"):
            command.add_argument("--checkpoint-dir", required=True)
            command.add_argument("--device", default="auto")
        if name == "train":
            command.add_argument("--modality", choices=(*MODALITIES, "all"), default="all")
        elif name == "evaluate":
            command.add_argument("--split", choices=("val", "test"), required=True)
            command.add_argument("--output", required=True)
        elif name == "demo":
            command.add_argument("--output")
        elif name == "validate-manifest":
            command.add_argument("--schema-only", action="store_true", help="Do not check local image files")
        elif name == "tune":
            command.add_argument("--predictions", required=True)
            command.add_argument("--output-config", required=True)
            command.add_argument("--report", required=True)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "demo":
            from .demo import run_demo
            result = run_demo(config)
            if args.output:
                write_json(args.output, result)
        elif args.command == "tune":
            from .tuning import tune_policy
            selected, report = tune_policy(config, args.predictions)
            write_json(args.output_config, selected)
            write_json(args.report, report)
            result = {"selected_policy": selected["policy"], "selection_split": "val"}
        else:
            rows = read_manifest(args.manifest, args.data_root,
                                 check_files=not getattr(args, "schema_only", False))
            if args.command == "validate-manifest":
                result = {"rows": len(rows), "patients": len({r['patient_id'] for r in rows}),
                          "image_files_checked": not args.schema_only}
            elif args.command == "train":
                from .training import train_expert
                modalities = MODALITIES if args.modality == "all" else (args.modality,)
                result = {m: train_expert(config, rows, m, args.checkpoint_dir, args.device) for m in modalities}
            else:
                from .pipeline import evaluate
                result = evaluate(config, rows, args.split, args.checkpoint_dir, args.output, args.device)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
