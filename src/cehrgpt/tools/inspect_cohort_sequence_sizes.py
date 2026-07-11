"""
Inspect cohort sequence sizes to debug Arrow 2GB overflow during extract_cohort_sequences.

Run with the same YAML config you use for finetuning (must contain cohort_folder and
tokenized_full_dataset_path). Optionally override writer_batch_size to simulate
different values.

Example:
  python -m cehrgpt.tools.inspect_cohort_sequence_sizes /path/to/finetune_config.yaml
  python -m cehrgpt.tools.inspect_cohort_sequence_sizes /path/to/config.yaml --writer_batch_size 1000
"""
import argparse
import json
import sys

# Parse config the same way as the finetune runner so we get data_args and cehrgpt_args
from cehrgpt.runners.gpt_runner_util import parse_runner_args
from cehrgpt.runners.data_utils import inspect_cohort_sequence_sizes


def main():
    parser = argparse.ArgumentParser(description="Inspect cohort sequence sizes for map overflow debugging.")
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to YAML/JSON config (same as finetune). If omitted, CLI args are used.",
    )
    parser.add_argument(
        "--writer_batch_size",
        type=int,
        default=1000,
        help="Writer batch size to simulate (default 1000).",
    )
    parser.add_argument(
        "--iter_batch_size",
        type=int,
        default=500,
        help="Batch size when iterating the dataset for inspection (default 500).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write JSON result.",
    )
    args, remaining = parser.parse_known_args()

    if args.config:
        # So parse_runner_args() sees only the config file (yaml/json)
        sys.argv = [sys.argv[0], args.config]
    cehrgpt_args, data_args, _model_args, _training_args = parse_runner_args()

    result = inspect_cohort_sequence_sizes(
        data_args=data_args,
        cehrgpt_args=cehrgpt_args,
        writer_batch_size=args.writer_batch_size,
        iter_batch_size=args.iter_batch_size,
    )

    # Make result JSON-serializable (numpy ints -> int)
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    out = to_serializable(result)
    print(json.dumps(out, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote result to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
