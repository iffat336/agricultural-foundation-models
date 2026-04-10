from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.ood_benchmark import (
    best_variant_by_split,
    load_experiment_log,
    render_markdown_report,
    summarize_by_split,
    variant_improvements,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize out-of-distribution benchmark runs for the agricultural foundation models project."
    )
    parser.add_argument(
        "--input",
        default="data/experiment_runs.csv",
        help="Path to the experiment log CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where benchmark summaries should be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input experiment log not found: {input_path}")

    runs = load_experiment_log(input_path)
    summary = summarize_by_split(runs)
    best = best_variant_by_split(summary)
    improvements = variant_improvements(summary)
    report = render_markdown_report(summary, best, improvements, input_path)
    output_files = write_outputs(summary, best, improvements, report, args.output_dir)

    print("OOD benchmark outputs written:")
    for output_file in output_files:
        print(f"- {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
