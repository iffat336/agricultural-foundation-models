from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = [
    "run_name",
    "use_case",
    "split",
    "model_variant",
    "score",
    "f1",
    "ece",
    "notes",
]


def load_experiment_log(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    for numeric_column in ("score", "f1", "ece"):
        df[numeric_column] = pd.to_numeric(df[numeric_column], errors="coerce")

    return df[REQUIRED_COLUMNS].copy()


def summarize_by_split(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["use_case", "split", "model_variant"], as_index=False)
        .agg(
            score_mean=("score", "mean"),
            f1_mean=("f1", "mean"),
            ece_mean=("ece", "mean"),
            run_count=("run_name", "count"),
        )
        .sort_values(["use_case", "split", "score_mean"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def best_variant_by_split(summary_df: pd.DataFrame) -> pd.DataFrame:
    best_rows = (
        summary_df.sort_values(["use_case", "split", "score_mean"], ascending=[True, True, False])
        .groupby(["use_case", "split"], as_index=False)
        .first()
    )
    return best_rows.reset_index(drop=True)


def variant_improvements(summary_df: pd.DataFrame, baseline_name: str = "baseline") -> pd.DataFrame:
    baseline = (
        summary_df[summary_df["model_variant"] == baseline_name][["use_case", "split", "score_mean"]]
        .rename(columns={"score_mean": "baseline_score"})
    )
    joined = summary_df.merge(baseline, on=["use_case", "split"], how="left")
    joined["score_gain_vs_baseline"] = joined["score_mean"] - joined["baseline_score"]
    return joined


def render_markdown_report(
    summary_df: pd.DataFrame,
    best_df: pd.DataFrame,
    improvements_df: pd.DataFrame,
    source_path: str | Path,
) -> str:
    lines: list[str] = []
    lines.append("# OOD Benchmark Summary")
    lines.append("")
    lines.append(f"Source log: `{source_path}`")
    lines.append("")
    lines.append("## Best Variant Per Split")
    lines.append("")

    for _, row in best_df.iterrows():
        lines.append(
            f"- `{row['use_case']}` / `{row['split']}`: `{row['model_variant']}` "
            f"(score={row['score_mean']:.2f}, f1={row['f1_mean']:.2f}, ece={row['ece_mean']:.2f})"
        )

    lines.append("")
    lines.append("## Variant Gains Vs Baseline")
    lines.append("")

    for use_case, case_df in improvements_df.groupby("use_case"):
        lines.append(f"### {use_case}")
        for _, row in case_df.iterrows():
            gain = row["score_gain_vs_baseline"]
            gain_text = "n/a" if pd.isna(gain) else f"{gain:+.2f}"
            lines.append(
                f"- `{row['split']}` / `{row['model_variant']}`: "
                f"score={row['score_mean']:.2f}, gain_vs_baseline={gain_text}"
            )
        lines.append("")

    lines.append("## Table Snapshot")
    lines.append("")
    lines.append("| use_case | split | variant | score | f1 | ece | runs |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['use_case']} | {row['split']} | {row['model_variant']} | "
            f"{row['score_mean']:.2f} | {row['f1_mean']:.2f} | {row['ece_mean']:.2f} | {int(row['run_count'])} |"
        )

    return "\n".join(lines) + "\n"


def write_outputs(
    summary_df: pd.DataFrame,
    best_df: pd.DataFrame,
    improvements_df: pd.DataFrame,
    markdown_text: str,
    output_dir: str | Path,
) -> Iterable[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_csv = output_path / "ood_summary.csv"
    best_csv = output_path / "ood_best_variants.csv"
    improvements_csv = output_path / "ood_improvements.csv"
    report_md = output_path / "ood_report.md"

    summary_df.to_csv(summary_csv, index=False)
    best_df.to_csv(best_csv, index=False)
    improvements_df.to_csv(improvements_csv, index=False)
    report_md.write_text(markdown_text, encoding="utf-8")

    return [summary_csv, best_csv, improvements_csv, report_md]
