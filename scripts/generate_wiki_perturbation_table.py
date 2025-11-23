#!/usr/bin/env python3
"""
Generate a wiki_continuation perturbation fragility table directly from the
Markovian comparison JSON artifacts.

The script reads the per-perturbation files inside a
`markovian_comparison_accuracy/` directory, aggregates the Effect Difference
values for every severity, and prints a severity (rows) × perturbation-type
(columns) table that also includes row/column means.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


COMPARISON_PREFIX = "comparison_results_accuracy_"
PERCENT_RE = re.compile(r"(\d+)(%?)")


def camel_case(token: str) -> str:
    return "".join(part.capitalize() for part in token.split("_"))


def normalize_severity(raw_label: str, pert_type: str) -> Tuple[str, int]:
    """Strip the perturbation-type prefix and return (label, numeric_severity)."""
    prefix = camel_case(pert_type)
    label = raw_label
    if raw_label.startswith(prefix):
        label = raw_label[len(prefix) :]

    match = PERCENT_RE.search(label)
    if match:
        value = int(match.group(1))
        suffix = match.group(2) or "%"
        pretty = f"{value}{suffix}"
        return pretty, value

    # Fallback if the pattern is unexpected
    return label, 0


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return math.nan
    return float(sum(vals) / len(vals))


def load_effect_means(path: Path) -> Dict[str, float]:
    with path.open() as f:
        entries = json.load(f)

    aggregates: Dict[str, List[float]] = defaultdict(list)
    for entry in entries:
        for degree, value in entry["Effect Difference"].items():
            aggregates[degree].append(float(value))

    return {degree: mean(vals) for degree, vals in aggregates.items()}


def build_table(comparison_dir: Path, include_baseline: bool = False):
    per_type: Dict[str, Dict[str, float]] = {}

    for file_path in sorted(comparison_dir.glob(f"{COMPARISON_PREFIX}*.json")):
        pert_type = file_path.stem[len(COMPARISON_PREFIX) :]
        per_type[pert_type] = load_effect_means(file_path)

    if not per_type:
        raise FileNotFoundError(
            f"No files matching '{COMPARISON_PREFIX}*.json' found in {comparison_dir}"
        )

    severities: Dict[int, Dict[str, float]] = defaultdict(dict)
    severity_labels: Dict[int, str] = {}

    for pert_type, effects in per_type.items():
        for raw_label, value in effects.items():
            label, numeric = normalize_severity(raw_label, pert_type)
            if not include_baseline and numeric == 0:
                continue
            severity_labels[numeric] = label
            severities[numeric][pert_type] = value

    if not severities:
        raise ValueError("No severities remained after filtering; check input data.")

    ordered_severities = [n for n in sorted(severities.keys())]
    ordered_types = list(per_type.keys())

    rows = OrderedDict()
    for severity in ordered_severities:
        label = severity_labels.get(severity, str(severity))
        entries = []
        for pert_type in ordered_types:
            entries.append(severities[severity].get(pert_type))
        rows[label] = entries

    return ordered_types, rows


def format_table(
    ordered_types: List[str],
    rows: Dict[str, List[float]],
    precision: int = 3,
) -> str:
    def fmt(value: float | None) -> str:
        if value is None or math.isnan(value):
            return "-"
        return f"{value:+0.{precision}f}"

    headers = ["Severity"] + ordered_types + ["Row Mean"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    column_sums = [0.0 for _ in ordered_types]
    column_counts = [0 for _ in ordered_types]

    for label, values in rows.items():
        row_vals = []
        row_sum = 0.0
        row_count = 0
        for idx, val in enumerate(values):
            if val is not None:
                row_sum += val
                row_count += 1
                column_sums[idx] += val
                column_counts[idx] += 1
            row_vals.append(fmt(val))
        row_mean = fmt(row_sum / row_count) if row_count else "-"
        lines.append("| " + " | ".join([label, *row_vals, row_mean]) + " |")

    column_means = [
        col_sum / count if count else math.nan
        for col_sum, count in zip(column_sums, column_counts)
    ]
    overall_sum = sum(val for val in column_sums if not math.isnan(val))
    overall_count = sum(column_counts)
    overall_mean = overall_sum / overall_count if overall_count else math.nan

    footer = ["Column Mean"] + [fmt(val) for val in column_means] + [fmt(overall_mean)]
    lines.append("| " + " | ".join(footer) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a wiki_continuation perturbation fragility table "
            "directly from markovian_comparison_accuracy JSON files."
        )
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        required=True,
        help="Path to the markovian_comparison_accuracy directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the markdown table (stdout otherwise).",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Keep the 0%% severity rows (default: drop baseline rows).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Decimal places for the table values (default: 3).",
    )
    args = parser.parse_args()

    ordered_types, rows = build_table(
        args.comparison_dir,
        include_baseline=args.include_baseline,
    )
    table = format_table(ordered_types, rows, precision=args.precision)

    prologue = (
        "> Units: Δlog P (Markovian drop − Non-Markovian drop, nats)\n\n" + table
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(prologue + "\n", encoding="utf-8")
    else:
        print(prologue)


if __name__ == "__main__":
    main()

