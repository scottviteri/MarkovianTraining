import json
from pathlib import Path

import pytest

from perturbation_analysis import (
    build_dataset_perturbation_matrix,
    generate_markovian_comparison_report,
    summarize_markovian_comparison_results,
)


def _sample_results():
    return [
        {
            "Batch Index": 0,
            "metric_type": "accuracy",
            "Markovian Effects": {"Original": 0.8, "Delete20%": 0.1},
            "Non_Markovian Effects": {"Original": 0.6, "Delete20%": 0.05},
            "Effect Difference": {"Original": 0.2, "Delete20%": 0.05},
        },
        {
            "Batch Index": 1,
            "metric_type": "accuracy",
            "Markovian Effects": {"Original": 0.6, "Delete20%": 0.2},
            "Non_Markovian Effects": {"Original": 0.5, "Delete20%": 0.15},
            "Effect Difference": {"Original": 0.1, "Delete20%": 0.05},
        },
    ]


def test_summarize_markovian_comparison_results_returns_means():
    summary_rows = summarize_markovian_comparison_results(
        _sample_results(), perturb_type="delete"
    )
    assert len(summary_rows) == 1

    delete_row = next(row for row in summary_rows if row["degree"] == "Delete20%")
    assert pytest.approx(delete_row["markovian_mean"]) == 0.15
    assert pytest.approx(delete_row["non_markovian_mean"]) == 0.1
    assert pytest.approx(delete_row["mean_difference"]) == 0.05
    assert delete_row["is_baseline"] is False
    assert all(row["degree"] != "Original" for row in summary_rows)


def test_generate_markovian_comparison_report_reads_files(tmp_path: Path):
    base_dir = tmp_path / "results" / "gsm8k" / "run_markovian"
    comparison_dir = base_dir / "markovian_comparison_accuracy"
    comparison_dir.mkdir(parents=True)

    file_path = comparison_dir / "comparison_results_accuracy_delete.json"
    file_path.write_text(json.dumps(_sample_results()))

    report = generate_markovian_comparison_report(str(tmp_path / "results"))
    assert len(report) == 1

    first_row = report[0]
    assert first_row["task"] == "gsm8k"
    assert first_row["run"] == "run_markovian"
    assert first_row["perturbation"] == "delete"
    assert "markovian_mean" in first_row


def test_build_dataset_perturbation_matrix(tmp_path: Path):
    base_root = tmp_path / "results"
    gsm_dir = base_root / "gsm8k" / "run_markovian"
    arc_dir = base_root / "arc" / "run_markovian"
    for dir_path in (gsm_dir, arc_dir):
        (dir_path / "markovian_comparison_accuracy").mkdir(parents=True)

    (gsm_dir / "markovian_comparison_accuracy" / "comparison_results_accuracy_delete.json").write_text(
        json.dumps(_sample_results())
    )
    arc_samples = _sample_results()
    # tweak numbers so aggregation differs
    arc_samples[0]["Effect Difference"]["Delete20%"] = 0.2
    arc_samples[1]["Effect Difference"]["Delete20%"] = 0.0
    (arc_dir / "markovian_comparison_accuracy" / "comparison_results_accuracy_delete.json").write_text(
        json.dumps(arc_samples)
    )

    matrix = build_dataset_perturbation_matrix(str(base_root))
    assert matrix["datasets"] == ["arc", "gsm8k"]
    assert matrix["degrees"] == ["Delete20%"]

    gsm_cell = matrix["cells"]["gsm8k"]["Delete20%"]
    assert pytest.approx(gsm_cell["mean_difference"]) == 0.05

    arc_cell = matrix["cells"]["arc"]["Delete20%"]
    assert pytest.approx(arc_cell["mean_difference"]) == 0.1

    assert matrix["dataset_average"]["gsm8k"]["num_runs"] == 1
    assert matrix["overall_average"]["num_runs"] == 2


def test_build_dataset_matrix_with_aggregation(tmp_path: Path):
    base_root = tmp_path / "results"
    gsm_dir = base_root / "gsm8k" / "run_markovian"
    (gsm_dir / "markovian_comparison_accuracy").mkdir(parents=True)
    results = _sample_results()
    for entry in results:
        entry["Markovian Effects"]["Delete40%"] = 0.0
        entry["Non_Markovian Effects"]["Delete40%"] = 0.0
        entry["Effect Difference"]["Delete40%"] = 0.0

    # Adjust Delete20% values
    results[0]["Markovian Effects"]["Delete20%"] = 0.45
    results[0]["Non_Markovian Effects"]["Delete20%"] = 0.05
    results[0]["Effect Difference"]["Delete20%"] = 0.4

    results[1]["Markovian Effects"]["Delete20%"] = 0.35
    results[1]["Non_Markovian Effects"]["Delete20%"] = 0.15
    results[1]["Effect Difference"]["Delete20%"] = 0.2

    # Add Delete40% entries
    results[0]["Markovian Effects"]["Delete40%"] = 0.35
    results[0]["Non_Markovian Effects"]["Delete40%"] = 0.05
    results[0]["Effect Difference"]["Delete40%"] = 0.3

    results[1]["Markovian Effects"]["Delete40%"] = 0.15
    results[1]["Non_Markovian Effects"]["Delete40%"] = 0.05
    results[1]["Effect Difference"]["Delete40%"] = 0.1
    (gsm_dir / "markovian_comparison_accuracy" / "comparison_results_accuracy_delete.json").write_text(
        json.dumps(results)
    )

    matrix = build_dataset_perturbation_matrix(
        str(base_root), aggregate_perturbation_types=True
    )
    assert matrix["degrees"] == ["Delete"]
    delete_cell = matrix["cells"]["gsm8k"]["Delete"]
    # Weighted mean across four entries: diffs = [0.4,0.2,0.3,0.1] -> mean 0.25
    assert pytest.approx(delete_cell["mean_difference"]) == 0.25

