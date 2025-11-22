import json
import os
from pathlib import Path

import pytest

from evaluation import (
    compute_haiku_accuracy,
    extract_answer,
    get_answer_format_for_task,
    save_task_results,
)
from train import exponential_weighted_average, get_default_eval_batch_size


def test_exponential_weighted_average_emphasizes_recent_values():
    values = [1.0, 2.0, 3.0]
    ema = exponential_weighted_average(values, 0.8)
    # EMA should be between mean and most recent value
    assert values[1] < ema < values[2]


@pytest.mark.parametrize(
    "train_batch,expected", [(1, 1), (2, 3), (8, 12)]
)
def test_get_default_eval_batch_size_scale(train_batch, expected):
    assert get_default_eval_batch_size(train_batch) == expected


def test_get_answer_format_for_task_mappings():
    assert get_answer_format_for_task("gsm8k") == "numeric"
    assert get_answer_format_for_task("mmlu") == "A-D"
    assert get_answer_format_for_task("mathqa") == "A-E"


def test_extract_answer_handles_multiple_formats():
    assert extract_answer("The final answer is 42.") == 42
    assert extract_answer("= -13") == -13
    assert extract_answer("No number here") == "[invalid]"


def test_compute_haiku_accuracy_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    entries = [{"generated_answer": "Answer: 4", "answer": "4"}]
    assert compute_haiku_accuracy(entries, "gsm8k", "numeric") is None


def test_save_task_results_creates_jsonl(tmp_path):
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    results = [{"question": "1 + 1", "answer": "2", "generated_answer": "Answer: 2"}]
    file_path = save_task_results(
        task_type="svamp",
        output_dir=str(out_dir),
        model_type="gpt2",
        accuracy=0.5,
        results=results,
        num_examples=1,
        extra_metrics={"custom": 1.0},
    )
    data_path = Path(file_path)
    assert data_path.exists()
    record = json.loads(data_path.read_text().splitlines()[-1])
    assert record["accuracy"] == 0.5
    assert record["custom"] == 1.0


def test_save_task_results_updates_gsm8k_plot(tmp_path):
    out_dir = tmp_path / "gsm8k"
    out_dir.mkdir()
    results = [{"question": "1 + 1", "answer": "2", "generated_answer": "Answer: 2"}]
    save_task_results(
        task_type="gsm8k",
        output_dir=str(out_dir),
        model_type="gpt2",
        accuracy=1.0,
        results=results,
        num_examples=1,
        batch_index=0,
    )
    plot_path = out_dir / "gsm8k_accuracy_over_batches.png"
    assert plot_path.exists()

