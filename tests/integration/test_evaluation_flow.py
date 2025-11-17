import json
from pathlib import Path

import pytest

from evaluation import evaluate_model_on_numeric, save_task_results


@pytest.mark.slow
def test_numeric_evaluation_pipeline(gpt2_components, tiny_training_hparams, tmp_path):
    actor_model = gpt2_components["actor"]
    critic_model = gpt2_components["critic"]
    tokenizer = gpt2_components["tokenizer"]
    device = gpt2_components["device"]

    dataset = [
        ("Add 1 and 2", "3"),
        ("What is 5 plus 7?", "12"),
    ]

    accuracy, results, _ = evaluate_model_on_numeric(
        actor_model,
        critic_model,
        tokenizer,
        device,
        dataset,
        tiny_training_hparams,
        batch_size=1,
        num_samples=len(dataset),
        answer_extraction_method="simple",
    )

    assert len(results) == len(dataset)
    assert 0.0 <= accuracy <= 1.0

    out_dir = tmp_path / "eval"
    out_dir.mkdir()

    results_file = save_task_results(
        task_type="svamp",
        output_dir=str(out_dir),
        model_type="gpt2",
        accuracy=accuracy,
        results=results,
        num_examples=len(dataset),
        batch_index=0,
    )

    stored = json.loads(Path(results_file).read_text().splitlines()[-1])
    assert pytest.approx(stored["accuracy"], rel=1e-6) == accuracy

