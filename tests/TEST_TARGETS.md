# Unit Test Targets

## Utils (`src/utils.py`)
- `load_arithmetic_dataset` / `generate_question_answer_batches` arithmetic modes
- `load_gsm8k_dataset`, `load_mmlu_dataset` sampling guardrails (smoke-level)
- `calculate_threshold`, `Colors` logging helpers, and `load_arithmetic_dataset` uniqueness behavior
- `generate_question_answer_batches` parallel/debug datapoint paths

## Train (`src/train.py`)
- `exponential_weighted_average`
- `get_default_eval_batch_size`
- `initialize_model_and_optimizer` argument validation
- `run_periodic_evaluation` helper selection logic (stubbed models)

## Evaluation (`src/evaluation.py`)
- `load_cli_dataset` path selection + metadata map
- `get_answer_format_for_task`
- `compute_haiku_accuracy` guard when key missing
- `save_task_results` JSONL and optional GSM8K plot trigger

