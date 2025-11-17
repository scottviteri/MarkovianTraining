import itertools

import pytest

from utils import generate_question_answer_batches, load_arithmetic_dataset


def test_load_arithmetic_dataset_test_split_has_expected_length():
    data = list(load_arithmetic_dataset(chunk_size=7, split="test"))
    assert len(data) == 7
    for question, answer in data:
        assert question.count("+") >= 1
        # Sums should be convertible to integers
        int(answer)


def test_generate_question_answer_batches_standard_mode(arithmetic_batch):
    assert len(arithmetic_batch) == 2
    unique_questions = {q for q, _ in arithmetic_batch}
    assert len(unique_questions) == 2
    for question, answer in arithmetic_batch:
        assert "+" in question
        int(answer)


def test_generate_question_answer_batches_parallel_mode():
    batch_iter = generate_question_answer_batches(
        num_batches=1,
        batch_size=3,
        task_type="arithmetic",
        tokenizer=None,
        hyperparameters={"parallel": True},
    )
    batch = next(batch_iter)
    assert len(batch) == 3
    first = batch[0]
    # Every entry should be identical when parallel mode repeats datapoints
    assert all(pair == first for pair in batch)


def test_debug_repeat_datapoint_mode():
    batch_iter = generate_question_answer_batches(
        num_batches=3,
        batch_size=2,
        task_type="arithmetic",
        tokenizer=None,
        hyperparameters={"debug_repeat_datapoint": True},
    )
    batches = list(itertools.islice(batch_iter, 3))
    assert len(batches) == 3
    # All batches should be identical when repeating the same datapoint
    assert batches[0] == batches[1] == batches[2]

