import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))

from utils import (
    load_gsm8k_dataset,
    load_mmlu_dataset,
    load_svamp_dataset,
    load_aqua_dataset,
    generate_question_answer_batches,
)


def take_first(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None
    except Exception as e:
        return {"error": str(e)}


def main():
    samples = []

    # Arithmetic
    try:
        batch = next(generate_question_answer_batches(
            num_batches=1,
            batch_size=1,
            task_type="arithmetic",
            tokenizer=None,
            hyperparameters={},
        ))
        q, a = batch[0]
        samples.append({"dataset": "arithmetic", "question": q, "answer": a})
    except Exception as e:
        samples.append({"dataset": "arithmetic", "error": str(e)})

    # GSM8K (test split)
    try:
        qa = take_first(load_gsm8k_dataset(split="test", chunk_size=1))
        if isinstance(qa, tuple):
            q, a = qa
            samples.append({"dataset": "gsm8k", "question": q, "answer": a})
        else:
            samples.append({"dataset": "gsm8k", "error": qa})
    except Exception as e:
        samples.append({"dataset": "gsm8k", "error": str(e)})

    # MMLU (validation split)
    try:
        qa = take_first(load_mmlu_dataset(split="validation", chunk_size=1))
        if isinstance(qa, tuple):
            q, a = qa
            samples.append({"dataset": "mmlu", "question": q, "answer": a})
        else:
            samples.append({"dataset": "mmlu", "error": qa})
    except Exception as e:
        samples.append({"dataset": "mmlu", "error": str(e)})

    # SVAMP
    try:
        qa = take_first(load_svamp_dataset(split="train", chunk_size=1))
        if isinstance(qa, tuple):
            q, a = qa
            samples.append({"dataset": "svamp", "question": q, "answer": a})
        else:
            samples.append({"dataset": "svamp", "error": qa})
    except Exception as e:
        samples.append({"dataset": "svamp", "error": str(e)})

    # AQuA
    try:
        qa = take_first(load_aqua_dataset(split="train", chunk_size=1))
        if isinstance(qa, tuple):
            q, a = qa
            samples.append({"dataset": "aqua", "question": q, "answer": a})
        else:
            samples.append({"dataset": "aqua", "error": qa})
    except Exception as e:
        samples.append({"dataset": "aqua", "error": str(e)})

    print(json.dumps(samples, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


