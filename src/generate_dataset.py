import random
from transformers import GPT2Tokenizer
import json
from typing import Iterator, Dict
from prepare_dataset import *
import json
from openai import OpenAI
from tqdm import tqdm


def jsonl_to_dict_iterator(filename: str) -> Iterator[Dict]:
    with open(filename, "r") as infile:
        for line in infile:
            yield json.loads(line)


def generate_arithmetic_question():
    num1 = random.randint(100, 999)
    num2 = random.randint(100, 999)
    operation = random.choice(["+"])
    question = f"{num1} {operation} {num2}"
    answer = str(eval(question))
    return {"Question": "Q: " + question, "Answer": "A: " + answer}

def generate_question_question():
    num1 = random.randint(100, 999)
    num2 = random.randint(100, 999)
    operation = random.choice(["+"])
    question = f"{num1} {operation} {num2}"
    answer = str(eval(question))
    return {"Question": "Q: " + question, "Answer": "A: " + question}


# arithmetic_questions = [generate_arithmetic_question() for _ in range(100000)]
# print(arithmetic_questions)


def save_to_jsonl(data, filename):
    with open(filename, "a") as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write("\n")


def read_from_jsonl(filename):
    data = []
    with open(filename, "r") as infile:
        for line in infile:
            data.append(json.loads(line))
    return data


def generate_pure_obs(batch_size, tokenizer):
    def mergeQA(d1, d2):
        t1 = tokenizer.encode(d1["Answer"] + "\n", return_tensors="pt")[0]
        t2 = tokenizer.encode(d2["Question"] + "\n", return_tensors="pt")[0]
        padding = torch.full(
            size=(15 - len(t1) - len(t2),), fill_value=tokenizer.eos_token_id
        )
        return torch.cat([t1, padding, t2])

    itr = jsonl_to_dict_iterator("arithmetic_questions.jsonl")
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    arithmetic_questions = read_from_jsonl("arithmetic_questions.jsonl")
    itr = batch(batch_size, itr)
    itr = group_pairs(itr)
    itr = map(lambda x: torch.stack(list(map(mergeQA, x[0], x[1]))), itr)
    return itr


client = OpenAI()


def generate_explanation(d):
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {
                "role": "system",
                "content": """
Decompose the addition into intermediate steps as concisely as possible. 
Only use numbers and the operations +, without using words.
Do not say the final answer.
For example: 
 Question: 446 + 372, Answer: 818,
 Explanation: 400 + 300 = 700\n46 + 70 = 116\n700 + 116 = 816\n816 + 2 
""",
            },
            {
                "role": "user",
                "content": f"""
    <Question> {d["Question"][3:]} </Question>
    <Answer> {d["Answer"][3:]} </Answer>
""",
            },
        ],
        max_tokens=40,
    )
    return response.choices[0].message.content


def add_explanations_to_file(n):
    # itr = jsonl_to_dict_iterator("arithmetic_explanations.jsonl")
    for _ in tqdm(range(n)):
        arith_question = generate_arithmetic_question()
        explanation = {
            **arith_question,
            "Explanation": generate_explanation(arith_question),
        }
        save_to_jsonl([explanation], "arithmetic_explanations.jsonl")


add_explanations_to_file(10000)
