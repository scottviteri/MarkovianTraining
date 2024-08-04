import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2Config, GPT2Block
import bitsandbytes
import random
import numpy as np
import copy
import json


class ValueHead(torch.nn.Module):
    def __init__(self, last_layer, hidden_size):
        super().__init__()
        self.last_layer = copy.deepcopy(last_layer)
        self.linear = torch.nn.Linear(hidden_size, 1, dtype=torch.bfloat16)

        # Zero initialization for linear layer weights
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.constant_(self.linear.bias, 0)

        # Get the current model name from the last_layer
        if isinstance(last_layer, GPT2Block):
            self.model_name = "gpt2"
        elif "MistralDecoderLayer" in str(type(last_layer)):
            self.model_name = "mistral"
        else:
            self.model_name = "unknown"
    def forward(self, hidden_states):
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)

        if self.model_name=="gpt2":
            output = self.last_layer(hidden_states)[0]
        else:
            batch_size = hidden_states.shape[0]
            position_ids = torch.arange(hidden_states.shape[1]).unsqueeze(0).repeat(batch_size, 1).to(hidden_states.device)
            output = self.last_layer(hidden_states, position_ids=position_ids)[0]
        last_token_hidden = output[:, -1, :]
        return self.linear(last_token_hidden).squeeze(-1)


class ActivationCapturer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.activation = []

    def hook(self, module, input, output):
        self.activation.append(output[0][:,[-1],:].detach())
        if len(self.activation) > 400:
            self.activation.pop(0)

def load_mistral_model():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    #model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    frozen_model = copy.deepcopy(model)
    for param in frozen_model.parameters():
        param.requires_grad = False

    # Register the hook only for the unfrozen model
    activation_capturer = ActivationCapturer("Unfrozen Model")

    if "mistral" in model_name.lower():
        last_layer = model.model.layers[-1]
        model.model.layers[-2].register_forward_hook(activation_capturer.hook)
        hidden_size = model.config.hidden_size
    else:  # DistilGPT2
        last_layer = model.transformer.h[-1]
        model.transformer.h[-2].register_forward_hook(activation_capturer.hook)
        hidden_size = 768  # DistilGPT2 hidden size

    # Add value head with randomly initialized GPT2 layer
    model.value_head = ValueHead(last_layer, hidden_size)

    # Ensure value head parameters require gradients
    for param in model.value_head.parameters():
        param.requires_grad = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    frozen_model.to(device)

    return model, frozen_model, tokenizer, device, activation_capturer


def generate_question_answer_batch(batch_size: int):
    qa_batch = []
    for _ in range(batch_size):
        numbers = [random.randint(1, 99) for _ in range(15)]
        qa_batch.append((" + ".join(map(str, numbers)), str(sum(numbers))))
    return qa_batch


def generate_question_answer_batches(num_batches: int, batch_size: int):
    return [generate_question_answer_batch(batch_size) for _ in range(num_batches)]


def generate_reasoning(model, tokenizer, device, questions):
    prompts = [
        f"[INST] Work through the following question step by step, concisely decomposing problems into subproblems.[/INST]\nQuestion: {q}\nStepByStep:"
        for q in questions
    ]
    tokenized_inputs = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            tokenized_inputs.input_ids,
            attention_mask=tokenized_inputs.attention_mask,
            max_new_tokens=400,
            min_new_tokens=400,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    reasoning = outputs[:, tokenized_inputs.input_ids.shape[1] :]
    reasoning_text = tokenizer.batch_decode(reasoning, skip_special_tokens=True)
    
    return reasoning_text


def calculate_log_probs(model, tokenizer, device, reasoning_text, answers):
    tokenized_cot_ans = tokenizer(
        [
            f"[INST] Use the following possibly mistaken reasoning to help predict the true answer, which will come immediately after the 'Answer:' tag. Try to spot flaws in the provided reasoning to guide your prediction.[/INST]\nStepByStep:: {r} \nAnswer: {a}"
            for r, a in zip(reasoning_text, answers)
        ],
        padding=True,
        return_tensors="pt",
    ).to(device)

    tokenized_answers = tokenizer(
        answers, padding=True, return_tensors="pt", add_special_tokens=False
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=tokenized_cot_ans.input_ids,
            attention_mask=tokenized_cot_ans.attention_mask,
        )
        logits = outputs.logits

    answer_ids = tokenized_answers.input_ids
    log_probs = torch.nn.functional.log_softmax(
        logits[:, -answer_ids.shape[1] - 1 : -1, :], dim=-1
    )
    answer_log_probs = log_probs.gather(2, answer_ids.unsqueeze(-1)).squeeze(-1)
    avg_log_probs = (answer_log_probs * tokenized_answers.attention_mask).sum(dim=1) / (
        tokenized_answers.attention_mask.sum(dim=1) + 1e-8
    )

    return avg_log_probs


def calculate_baseline(previous_advantages, window_size=100):
    if len(previous_advantages) < window_size:
        return np.mean(previous_advantages) if previous_advantages else 0
    return np.mean(previous_advantages[-window_size:])


def calculate_ppo_loss(current_log_probs, old_log_probs, advantages, epsilon=0.2):
    ratio = torch.exp(current_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    return loss.mean()


def test_value_head_gradient_isolation():
    model, tokenizer, device, activation_capturer = load_mistral_model()
    
    # Prepare a sample input
    sample_text = "This is a test input."
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the activation from the last layer
    last_hidden_state = torch.cat(activation_capturer.activation, dim=1)[:, -1, :]
    
    # Compute value prediction
    value_prediction = model.value_head(last_hidden_state)
    
    # Compute loss and backward pass
    loss = value_prediction.mean()
    loss.backward()
    
    # Check gradients
    main_model_has_grad = False
    value_head_has_grad = False
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if 'value_head' in name:
                value_head_has_grad = True
            else:
                main_model_has_grad = True
                break  # We can stop as soon as we find a gradient in the main model
    
    print("Main model has gradients:", main_model_has_grad)
    print("Value head has gradients:", value_head_has_grad)
    
    assert not main_model_has_grad, "Main model should not have gradients"
    assert value_head_has_grad, "Value head should have gradients"
    
    print("Test passed: Value head gradients are isolated from the main model.")

def train():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"src/AnalyzeResults/PolicyGradientDictionary_{timestamp}.log"

    with open(filename, "w") as log_file:
        pass  # Empty file created

    model, frozen_model, tokenizer, device, activation_capturer = load_mistral_model()
    value_head_learning_rate = 1e-4
    model_learning_rate = 1e-4
    value_head_optimizer = torch.optim.AdamW(model.value_head.parameters(), lr=value_head_learning_rate)
    model_optimizer = bitsandbytes.optim.AdamW8bit(
        [p for n, p in model.named_parameters() if not n.startswith('value_head')],
        lr=model_learning_rate
    )
    # Train the model to make q_cot_stub more likely
    batch_size = 4
    gradient_accumulation_steps = 8  # Only for the main model, not the value head
    use_ppo = True
    ppo_epsilon = 0.2
    previous_advantages = []

    num_batches = 10000
    qa_batches = list(generate_question_answer_batches(num_batches=num_batches, batch_size=batch_size))

    # Log hyperparameters before starting the training loop
    with open(filename, "w") as log_file:
        hyperparameters = {
            "model_learning_rate": model_learning_rate,
            "value_head_learning_rate": value_head_learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "num_batches": num_batches,
            "use_ppo": use_ppo,
            "ppo_epsilon": ppo_epsilon,
        }
        json.dump(hyperparameters, log_file)
        log_file.write("\n")  # Add a newline after the hyperparameters

    model_optimizer.zero_grad()  # Move this outside the loop
    value_head_optimizer.zero_grad()

    for batch_index, qa_batch in enumerate(qa_batches):
        questions, answers = zip(*qa_batch)

        # Generate reasoning using the trained model
        reasoning_text = generate_reasoning(
            model, tokenizer, device, questions
        )
        last_hidden_state = torch.cat(activation_capturer.activation, dim=1)
        assert last_hidden_state.shape[1] == 400
        # Calculate value prediction
        value_prediction = model.value_head(last_hidden_state)
        # Calculate log probs using the frozen model
        avg_log_probs = calculate_log_probs(
            frozen_model, tokenizer, device, reasoning_text, answers
        )

        # Generate baseline reasoning using the frozen model
        baseline_reasoning_text = generate_reasoning(
            frozen_model, tokenizer, device, questions
        )

        # Calculate baseline log probs
        baseline_avg_log_probs = calculate_log_probs(
            frozen_model, tokenizer, device, baseline_reasoning_text, answers
        )

        # Calculate the initial advantage and final advantage
        initial_advantage = avg_log_probs - baseline_avg_log_probs
        advantage = initial_advantage - value_prediction.detach()

        # Tokenize questions and reasoning together
        tokenized_q_cot = tokenizer(
            [
                f"[INST] Work through the following question step by step, concisely decomposing problems into subproblems.[/INST]\nQuestion: {q}\nStepByStep: {r}"
                for q, r in zip(questions, reasoning_text)
            ],
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Calculate old log probabilities
        with torch.no_grad():
            old_q_cot_outputs = frozen_model(
                input_ids=tokenized_q_cot.input_ids, attention_mask=tokenized_q_cot.attention_mask
            )
            old_q_cot_log_probs = torch.nn.functional.log_softmax(
                old_q_cot_outputs.logits[:, -401:-1, :], dim=-1
            )
            old_cot_log_probs = old_q_cot_log_probs.gather(
                2, tokenized_q_cot.input_ids[:, -400:].unsqueeze(-1)
            ).squeeze(-1)

        # Calculate current log probabilities
        q_cot_outputs = model(input_ids=tokenized_q_cot.input_ids, attention_mask=tokenized_q_cot.attention_mask)
        q_cot_log_probs = torch.nn.functional.log_softmax(
            q_cot_outputs.logits[:, -401:-1, :], dim=-1
        )
        current_cot_log_probs = q_cot_log_probs.gather(
            2, tokenized_q_cot.input_ids[:, -400:].unsqueeze(-1)
        ).squeeze(-1)

        if use_ppo:
            # Use PPO loss
            ppo_ratio = torch.exp(
                current_cot_log_probs.mean(dim=1) - old_cot_log_probs.mean(dim=1)
            )
            clipped_ratio = torch.clamp(
                ppo_ratio, 1 - ppo_epsilon, 1 + ppo_epsilon
            )  # keep for logging
            policy_loss = (
                calculate_ppo_loss(
                    current_cot_log_probs.mean(dim=1),
                    old_cot_log_probs.mean(dim=1),
                    advantage,
                    epsilon=ppo_epsilon,
                )
                / gradient_accumulation_steps
            )
            value_loss = torch.abs(
                initial_advantage - value_prediction
            ).mean()  # or use torch.square()
            loss = policy_loss + value_loss
        else:
            # Use original policy gradient loss
            policy_loss = (current_cot_log_probs.mean(dim=1) * -advantage).mean()
            value_loss = torch.abs(
                initial_advantage - value_prediction
            ).mean()  # or use torch.square()
            loss = (policy_loss + value_loss) / gradient_accumulation_steps

        loss.backward()
        
        # Debug prints
        print(f"Value prediction: {value_prediction.detach().to(torch.float32).cpu().numpy()}")
        print(f"Value head linear weight grad: {model.value_head.linear.weight.grad}")
        print(f"Value head linear bias grad: {model.value_head.linear.bias.grad}")

        if batch_index == 128-gradient_accumulation_steps:
            model_optimizer.zero_grad()
        # Only update weights after accumulating gradients
        if batch_index >= 128  and batch_index % gradient_accumulation_steps == 0:
            model_optimizer.step()
            model_optimizer.zero_grad()
        value_head_optimizer.step()
        value_head_optimizer.zero_grad()

        # More debug prints after optimization step
        if batch_index % 10 == 0:
            print(f"Value head linear weight: {model.value_head.linear.weight.detach().to(torch.float32).cpu().numpy()}")
            print(f"Value head linear bias: {model.value_head.linear.bias.detach().cpu().to(torch.float32).numpy()}")

        # Update previous_advantages and log results
        previous_advantages.extend(initial_advantage.tolist())

        # Log only the first element of the batch
        q = questions[0]
        ans = answers[0]
        reasoning_text_first = reasoning_text[0]
        avg_log_prob = avg_log_probs[0].item()
        baseline_avg_log_prob = baseline_avg_log_probs[0].item()
        initial_advantage_value = initial_advantage[0].item()
        advantage_value = advantage[0].item()
        print(reasoning_text_first)
        print("Ans: ", ans, "Avg Log Prob: ", avg_log_prob)

        # Update log entry
        log_entry = {
            "Aggregate loss": loss.item() * gradient_accumulation_steps,
            "Batch Index": batch_index,
            "Prev Observation": f"Question: {q}",
            "Action": f"StepByStep: {reasoning_text_first}",
            "Observation": f"Answer: {ans}",
            "Reasoning Contains Answer": str(ans) in reasoning_text_first,
            "Avg Log Prob": avg_log_prob,
            "Baseline Avg Log Prob": baseline_avg_log_prob,
            "Initial Advantage": initial_advantage_value,
            "Advantage": advantage_value,
            "Value Prediction": value_prediction[0].item(),
            "Value Loss": value_loss.item(),
            "Policy Loss": policy_loss.item(),
        }

        if use_ppo:
            log_entry.update(
                {
                    "PPO Ratio": ppo_ratio[0].item(),
                    "PPO Clipped Ratio": clipped_ratio[0].item(),
                }
            )

        # Write progress to a file iteratively
        with open(filename, "a") as log_file:
            json.dump(log_entry, log_file)
            log_file.write("\n")  # Add a newline for each entry


if __name__ == "__main__":
    # Run the test
    #test_value_head_gradient_isolation()
    train()