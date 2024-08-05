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
        return self.linear(output).squeeze(-1)


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


def calculate_answer_log_probs(model, tokenizer, device, reasoning_tokens, answers):
    # Decode reasoning tokens to text
    reasoning_text = tokenizer.batch_decode(reasoning_tokens, skip_special_tokens=True)
    
    # Prepare the full prompts with instruction, reasoning, and answer
    full_prompts = [
        f"Use the following possibly mistaken reasoning to help predict the true answer, which will come immediately after the 'Answer:' tag. Try to spot flaws in the provided reasoning to guide your prediction.\nStepByStep: {r} \nAnswer: {a}"
        for r, a in zip(reasoning_text, answers)
    ]
    
    # Tokenize the full prompts
    tokenized_full_prompts = tokenizer(
        full_prompts,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Find the position of "Answer:" in the tokenized input
    # 28705 is space token, leading to the final space before the answer (which doesn't contain spaces)
    answer_start_positions = [
        (input_ids == 28705).nonzero(as_tuple=True)[0][-1].item() + 1
        for input_ids in tokenized_full_prompts.input_ids
    ]

    with torch.no_grad():
        outputs = model(
            input_ids=tokenized_full_prompts.input_ids,
            attention_mask=tokenized_full_prompts.attention_mask,
        )
        logits = outputs.logits

    # Calculate log probabilities for the answer tokens
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    answer_log_probs = [
        log_probs[i, start-1:-1].gather(1, tokenized_full_prompts.input_ids[i, start:].unsqueeze(-1)).squeeze(-1)
        for i, start in enumerate(answer_start_positions)
    ]

    # Calculate average log probability for each answer
    avg_log_probs = torch.stack([
        (probs * mask[start_idx:]).sum() / (mask[start_idx:].sum() + 1e-8)
        for (mask, probs, start_idx) in zip(tokenized_full_prompts.input_ids, answer_log_probs, answer_start_positions)
    ])
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

def calculate_advantages(model, frozen_model, tokenizer, device, tokenized_inputs, outputs, baseline_outputs, answers, activation_capturer):
    last_hidden_state = torch.cat(activation_capturer.activation, dim=1)
    assert last_hidden_state.shape[1] == 400, f"Expected last_hidden_state to have 400 tokens, but got {last_hidden_state.shape[1]}"
    
    value_prediction = model.value_head(last_hidden_state)
    assert value_prediction.shape == (outputs.shape[0], 400), f"Expected value_prediction shape ({outputs.shape[0]}, 400), but got {value_prediction.shape}"

    reasoning_tokens = outputs[:, tokenized_inputs.input_ids.shape[1]:]
    baseline_reasoning_tokens = baseline_outputs[:, tokenized_inputs.input_ids.shape[1]:]

    log_prob_ans_given_reasoning = calculate_answer_log_probs(
        frozen_model, tokenizer, device, reasoning_tokens, answers
    )
    assert log_prob_ans_given_reasoning.shape == (outputs.shape[0],), f"Expected log_prob_ans_given_reasoning shape ({outputs.shape[0]},), but got {log_prob_ans_given_reasoning.shape}"

    log_prob_ans_given_default_reasoning = calculate_answer_log_probs(
        frozen_model, tokenizer, device, baseline_reasoning_tokens, answers
    )
    assert log_prob_ans_given_default_reasoning.shape == (outputs.shape[0],), f"Expected log_prob_ans_given_default_reasoning shape ({outputs.shape[0]},), but got {log_prob_ans_given_default_reasoning.shape}"

    initial_advantage = log_prob_ans_given_reasoning - log_prob_ans_given_default_reasoning
    advantage = initial_advantage - value_prediction.mean(dim=1)
    assert advantage.shape == (outputs.shape[0],), f"Expected advantage shape ({outputs.shape[0]},), but got {advantage.shape}"

    return value_prediction, initial_advantage, advantage, reasoning_tokens, log_prob_ans_given_reasoning, log_prob_ans_given_default_reasoning

def calculate_losses(unfrozen_avg_log_probs_reasoning_given_question, frozen_avg_log_probs_reasoning_given_question, advantage, value_prediction, initial_advantage, use_ppo, ppo_epsilon):
    if use_ppo:
        ppo_ratio = torch.exp(
            unfrozen_avg_log_probs_reasoning_given_question - frozen_avg_log_probs_reasoning_given_question
        )
        clipped_ratio = torch.clamp(ppo_ratio, 1 - ppo_epsilon, 1 + ppo_epsilon)
        policy_loss = calculate_ppo_loss(
            unfrozen_avg_log_probs_reasoning_given_question,
            frozen_avg_log_probs_reasoning_given_question,
            advantage,
            epsilon=ppo_epsilon,
        )
    else:
        policy_loss = (unfrozen_avg_log_probs_reasoning_given_question * -advantage).mean()
        ppo_ratio = None
        clipped_ratio = None

    value_loss = torch.abs(value_prediction.mean(dim=1) - initial_advantage).mean()
    return policy_loss, value_loss, ppo_ratio, clipped_ratio

def train():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"src/AnalyzeResults/PolicyGradientDictionary_{timestamp}.log"

    with open(filename, "w") as log_file:
        pass  # Empty file created

    model, frozen_model, tokenizer, device, activation_capturer = load_mistral_model()
    value_head_learning_rate = 3e-4
    model_learning_rate = 1e-4
    value_head_optimizer = torch.optim.AdamW(model.value_head.parameters(), lr=value_head_learning_rate)
    model_optimizer = bitsandbytes.optim.AdamW8bit(
        [p for n, p in model.named_parameters() if not n.startswith('value_head')],
        lr=model_learning_rate
    )
    # Train the model to make q_cot_stub more likely
    batch_size = 3
    gradient_accumulation_steps = 2  # Only for the main model, not the value head
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

        prompts = [
            f"Work through the following question step by step, concisely decomposing problems into subproblems.\nQuestion: {q}\nStepByStep:"
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
            baseline_outputs = frozen_model.generate(
                tokenized_inputs.input_ids,
                attention_mask=tokenized_inputs.attention_mask,
                max_new_tokens=400,
                min_new_tokens=400,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        value_prediction, initial_advantage, advantage, reasoning_tokens, log_prob_ans_given_reasoning, log_prob_ans_given_default_reasoning = calculate_advantages(
            model, frozen_model, tokenizer, device, tokenized_inputs, outputs, baseline_outputs, answers, activation_capturer
        )

        with torch.no_grad():
            full_attention_mask = torch.cat([tokenized_inputs.attention_mask, torch.ones_like(reasoning_tokens)], dim=1)
            unfrozen_outputs = model(
                input_ids=outputs,
                attention_mask=full_attention_mask
            )
            frozen_outputs = frozen_model(
                input_ids=outputs,
                attention_mask=full_attention_mask
            )
            unfrozen_logits = unfrozen_outputs.logits[:,tokenized_inputs.input_ids.shape[1]-1:-1,:]
            frozen_logits = frozen_outputs.logits[:,tokenized_inputs.input_ids.shape[1]-1:-1,:]
        
        unfrozen_log_probs = torch.nn.functional.log_softmax(unfrozen_logits, dim=-1)
        frozen_log_probs = torch.nn.functional.log_softmax(frozen_logits, dim=-1)
        
        unfrozen_token_log_probs = unfrozen_log_probs.gather(2, reasoning_tokens.unsqueeze(-1)).squeeze(-1)
        frozen_token_log_probs = frozen_log_probs.gather(2, reasoning_tokens.unsqueeze(-1)).squeeze(-1)
    
        # Calculate average log probability for each generated sequence
        unfrozen_avg_log_probs_reasoning_given_question = unfrozen_token_log_probs.mean(dim=1)
        frozen_avg_log_probs_reasoning_given_question = frozen_token_log_probs.mean(dim=1)

        policy_loss, value_loss, ppo_ratio, clipped_ratio = calculate_losses(
            unfrozen_avg_log_probs_reasoning_given_question,
            frozen_avg_log_probs_reasoning_given_question,
            advantage,
            value_prediction,
            initial_advantage,
            use_ppo,
            ppo_epsilon
        )

        loss = (policy_loss + value_loss) / gradient_accumulation_steps
        loss.backward()
        
        # Debug prints
        print(f"Value prediction: {value_prediction.detach().to(torch.float32).cpu().numpy()}")
        print(f"Value head linear weight grad: {model.value_head.linear.weight.grad}")
        print(f"Value head linear bias grad: {model.value_head.linear.bias.grad}")

        if batch_index == 4-gradient_accumulation_steps:
            model_optimizer.zero_grad()
        # Only update weights after accumulating gradients
        if batch_index >= 4  and batch_index % gradient_accumulation_steps == 0:
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
        reasoning_text_first = tokenizer.decode(reasoning_tokens[0], skip_special_tokens=True)
        avg_log_prob = log_prob_ans_given_reasoning[0].item()
        baseline_avg_log_prob = log_prob_ans_given_default_reasoning[0].item()
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
            "Value Prediction": value_prediction.mean().item(),
            "Value Loss": value_loss.item(),
            "Policy Loss": policy_loss.item(),
        }

        if use_ppo and ppo_ratio is not None:
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