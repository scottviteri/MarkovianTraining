from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer
from accelerate import Accelerator, DeepSpeedPlugin

# Define the path to the pretrained model and tokenizer
model_path = "/home/ubuntu/hf-llama-community/llama-7b"
tokenizer_path = "/home/ubuntu/hf-llama-community/tokenizer"

# Load the pretrained tokenizer
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

# Load the pretrained model
model = LlamaForCausalLM.from_pretrained(model_path)

# Initialize the DeepSpeed plugin
deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=2)

# Initialize the accelerator with the DeepSpeed plugin
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)

# Move the model and tokenizer to the accelerator
model, tokenizer = accelerator.prepare(model, tokenizer)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    deepspeed="ds_config.json",  # Add the path to the DeepSpeed config file
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
)

# Now you can use the model and tokenizer for your task
# For example, to generate text:

input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda')

# Generate text
output = trainer.model.generate(input_ids, max_length=100, temperature=0.7, do_sample=True)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)
