# 📚 COMPLETE LIST OF SUPPORTED MODELS

## 🎯 Main Training Models (src/train.py)

| Model Type | HuggingFace Model | Parameters | Authentication Required | Notes |
|------------|-------------------|------------|------------------------|-------|
| `llama` | `meta-llama/Llama-3.1-8B-Instruct` | 8B | ✅ Yes (Gated) | Default model |
| `mistral` | `mistralai/Mistral-7B-Instruct-v0.2` | 7B | ✅ Yes (Gated) | Popular choice |
| `gpt2` | `openai-community/gpt2` | 124M | ❌ No | Good for testing |
| `tinystories` | `roneneldan/TinyStories` | 3.7M | ❌ No | Lightweight option |
| `phi` | `microsoft/Phi-3.5-mini-instruct` | 3.8B | ❌ No | Microsoft model (upgraded) |
| `phi-4` | `microsoft/phi-4` | ? | ⚠️ Possibly gated | Latest Phi model |
| `qwen3` | `Qwen/Qwen3-4B` | 4B | ❌ No | Newer Qwen version |
| `gemma-3` | `google/gemma-3-12b-it` | 12B | ✅ Yes (Gated) | Large Gemma |
| `gemma-3-small` | `google/gemma-3-1b-it` | 1B | ✅ Yes (Gated) | Small Gemma |

## 🔧 Special Features by Model

### Models with `trust_remote_code=True`
- `phi` (Phi-3.5-mini-instruct)
- `phi-4` (phi-4)  
- `gemma-3` (gemma-3-12b-it)
- `gemma-3-small` (gemma-3-1b-it)

### GRPO Compatible Models
All models support GRPO (Group-Relative Policy Optimization) with `--parallel_samples > 1`

### Performance Tier Recommendations

#### 🚀 High Performance (H100/A100)
- `llama` - 8B parameters, excellent for complex tasks
- `mistral` - 7B parameters, strong performance
- `gemma-3` - 12B parameters, latest Google model


#### ⚡ Medium Performance (RTX 4090/A40)
- `qwen3` - 4B parameters, good balance
- `phi` - 3.8B parameters, Microsoft optimized
- `gemma-3-small` - 1B parameters, efficient

#### 💡 Testing/Development (Any GPU)
- `gpt2` - 124M parameters, fast loading
- `tinystories` - 3.7M parameters, minimal resources

## 🔐 Authentication Requirements

### Gated Models (Require HF Login + License Acceptance)
1. **Llama 3.1**: Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. **Mistral**: Visit https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2  
3. **Gemma**: Visit https://huggingface.co/google/gemma-3-1b-it

### Free Models (No Authentication)
- GPT-2, TinyStories, Qwen3

## 🎯 Task Compatibility

All models support these tasks:
- `arithmetic` - Basic math problems
- `arithmetic_negative` - Math with negative numbers
- `gsm8k` - Grade school math dataset
- `mmlu` - Massive multitask language understanding
- `wiki_compression` - Wikipedia compression tasks
- `wiki_continuation` - Wikipedia continuation tasks

## 💡 Usage Examples

```bash
# Fast testing with no authentication
python src/train.py --model_type gpt2 --task_type arithmetic --num_batches 2

# High performance with authentication  
python src/train.py --model_type llama --task_type gsm8k --use_ppo

# GRPO training
python src/train.py --model_type gemma-3-small --parallel_samples 4 --task_type arithmetic

# Qwen3 training
python src/train.py --model_type qwen3 --task_type arithmetic

# Lightweight training
python src/train.py --model_type tinystories --task_type wiki_continuation
```

## 📝 Notes

- Default model is `llama` if no `--model_type` specified
- All models use LoRA for efficient fine-tuning
- All models support bfloat16 precision
- Device mapping is automatic (`device_map="auto"`)
- Tokenizer padding is set to "left" for all models
