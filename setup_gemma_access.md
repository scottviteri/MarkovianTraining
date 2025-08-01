# GEMMA SPEED TEST SETUP GUIDE

## 🔐 Step 1: Authenticate with HuggingFace

### Option A: Web Authentication (Recommended)
1. Go to: https://huggingface.co/google/gemma-3-1b-it
2. Click "Log in or Sign Up to review the conditions and access this model content"
3. Accept Google's Gemma license terms
4. Access is granted **immediately**

### Option B: CLI Authentication
```bash
# Install/update huggingface hub
pip install -U huggingface_hub

# Login with your HF token
huggingface-cli login
# or the new command:
hf auth login
```

## 🚀 Step 2: Run Speed Tests

### Quick Test (Single Model)
```bash
python src/train.py --parallel --task_type arithmetic --num_batches 1 --batch_size 1 --model_type gemma-3-small
```

### Comprehensive Speed Test
```bash
python gemma_speed_test.py
```

## 📊 Expected Performance on Your H100

Based on your hardware specs:
- **GPU**: NVIDIA H100 PCIe (81GB VRAM)  
- **CUDA**: Version 12.9
- **Memory**: 81GB GPU RAM
- **Baseline**: 65 tokens/sec with GPT-2

### Estimated Gemma Speeds:

| Model | Size | Expected Speed | Memory Usage | Load Time |
|-------|------|---------------|--------------|-----------|
| Gemma 3 1B | 1B params | 150-200 tok/s | ~3GB | <5s |
| Gemma 3 4B | 4B params | 80-120 tok/s | ~10GB | 5-10s |
| Gemma 3 12B | 12B params | 40-60 tok/s | ~25GB | 10-15s |

### Why Your Hardware is Excellent for Gemma:

✅ **Memory**: 81GB easily handles all Gemma models with room for large batches
✅ **Compute**: H100 has ~2-3x better performance than A100 for inference  
✅ **Speed**: bfloat16 precision + device_map="auto" = optimal performance
✅ **Parallel Training**: Perfect for GRPO with multiple parallel samples

## 🎯 GRPO Speed Test with Gemma

Once authenticated, test GRPO performance:
```bash
# Test GRPO with parallel sampling
python src/train.py --parallel --model_type gemma-3-small --task_type arithmetic --num_batches 2 --batch_size 4

# Compare standard vs GRPO:
python src/train.py --model_type gemma-3-small --task_type arithmetic --num_batches 2 --batch_size 2  # Standard mode
python src/train.py --parallel --model_type gemma-3-small --task_type arithmetic --num_batches 2 --batch_size 4  # GRPO mode
```

## 🔧 Hardware Optimizations (Already Optimized!)

Your setup already includes:
- ✅ Flash Attention (rebuilt successfully)
- ✅ Mixed Precision (bfloat16)  
- ✅ Device Mapping (auto)
- ✅ CUDA 12.9 (latest)

No additional optimizations needed!

## 🐛 Troubleshooting

**If you get authentication errors:**
1. Make sure you're logged into HuggingFace
2. Accept the Gemma license at https://huggingface.co/google/gemma-3-1b-it
3. Try: `hf auth login` with your token

**If speeds seem slow:**
- Your H100 should easily exceed the estimates above
- Check `nvidia-smi` for GPU utilization
- Ensure no other processes are using GPU memory

