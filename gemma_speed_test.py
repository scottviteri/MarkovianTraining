#!/usr/bin/env python
"""
Comprehensive Gemma Speed Test
This script will test Gemma models once you have authentication set up.
"""

import time
import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any

def get_system_info():
    """Get system performance info"""
    return {
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "GPU_Memory_GB": torch.cuda.get_device_properties(0).total_memory // (1024**3) if torch.cuda.is_available() else 0,
        "CPU_Cores": psutil.cpu_count(),
        "RAM_GB": psutil.virtual_memory().total // (1024**3),
        "CUDA_Available": torch.cuda.is_available(),
        "PyTorch_Version": torch.__version__
    }

def test_model_speed(model_name: str, test_name: str) -> Dict[str, Any]:
    """Test loading and generation speed for a model"""
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Test loading speed
        print("Loading model...")
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        load_time = time.time() - start_time
        model_size = sum(p.numel() for p in model.parameters())
        
        print(f"✅ Load Time: {load_time:.2f} seconds")
        print(f"✅ Model Size: {model_size:,} parameters ({model_size/1e9:.1f}B)")
        
        # Test generation speed
        print("\nTesting generation speed...")
        test_prompts = [
            "The capital of France is",
            "To solve this math problem: 25 + 37 = ",
            "Write a short story about",
            "Explain quantum computing in simple terms:"
        ]
        
        total_tokens = 0
        total_time = 0
        
        for i, prompt in enumerate(test_prompts):
            print(f"  Prompt {i+1}/4...")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            gen_time = time.time() - start_time
            
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            tokens_generated = len(outputs[0]) - len(inputs["input_ids"][0])
            
            total_tokens += tokens_generated
            total_time += gen_time
            
            print(f"    Generated {tokens_generated} tokens in {gen_time:.2f}s ({tokens_generated/gen_time:.1f} tok/s)")
        
        avg_speed = total_tokens / total_time
        
        # Memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / (1024**3)
            memory_available = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_percent = (memory_used / memory_available) * 100
        else:
            memory_used = memory_available = memory_percent = 0
        
        results = {
            "model_name": model_name,
            "test_name": test_name,
            "load_time": load_time,
            "model_size": model_size,
            "avg_generation_speed": avg_speed,
            "memory_used_gb": memory_used,
            "memory_percent": memory_percent,
            "success": True
        }
        
        print(f"\n🎯 SUMMARY:")
        print(f"   Average Speed: {avg_speed:.1f} tokens/second")
        print(f"   Memory Used: {memory_used:.1f}GB ({memory_percent:.1f}%)")
        print(f"   Load Time: {load_time:.2f} seconds")
        
        # Clean up
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f"❌ Error testing {model_name}: {str(e)}")
        return {
            "model_name": model_name,
            "test_name": test_name,
            "error": str(e),
            "success": False
        }

def main():
    print("🚀 GEMMA SPEED TEST SUITE")
    print("="*60)
    
    # System info
    sys_info = get_system_info()
    print("SYSTEM INFO:")
    for key, value in sys_info.items():
        print(f"  {key}: {value}")
    
    # Models to test (in order of size)
    models_to_test = [
        ("google/gemma-3-1b-it", "Gemma 3 1B (Instruction)"),
        ("google/gemma-3-4b-it", "Gemma 3 4B (Instruction)"),  
        ("google/gemma-3-12b-it", "Gemma 3 12B (Instruction)")
    ]
    
    results = []
    
    for model_name, test_name in models_to_test:
        result = test_model_speed(model_name, test_name)
        results.append(result)
        
        # Break if we hit memory limits
        if not result["success"] and "memory" in str(result.get("error", "")).lower():
            print("⚠️  Stopping tests due to memory constraints")
            break
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SPEED COMPARISON")
    print(f"{'='*60}")
    
    successful_results = [r for r in results if r["success"]]
    
    for result in successful_results:
        print(f"{result['test_name']:25} | "
              f"{result['avg_generation_speed']:6.1f} tok/s | "
              f"{result['load_time']:6.1f}s load | "
              f"{result['memory_used_gb']:5.1f}GB RAM")
    
    if successful_results:
        fastest = max(successful_results, key=lambda x: x["avg_generation_speed"])
        print(f"\n🏆 Fastest Model: {fastest['test_name']} at {fastest['avg_generation_speed']:.1f} tok/s")
    
    print(f"\n✅ Test completed! Your H100 should handle all Gemma models excellently.")

if __name__ == "__main__":
    main()
