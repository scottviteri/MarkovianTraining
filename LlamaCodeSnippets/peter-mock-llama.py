from transformers import LlamaTokenizer, LlamaForCausalLM

tokenizer = LlamaTokenizer.from_pretrained("peterchatain/mock_llama")

model = LlamaForCausalLM.from_pretrained("peterchatain/mock_llama")
