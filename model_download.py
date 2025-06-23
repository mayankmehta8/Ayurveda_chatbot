from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
save_path = "models/mistral-7b-instruct"

print("Downloading model...")
AutoTokenizer.from_pretrained(model_name).save_pretrained(save_path)
AutoModelForCausalLM.from_pretrained(model_name).save_pretrained(save_path)
print(f"Model saved to {save_path}")