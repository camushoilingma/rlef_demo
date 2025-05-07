import os
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
from tqdm import tqdm

# Prevent CUDA fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "meta-llama/CodeLlama-7b-Instruct-hf"
CACHE_DIR = "./hf_cache"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model and ref model
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    quantization_config=bnb_config,
    device_map="auto"
)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    quantization_config=bnb_config,
    device_map="auto"
)

device = next(model.parameters()).device

# Prompt dataset
prompts = [
    "Write a Python function that returns the square of a number.",
    "Write a function to check if a string is a palindrome."
]
dataset = Dataset.from_dict({"prompt": prompts})

# PPO config
ppo_config = PPOConfig(
    model_name=MODEL_NAME,
    learning_rate=1e-6,
    batch_size=1,
    mini_batch_size=1,
    gradient_accumulation_steps=1
)

# PPO Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=None,
    data_collator=lambda x: {
        "input_ids": tokenizer(x["prompt"], return_tensors="pt", padding=True).input_ids.squeeze(1)
    }
)

# Safe generation settings
generation_kwargs = {
    "do_sample": True,
    "max_new_tokens": 64,
    "top_k": 20,
    "top_p": 0.9,
    "temperature": 0.7,
    "pad_token_id": tokenizer.eos_token_id
}

# Training loop
for sample in tqdm(dataset, desc="Training"):
    prompt = sample["prompt"]
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Check logits for NaNs/Infs
    with torch.no_grad():
        output = model(input_ids=input_ids)
        logits = output[0]
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("🚨 Skipping generation due to bad logits.")
            continue

    # Generate response
    with torch.no_grad():
        try:
            response_tensor = model.generate(input_ids=input_ids, **generation_kwargs)[0]
        except RuntimeError as e:
            print(f"💥 Generation error: {e}")
            continue

    response = tokenizer.decode(response_tensor, skip_special_tokens=True)

    # Reward from code execution
    code = response.split(prompt)[-1].strip()
    try:
        exec_globals = {}
        exec(code, exec_globals)
        if "square" in exec_globals:
            passed = exec_globals == 9 and exec_globals["square"](-2) == 4
        elif "is_palindrome" in exec_globals:
            passed = (
                exec_globals["is_palindrome"]("radar") is True and
                exec_globals["is_palindrome"]("car") is False
            )
        else:
            passed = False
        reward = 1.0 if passed else -1.0
    except Exception:
        reward = -1.0

    print(f"\n🧠 Prompt: {prompt}")
    print(f"📤 Response:\n{response}")
    print(f"🏅 Reward: {reward}")

    # PPO step (positional args only)
    ppo_trainer.step(
        [input_ids.squeeze(0)],
        [response_tensor],
        [torch.tensor([reward]).to(device)]
    )

print("✅ RLEF PPO training complete.")

