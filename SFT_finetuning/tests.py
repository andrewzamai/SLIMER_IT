import torch
from transformers import AutoTokenizer

from src.SFT_finetuning.commons import initialization

from huggingface_hub import login

HF_ACCESS_TOKEN = initialization.get_HF_access_token('../../.env')
login(token=HF_ACCESS_TOKEN)

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
#tokenizer = tr.AutoTokenizer.from_pretrained("swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA")

MY_SYSTEM_PROMPT_SHORT = (
  "Tu sei Modello Italia, un modello di linguaggio naturale addestrato da iGenius."
)
prompt = "Ciao, chi sei?"
messages = [
  {"role": "user", "content": prompt},
  {"role": "assistant", "content": MY_SYSTEM_PROMPT_SHORT},
]
tokenized_chat = tokenizer.apply_chat_template(
  messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to(device)

print(tokenized_chat)

input_ids = tokenized_chat[0].tolist()  # assuming a batch size of 1

# Decode the token IDs back to text
decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
print("Decoded text:", decoded_text)
