import argparse
import torch
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='''Llama merger parser''')
    parser.add_argument('base_model', type=str, help='base_model')
    parser.add_argument('path_to_LORA', type=str, help='path_to_LORA')
    parser.add_argument('path_to_save_to', type=str, help='path_to_save_to')
    args = parser.parse_args()

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map={"": "cpu"}, torch_dtype=torch.bfloat16)

    model = PeftModel.from_pretrained(base_model, args.path_to_LORA, torch_dtype=torch.bfloat16, device_map={"": "cpu"})

    print("merge and unload...")
    model = model.merge_and_unload()

    print("saving...")
    model.save_pretrained(args.path_to_save_to)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.path_to_LORA)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(args.path_to_save_to)
