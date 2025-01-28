from model import GPT, GPTConfig
import torch
from perplexity import calculating_perplexity    
import argparse

parser = argparse.ArgumentParser(description="Separate the principal singular value and singular vectors from base model")
parser.add_argument("--model_path", type=str, required=True, help="The path of the base model.")
parser.add_argument("--qk_head_embd", type=int, default=64)
parser.add_argument("--dataset_path", type=str, default="/data2/mengfanxu/CLOVer/pruning_gpt2/dataset/wikitext-2-raw-v1")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()
print(args)

config = GPTConfig(
    block_size=1024,
    vocab_size=50257,
    n_layer=48,
    n_head=25,
    n_embd=1600,
    qk_head_embd=args.qk_head_embd,
    dropout=0,
    bias=True
)
model = GPT(config)
model = model.to(args.device)
print(model)
print(model.pruning_and_load_state_dict(args.model_path))
print(calculating_perplexity(model, args.dataset_path))