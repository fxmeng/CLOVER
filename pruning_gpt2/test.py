from model import GPT, GPTConfig
import torch
from perplexity import calculating_perplexity

config = GPTConfig(
    block_size=1024,
    vocab_size=50257,
    n_layer=48,
    n_head=25,
    n_embd=1600,
    qk_head_embd=16,
    dropout=0,
    bias=True
)
model = GPT(config)
#model.pruning_and_load_state_dict("/data2/mengfanxu/CLOVer/output/state_dict/orthogonal/gpt2-xl.pt")
model = model.to("cuda")
print(model)
#print(calculating_perplexity(model, "/data2/mengfanxu/CLOVer/output/wikitext-2-raw-v1"))
#torch.save(model.state_dict(), "/data2/mengfanxu/CLOVer/output/state_dict/pruning/gpt2-xl-qk16.pt")

model.pruning_and_load_state_dict("/data2/mengfanxu/CLOVer/output/state_dict/pruning/gpt2-xl-qk16.pt")
print(calculating_perplexity(model, "/data2/mengfanxu/CLOVer/output/wikitext-2-raw-v1"))