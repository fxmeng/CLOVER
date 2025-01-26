import torch
from tqdm import tqdm
import tiktoken
from datasets import load_from_disk

def calculating_perplexity(model, data_path, stride = 512, device="cuda"):
    test = load_from_disk(data_path)['test']
    enc = tiktoken.get_encoding("gpt2")
    data_input_ids = torch.LongTensor(enc.encode("\n\n".join(test["text"]))).unsqueeze(0)
    seq_len = data_input_ids.size(1)-1
    max_length = model.config.block_size
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = data_input_ids[:, begin_loc:end_loc].to(device)
        target_ids = data_input_ids[:, begin_loc+1:end_loc+1].to(device)
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            _,loss,_,_ = model(input_ids, targets=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = loss

        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens

        n_tokens += num_loss_tokens
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)
    return avg_nll.item(), ppl.item()
    