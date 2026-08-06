import torch
from cs336_basics.bpe import Tokenizer
from cs336_basics.transformer import softmax

@torch.inference_mode()
def decoder(
        model,
        tokenizer,
        prompt: str,
        max_tokens: int,
        seq_len: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
        device: torch.device = None,
) -> str:
    if prompt == "":
        raise ValueError("prompt 不能为空")
    if not 0 < top_p <= 1:
        raise ValueError("top_p 必须位于 (0, 1]")
    eos_id = tokenizer.reversed_vocab[b"<|endoftext|>"]
    model.eval()
    tokens_id = tokenizer.encode(prompt)
    x = torch.tensor(tokens_id, dtype = torch.long,device = device)
    len_new_tokens = 0
    while True:
        if len_new_tokens >= max_tokens or x[-1] == eos_id:
            break
        if len(x) > seq_len:
            context = x[-seq_len:]
        else:
            context = x
        logits = model(context)[-1,:]
        if temperature == 0:
            next_token_id = torch.argmax(logits).item()
        elif temperature < 0:
            raise ValueError('temperature 不能为负数')
        else:
            prob = softmax(logits / temperature, dim = -1)
            sorted_prob, sorted_id = torch.sort(prob, descending = True)
            cumulative_prob = torch.cumsum(sorted_prob, dim = -1)
            keep = (cumulative_prob - sorted_prob) < top_p  ##核心！！！
            final_prob = sorted_prob * keep
            final_prob /= final_prob.sum()
            index = torch.multinomial(final_prob,num_samples = 1)
            next_token_id = sorted_id[index].item()
        next_token = torch.tensor([next_token_id],dtype=x.dtype,device = x.device)
        x = torch.cat([x,next_token], dim = 0)
        len_new_tokens += 1
    return tokenizer.decode(x[len(tokens_id):].tolist())
        

    
