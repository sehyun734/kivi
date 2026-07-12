from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm.auto import tqdm

from cache import Cache
from configs import BaseConfig
from eval.base import register
from datasets import load_dataset


@dataclass
class PPLConfig:
    seq_len: int = 1024
    limit: Optional[int] = 10


def _load_wikitext():
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(data["text"])
    return text


@register("ppl", PPLConfig)
@torch.no_grad()
def main(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    conf: BaseConfig,
    ppl_conf: PPLConfig,
) -> float:
    device = next(model.parameters()).device
    text = _load_wikitext()
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    seq_len = ppl_conf.seq_len
    n_chunks = ids.shape[1] // seq_len
    if ppl_conf.limit:
        n_chunks = min(n_chunks, ppl_conf.limit)
    nll = torch.zeros((), device=device)
    n_tokens = 0
    for i in tqdm(range(n_chunks), desc="ppl"):
        chunk = ids[:, i * seq_len : (i + 1) * seq_len]
        cache = Cache(conf.k_bits, conf.v_bits, conf.grp_size, conf.res_len)
        logits = model(input_ids=chunk, use_cache=True, past_key_values=cache).logits

        shift_logits = logits[:, :-1].reshape(-1, logits.shape[-1]).float()
        shift_labels = chunk[:, 1:].reshape(-1)
        nll += F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        n_tokens += shift_labels.numel()

    return torch.exp(nll / n_tokens).item()
