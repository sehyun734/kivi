from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from cache import Cache
from configs import BaseConfig
from eval.base import register


@dataclass
class PeakMemConfig:
    batch_sizes: list[int] = field(default_factory=lambda: [1, 4, 16, 32, 64])
    prompt_len: int = 512
    gen_len: int = 128


@register("peak_mem", PeakMemConfig)
@torch.no_grad()
def main(
    model: PreTrainedModel,
    _: PreTrainedTokenizer,
    conf: BaseConfig,
    peak_mem_conf: PeakMemConfig,
) -> float:
    for batch_size in peak_mem_conf.batch_sizes:
        try:
            torch.cuda.empty_cache()
            device = next(model.parameters()).device
            torch.cuda.reset_peak_memory_stats(device)  # 이전 peak 메모리 초기화

            vocab_size = model.config.vocab_size
            ids = torch.randint(vocab_size, (batch_size, peak_mem_conf.prompt_len), device=device)
            cache = Cache(conf.k_bits, conf.v_bits, conf.grp_size, conf.res_len)
            logits = model(input_ids=ids, use_cache=True, past_key_values=cache).logits
            next_id = logits[:, -1].argmax(-1, keepdim=True)
            for _ in range(peak_mem_conf.gen_len):
                logits = model(input_ids=next_id, use_cache=True, past_key_values=cache).logits
                next_id = logits[:, -1].argmax(-1, keepdim=True)

            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"  batch_size={batch_size} peak_mem={peak:.2f}gb", flush=True)

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  batch_size={batch_size} OOM", flush=True)
            break
    return peak
