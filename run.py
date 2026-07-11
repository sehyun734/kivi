from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from simple_parsing import parse

from cache import Cache
from configs import BaseConfig
from models.base import pick
from patch import patch


def load_model(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    adapter = pick(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = patch(model, adapter)
    return model, tokenizer


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    conf: BaseConfig,
    max_tokens: int,
    stop: str | None = None,
) -> str:
    device = next(model.parameters()).device
    cache = Cache(conf.k_bits, conf.v_bits, conf.grp_size, conf.res_len)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # prefill 단계
    logits = model(input_ids=ids, past_key_values=cache, use_cache=True).logits
    next_id = logits[:, -1].argmax(-1, keepdim=True)
    out_ids = [next_id]

    # decode 단계
    for _ in range(max_tokens - 1):
        if next_id.item() == tokenizer.eos_token_id:
            break
        logits = model(input_ids=next_id, past_key_values=cache, use_cache=True).logits
        next_id = logits[:, -1].argmax(-1, keepdim=True)
        out_ids.append(next_id)
        out = tokenizer.decode(torch.cat(out_ids, dim=-1)[0], skip_special_tokens=True)
        if stop is not None and stop in out:
            break

    return out


@dataclass
class RunConfig(BaseConfig):
    prompt: str = "Hello"
    max_new_tokens: int = 128


def main():
    conf = parse(RunConfig)
    model, tokenizer = load_model(conf.model)
    print(generate(model, tokenizer, conf.prompt, conf, conf.max_new_tokens))


if __name__ == "__main__":
    main()
