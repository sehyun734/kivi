import re
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset

from configs import BaseConfig
from eval.base import register
from run import generate


@dataclass
class GSM8KConfig:
    n_shots: int = 5
    max_tokens: int = 256


def _load_gsm8k(n_shots: int) -> tuple[str, list[tuple[str, str]]]:
    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    shots = ""
    for ex in train.select(range(n_shots)):
        shots += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    tests = [(ex["question"], ex["answer"]) for ex in test]
    return shots, tests


def _extract_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def _extract_pred(generated: str) -> Optional[str]:
    nums = re.findall(r"-?[\d,]*\.?\d+", generated)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


@register("gsm8k", GSM8KConfig)
@torch.no_grad()
def main(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    conf: BaseConfig,
    gsm8k_conf: GSM8KConfig,
) -> float:
    shots, tests = _load_gsm8k(gsm8k_conf.n_shots)
    n_corrects = 0
    for question, answer in tests:
        generated = generate(
            model,
            tokenizer,
            shots + f"Question: {question}\nAnswer:",
            conf,
            gsm8k_conf.max_tokens,
            stop="Question:",
        )
        pred = _extract_pred(generated)
        gold = _extract_gold(answer)
        n_corrects += pred == gold
    return n_corrects / len(tests)
