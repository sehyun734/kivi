from dataclasses import dataclass


@dataclass
class BaseConfig:
    model: str = "openai-community/gpt2"
    k_bits: int = 2
    v_bits: int = 2
    grp_size: int = 32
    res_len: int = 128
