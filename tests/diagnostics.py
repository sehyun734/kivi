from torch import Tensor


def mae(a: Tensor, b: Tensor) -> float:
    return (a - b).abs().mean().item()
