from typing import Protocol

from transformers import PreTrainedModel, PreTrainedTokenizer

from configs import BaseConfig


class Metric(Protocol):  # "eval/*"에 들어오는 평가 함수들에 대한 타입 명시
    def __call__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        conf: BaseConfig,
        metrics_conf,
    ) -> float:
        pass


_METRICS: dict[str, tuple[type, Metric]] = {}


def register(metric: str, config: type):
    def deco(fn: Metric) -> Metric:
        _METRICS[metric] = (config, fn)
        return fn

    return deco


def pick(metric: str) -> tuple[type, Metric]:
    if metric not in _METRICS:
        raise KeyError(metric)
    return _METRICS[metric]


def all_metrics() -> list[str]:
    return list(_METRICS)


def all_configs() -> dict[str, type]:
    return {metric: conf for metric, (conf, _) in _METRICS.items()}
