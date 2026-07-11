from dataclasses import field, make_dataclass

from simple_parsing import parse

from configs import BaseConfig
from run import load_model
from eval.base import pick, all_metrics, all_configs

EvalConfig = make_dataclass(
    "EvalConfig",
    [("metrics", list[str], field(default_factory=all_metrics))]
    + [(metric, conf, field(default_factory=conf)) for metric, conf in all_configs().items()],
    bases=(BaseConfig,),
)


def main():
    conf = parse(EvalConfig)
    model, tokenizer = load_model(conf.model)
    for name in conf.metrics:
        score = pick(name)[1](model, tokenizer, conf, getattr(conf, name))
        print(f"{name}={score:.4f}")


if __name__ == "__main__":
    main()
