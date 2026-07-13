import argparse
from dataclasses import asdict, field, make_dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Optional

from simple_parsing import ArgumentGenerationMode, parse

from configs import BaseConfig
from run import load_model
from eval.base import pick, all_metrics, all_configs


def _get_fname(tag: Optional[str] = None) -> str:
    kst = timezone(timedelta(hours=9))
    now = datetime.now(kst).strftime("%y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    return now + suffix


def _peek_metrics() -> list[str]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--metrics", nargs="+", default=all_metrics())
    known, _ = pre.parse_known_args()
    return known.metrics


def main():
    metrics = _peek_metrics()
    configs = all_configs()

    # 인자로 전달되는 metric 동적 추가
    EvalConfig = make_dataclass(
        "EvalConfig",
        [("metrics", list[str], field(default_factory=lambda: metrics))]
        + [(metric, configs[metric], field(default_factory=configs[metric])) for metric in metrics],
        bases=(BaseConfig,),
    )

    conf = parse(EvalConfig, argument_generation_mode=ArgumentGenerationMode.NESTED)
    print("load model...", end=" ", flush=True)
    model, tokenizer = load_model(conf.model)
    print("(done)")

    scores = {}
    for name in conf.metrics:
        print(f"{name} evaluating...")
        scores[name] = pick(name)[1](model, tokenizer, conf, getattr(conf, name))
        print(f"  (done) {scores[name]:.4f}")

    print("save metrics...", end=" ", flush=True)
    fname = _get_fname(conf.tag)
    dir = Path.cwd() / "runs"
    dir.mkdir(exist_ok=True)
    with open(dir / f"{fname}.json", "w") as f:
        json.dump({"config": asdict(conf), "metrics": scores}, f, indent=2, ensure_ascii=False)
    print("(done)")


if __name__ == "__main__":
    main()
