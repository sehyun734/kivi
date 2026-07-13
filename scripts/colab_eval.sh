#!/usr/bin/env bash

set -euo pipefail
# e: 명령 하나라도 실패 시 즉시 중단
# u: 미정의 변수 참조 시 에러
# o pipefail: 파이프 중간 명령 실패도 실패로 처리

session="kivi"
gpu="A100"
root="$(cd "$(dirname "$0")/.." && pwd)"
remote="/content/kivi"
tmp="$(mktemp)"

export REQUEST_TIMEOUT=60

colab new -s "$session" --gpu "$gpu"
trap 'rm -f "$tmp"; colab stop -s "$session" >/dev/null 2>&1' EXIT # 성공/실패 관계없이 정리

colab install -s "$session" \
    accelerate \
    datasets \
    transformers \
    huggingface_hub \
    jaxtyping \
    simple-parsing

# 프로젝트 압축 및 업로드 후 원격에서 해제
tar czf "$tmp" -C "$root" \
    --no-xattrs --no-mac-metadata \
    --exclude '__pycache__' \
    --exclude '.venv' \
    --exclude '.git' \
    --exclude 'runs' \
    --exclude '.DS_Store' \
    .

colab upload -s "$session" "$tmp" "$remote.tgz"

# colab console 사용 시 세션 종료 후 터미널에 응답이 새어나오는 사소한 버그? 발생
colab exec -s "$session" <<EOF
import subprocess
_ = subprocess.run(
    "mkdir -p $remote && tar xzf $remote.tgz -C $remote && rm $remote.tgz",
    shell=True, check=True,
)
EOF

colab exec -s "$session" <<EOF
import os, sys, runpy

os.chdir("$remote")
sys.path.insert(0, "$remote")

# env 파싱
os.environ.update(dict(l.strip().split("=", 1) for l in open(".env") if "=" in l)) 

# HF 모델 다운로드 진행바 비활성화 (colab exec은 tty를 지원하지 않아 제외)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

# 명령어 인자 파싱
sys.argv = "eval $@".split()
_ = runpy.run_module("eval", run_name="__main__") # colab 자체 출력 기능 방지를 위한 변수 선언
EOF

# "runs/*"에는 새로운 세션이라 현재 결과 파일 하나밖에 없음
name="$(colab exec -s "$session" <<EOF | tail -n 1
from pathlib import Path
print(next(Path("$remote/runs").glob("*.json")).name)
EOF
)"

mkdir -p "$root/runs"
colab download -s "$session" "$remote/runs/$name" "$root/runs/$name"