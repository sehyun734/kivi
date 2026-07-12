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

colab new -s "$session" --gpu "$gpu"
trap 'rm -f "$tmp"; colab stop -s "$session"' EXIT # 성공/실패 관계없이 정리

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
    --exclude '.DS_Store' \
    .
colab upload -s "$session" "$tmp" "$remote.tgz"
echo "mkdir -p $remote && tar xzf $remote.tgz -C $remote && rm $remote.tgz" | colab console -s "$session"

colab exec -s "$session" <<EOF
import os, sys, runpy
os.chdir("$remote")
sys.path.insert(0, "$remote")
os.environ.update(dict(l.strip().split("=", 1) for l in open(".env") if "=" in l)) 
sys.argv = "eval $@".split()
_ = runpy.run_module("eval", run_name="__main__") # colab 자체 출력 기능 방지를 위한 변수 선언
EOF