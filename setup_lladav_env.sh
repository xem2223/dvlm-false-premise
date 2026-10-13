#!/usr/bin/env bash
# LLaDA-V inference env for the scaffold-vs-grounding pilot (n01, RTX A6000 x4)
# Usage: bash setup_lladav_env.sh
#   ENV_NAME=lladav REPO_DIR=~/repos/LLaDA-V TORCH_CUDA=cu121 bash setup_lladav_env.sh
set -euo pipefail

ENV_NAME=${ENV_NAME:-lladav}
REPO_DIR=${REPO_DIR:-$HOME/repos/LLaDA-V}
TORCH_CUDA=${TORCH_CUDA:-cu121}   # driver < 530 이면 cu118
LOG_DIR=${LOG_DIR:-$(dirname "$REPO_DIR")}

eval "$(conda shell.bash hook)"

# 1) conda env (python 3.10, lavida env와 동일)
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[skip] conda env '$ENV_NAME' already exists"
else
    conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"
python -m pip install --upgrade pip

# 2) torch 먼저 (repo pyproject에서 torch 핀이 주석 처리돼 있음, deepspeed 빌드에도 필요)
pip install torch==2.1.2 torchvision==0.16.2 --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# 3) repo clone + commit 기록
mkdir -p "$LOG_DIR"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/ML-GSAI/LLaDA-V "$REPO_DIR"
fi
git -C "$REPO_DIR" rev-parse HEAD > "$LOG_DIR/lladav_commit.txt"

# 4) 공식 train/init_env.sh 와 동일 (llava 패키지 + [train] extras)
#    tokenizers~=0.15.2 핀 때문에 transformers는 4.39.x로 풀림
cd "$REPO_DIR/train"
DS_BUILD_OPS=0 pip install -e ".[train]"
pip install webdataset
pip install "numpy<2"

# 5) 버전 고정 기록
pip freeze > "$LOG_DIR/lladav_freeze.txt"

# 6) import / 버전 확인
python - << 'PY'
import torch, transformers, tokenizers, datasets, numpy, accelerate
print(f"torch        {torch.__version__}  cuda={torch.cuda.is_available()}  n_gpu={torch.cuda.device_count()}")
print(f"transformers {transformers.__version__}")
print(f"tokenizers   {tokenizers.__version__}")
print(f"accelerate   {accelerate.__version__}")
print(f"datasets     {datasets.__version__}")
print(f"numpy        {numpy.__version__}")
assert torch.cuda.is_available(), "CUDA not available"
assert transformers.__version__.startswith("4.39"), "unexpected transformers version"
assert int(numpy.__version__.split('.')[0]) < 2, "numpy>=2 installed"
from llava.model.builder import load_pretrained_model  # noqa: F401
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: F401
print("[ok] llava imports")
PY

echo "[done] env=$ENV_NAME repo=$REPO_DIR"
echo "[done] commit -> $LOG_DIR/lladav_commit.txt, freeze -> $LOG_DIR/lladav_freeze.txt"
echo "[next] cd $REPO_DIR/train && python /path/to/smoke_lladav.py"