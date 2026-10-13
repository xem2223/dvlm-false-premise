#!/usr/bin/env bash
# LLaVA-NeXT (AR control) env: clone of lladav, llava package swapped to LLaVA-NeXT.
# Both repos install a package named `llava`, so they cannot share one env.
set -euo pipefail

SRC_ENV=${SRC_ENV:-lladav}
ENV_NAME=${ENV_NAME:-llavanext}
REPO_DIR=${REPO_DIR:-$HOME/repos/LLaVA-NeXT}
LOG_DIR=${LOG_DIR:-$HOME/project3}

eval "$(conda shell.bash hook)"

# 1) clone env (torch 2.1.2+cu121, transformers 4.39.3, tokenizers 0.15.2 carried over)
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[skip] env '$ENV_NAME' exists"
else
    conda create -y -n "$ENV_NAME" --clone "$SRC_ENV"
fi
conda env config vars set -n "$ENV_NAME" LD_LIBRARY_PATH="" PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1
conda activate "$ENV_NAME"

# 2) swap llava package: remove LLaDA-V's editable install, add LLaVA-NeXT's
pip uninstall -y llava || true
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/LLaVA-VL/LLaVA-NeXT "$REPO_DIR"
fi
git -C "$REPO_DIR" rev-parse HEAD > "$LOG_DIR/llavanext_commit.txt"
pip install -e "$REPO_DIR" --no-deps
pip freeze > "$LOG_DIR/llavanext_freeze.txt"

# 3) checks
python - << 'PY'
import os, torch, transformers, tokenizers, llava
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} | transformers {transformers.__version__} | tokenizers {tokenizers.__version__}")
print("llava from:", os.path.dirname(llava.__file__))
assert "LLaVA-NeXT" in llava.__file__, "llava still points to another repo"
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM  # noqa: F401
from llava.mm_utils import process_images, tokenizer_image_token  # noqa: F401
print("[ok] LLaVA-NeXT imports")
PY
echo "[done] env=$ENV_NAME  (HF_HUB_OFFLINE=1 set: models load from local dir / cache only)"