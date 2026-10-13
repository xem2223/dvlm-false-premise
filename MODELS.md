# Models used (downloaded 2026-10, public checkpoints, unmodified)

| local dir | HF repo | used by |
|---|---|---|
| models/LLaDA-V | GSAI-ML/LLaDA-V | run_pilot.py, decode_order.py, patch/inject scripts (--lladav_path) |
| models/llama3-llava-next-8b | lmms-lab/llama3-llava-next-8b | run_ar.py, ar_* scripts (AR baseline) |
| models/siglip-so400m-patch14-384 | google/siglip-so400m-patch14-384 | LaViDa vision tower (--vision_tower) |
| LaViDa/lavida-ckpts/lavida-llada-hd | jacklishufan/lavida-ckpts (subfolder lavida-llada-hd) - verify | lavida/*.py (--ckpt) |

Download on a new server:
    huggingface-cli download GSAI-ML/LLaDA-V --local-dir <models>/LLaDA-V
    huggingface-cli download lmms-lab/llama3-llava-next-8b --local-dir <models>/llama3-llava-next-8b
    huggingface-cli download google/siglip-so400m-patch14-384 --local-dir <models>/siglip-so400m-patch14-384
config.json sha256 of the copies used in the experiments: models_config_sha256.txt
