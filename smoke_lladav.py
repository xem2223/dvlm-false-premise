"""
LLaDA-V smoke test: vanilla decoding (no Fast-dLLM / no dLLM-Cache).
Run from LLaDA-V/train (uses its test.jpg):
    cd ~/repos/LLaDA-V/train && CUDA_VISIBLE_DEVICES=0 python smoke_lladav.py
"""
import copy
import sys
import time
import warnings

import torch
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

warnings.filterwarnings("ignore")
torch.manual_seed(0)

PRETRAINED = "/users2/user2/models/LLaDA-V"
DEVICE = "cuda:0"
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
QUESTION = sys.argv[2] if len(sys.argv) > 2 else "What objects are in the image? Answer briefly."

t0 = time.time()
tokenizer, model, image_processor, _ = load_pretrained_model(
    PRETRAINED, None, "llava_llada", attn_implementation="sdpa", device_map=DEVICE)
model.eval()
print(f"[load] {time.time() - t0:.1f}s  mem={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")

image = Image.open(IMAGE_PATH).convert("RGB")
image_tensor = [t.to(dtype=torch.float16, device=DEVICE)
                for t in process_images([image], image_processor, model.config)]

conv = copy.deepcopy(conv_templates["llava_llada"])
conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + QUESTION)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                  return_tensors="pt").unsqueeze(0).to(DEVICE)

t0 = time.time()
out = model.generate(
    input_ids, images=image_tensor, image_sizes=[image.size],
    steps=64, gen_length=64, block_length=64, temperature=0.0,
    remasking="low_confidence", tokenizer=tokenizer,
    stopping_criteria=["<|eot_id|>"],
)
print(f"[gen] {time.time() - t0:.1f}s  peak mem={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")
print("[prompt tail]", repr(prompt[-120:]))
print("[output]", tokenizer.batch_decode(out, skip_special_tokens=True)[0])