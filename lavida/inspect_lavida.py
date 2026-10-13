"""
LaViDa structure check before porting the patching / injection scripts.

Run (env lavida, from ~/project3/LaViDa):
    CUDA_VISIBLE_DEVICES=0 python inspect_lavida.py \
        --image <one POPE image path> --question "What color is the dog in the image?"
Also useful:
    grep -rn "prefix_lm" llava/ | head -40

What it reports
  1. model class and the module tree (where the decoder layer list lives, number of layers)
  2. mask token id and the vocab size (vs LLaDA-V: 126336 / 126464)
  3. forward / generate signatures (which argument switches the prefix attention mask, cache)
  4. one generation with the official settings (prefix_lm=True) -> sanity check that the model works
  5. number of image tokens after prepare_inputs_labels_for_multimodal (official 980 per image)
"""
import argparse
import inspect

import torch


def section(t):
    print("\n" + "=" * 8 + " " + t + " " + "=" * 8, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/users2/user2/LaViDa/lavida-ckpts/lavida-llada-hd")
    ap.add_argument("--image", default=None)
    ap.add_argument("--question", default="What color is the dog in the image?")
    args = ap.parse_args()

    from llava.model.builder import load_pretrained_model

    vision_kwargs = dict(mm_vision_tower="/users2/user2/models/siglip-so400m-patch14-384",
                         mm_resampler_type=None, mm_projector_type="mlp2x_gelu",
                         mm_hidden_size=1152, use_mm_proj=True)
    tok, model, proc, _ = load_pretrained_model(args.ckpt, None, "llava_llada", device_map="cuda:0",
                                                vision_kwargs=vision_kwargs, torch_dtype="bfloat16")
    model.eval()

    section("1. classes and module tree")
    print("model:", type(model).__module__, type(model).__name__)
    for n, m in model.named_modules():
        if n and n.count(".") <= 2 and not n.split(".")[-1].isdigit():
            print(f"  {n:60s} {type(m).__name__}")
    cands = [(n, len(m)) for n, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and len(m) >= 16]
    print("ModuleList candidates for the decoder layers:", cands)
    for n, _ in cands:
        blk = dict(model.named_modules())[n][0]
        print(f"\n  first block of {n}: {type(blk).__name__}")
        for cn, cm in blk.named_children():
            print(f"    .{cn:30s} {type(cm).__name__}")

    section("2. tokens")
    for t in ["<|mdm_mask|>", "<|eot_id|>", "<image>"]:
        print(f"  {t!r:16s} -> {tok.convert_tokens_to_ids(t)}")
    print("  len(tokenizer) =", len(tok), "| embedding rows =", model.get_input_embeddings().weight.shape[0])
    for w in ["There", " There", "The", " The", "I", "No", "Sorry", "Unfortunately", "Upon"]:
        print(f"  {w!r:16s} -> {tok.encode(w, add_special_tokens=False)}")

    section("3. signatures")
    for name in ["forward", "generate", "prepare_inputs_labels_for_multimodal"]:
        f = getattr(model, name, None)
        if f is not None:
            print(f"  model.{name}{inspect.signature(f)}")
    for cand in ["model", "transformer", "base_model"]:
        sub = getattr(model, cand, None)
        if sub is not None and hasattr(sub, "forward"):
            print(f"  model.{cand}.forward{inspect.signature(sub.forward)}")

    if args.image is None:
        print("\n(no --image given: skipping generation and image-token count)")
        return

    section("4. one generation with official settings")
    from PIL import Image
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    import copy

    im = Image.open(args.image).convert("RGB")
    itensor = process_images([im], proc, model.config)
    itensor = [x.to(dtype=torch.bfloat16, device="cuda:0") for x in itensor] if isinstance(itensor, list) \
        else itensor.to(dtype=torch.bfloat16, device="cuda:0")
    conv = copy.deepcopy(conv_templates["llada"])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + args.question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print("  prompt:", repr(prompt[-200:]))
    ids = tokenizer_image_token(prompt, tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda:0")
    with torch.no_grad():
        out = model.generate(ids, images=itensor, image_sizes=[im.size], do_sample=False, temperature=0.1,
                             max_new_tokens=64, block_length=64, step_ratio=0.5, tokenizer=tok,
                             prefix_lm=True, verbose=False, schedule="shift")
    seq = out[0] if isinstance(out, (tuple, list)) else out
    print("  answer:", tok.batch_decode(seq, skip_special_tokens=True))

    section("5. image tokens")
    with torch.no_grad():
        res = model.prepare_inputs_labels_for_multimodal(ids, None, None, None, None, itensor, ["image"],
                                                         image_sizes=[im.size])
    emb = [r for r in res if isinstance(r, torch.Tensor) and r.dim() == 3]
    print("  returned tensors:", [tuple(r.shape) if isinstance(r, torch.Tensor) else type(r).__name__ for r in res])
    if emb:
        L = emb[0].shape[1]
        print(f"  prompt length after splicing = {L}, text tokens = {ids.shape[1] - 1}, "
              f"image tokens = {L - (ids.shape[1] - 1)}")


if __name__ == "__main__":
    main()