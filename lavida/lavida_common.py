"""
LaViDa (LLaDA backbone) adapter for the false-premise analyses.

Run from ~/project3/LaViDa in env `lavida` (LaViDa's own `llava` package; NOT LLaDA-V's).

Structure (checked with inspect_lavida.py)
  decoder blocks : model.model.transformer.blocks   (32 x LLaDALlamaBlock, output = (x, cache))
  attention out  : block.attn_out  (Linear)  -> attention contribution to the residual
  MLP out        : block.ff_out    (Linear)  -> MLP contribution to the residual
  embedding      : model.model.transformer.wte;   logits computed inside model.model(...)
  mask token     : 126336 (same as LLaDA-V);  openers / "The" are single tokens

Attention masks (both supported, chosen per forward)
  full   : fully bidirectional (prefix_length=None, no bias)
  prefix : Prefix-DLM — prompt positions (image + question) cannot see response positions;
           response positions see everything. This is what official inference does (via a prompt
           KV cache). Implemented here as a boolean attention_bias so that no flex-attention
           recompilation happens for every new sequence length. `selfcheck` verifies it against
           the official cache path.

Score at the first response position (same as the LLaDA-V analyses)
  s = log(opener mass) - log("The" mass). Softmax normalisation cancels in the ratio, so no
  token blocking is needed.
"""
import copy
import math
import os

import torch
import torch.nn.functional as F
from PIL import Image

OPENERS = ["There", "I", "Unfortunately", "Sorry", "No", "Upon"]
EPS = 1e-6
MASK_ID = 126336
DEFAULT_CKPT = "/users2/user2/LaViDa/lavida-ckpts/lavida-llada-hd"
DEFAULT_VT = "/users2/user2/models/siglip-so400m-patch14-384"


class LaViDa:
    def __init__(self, ckpt=DEFAULT_CKPT, vision_tower=DEFAULT_VT, device="cuda:0"):
        import warnings
        warnings.filterwarnings("ignore")
        from llava.model.builder import load_pretrained_model
        vision_kwargs = dict(mm_vision_tower=vision_tower, mm_resampler_type=None,
                             mm_projector_type="mlp2x_gelu", mm_hidden_size=1152, use_mm_proj=True)
        self.tok, self.model, self.proc, _ = load_pretrained_model(
            ckpt, None, "llava_llada", device_map=device, vision_kwargs=vision_kwargs,
            torch_dtype="bfloat16")
        self.model.eval()
        self.device = device
        self.inner = self.model.model                      # LlavaLladaModel (LLaDA core + vision)
        self.blocks = self.inner.transformer.blocks
        self.wte = self.inner.transformer.wte
        self.n_layers = len(self.blocks)
        self.open_ids = self._single_ids(OPENERS)
        self.the_ids = self._single_ids(["The"])

    # ------------------------------------------------------------------ tokens
    def _single_ids(self, words):
        s = set()
        for w in words:
            for v in (w, " " + w):
                t = self.tok.encode(v, add_special_tokens=False)
                if len(t) == 1:
                    s.add(t[0])
        return torch.tensor(sorted(s), device=self.device)

    def s_of(self, logits):
        p = F.softmax(logits.float(), dim=-1)
        return math.log(float(p[self.open_ids].sum()) + EPS) - math.log(float(p[self.the_ids].sum()) + EPS)

    def probs(self, logits):
        """model-agnostic decision summary at one position.
        lo      = log p(openers) - log(1 - p(openers))   (refusal log-odds vs EVERYTHING else)
        margin  = log p(openers) - log max_{t not opener} p(t)   (> 0: opener mass beats every single
                  non-opener token, i.e. greedy-ish first word would be a refusal opener)"""
        p = F.softmax(logits.float(), dim=-1)
        po = float(p[self.open_ids].sum())
        pt = float(p[self.the_ids].sum())
        q = p.clone()
        q[self.open_ids] = 0
        pmax_other, imax = q.max(0)
        return {"p_open": po, "p_the": pt,
                "lo": math.log(po + EPS) - math.log(1 - po + EPS),
                "margin": math.log(po + EPS) - math.log(float(pmax_other) + EPS),
                "top_other": self.tok.decode([int(imax)])}

    def top(self, logits, k=5):
        p = F.softmax(logits.float(), dim=-1)
        v, i = p.topk(k)
        return [(self.tok.decode([int(t)]), round(float(x), 4)) for x, t in zip(v, i)]

    # ------------------------------------------------------------------ inputs
    def prompt_ids(self, question):
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        conv = copy.deepcopy(conv_templates["llada"])
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        ids = tokenizer_image_token(prompt, self.tok, IMAGE_TOKEN_INDEX, return_tensors="pt")
        return ids.unsqueeze(0).to(self.device)

    def image(self, path, target=None):
        from llava.mm_utils import process_images
        im = Image.open(path).convert("RGB")
        orig = im.size
        if target is not None and tuple(target) != tuple(im.size):
            im = im.resize(tuple(target), Image.BICUBIC)
        t = process_images([im], self.proc, self.model.config)
        t = [x.to(dtype=torch.bfloat16, device=self.device) for x in t] if isinstance(t, list) \
            else t.to(dtype=torch.bfloat16, device=self.device)
        return im.size, orig, t

    @torch.no_grad()
    def prepare(self, image_path, question, gen_len=32, target=None):
        """step-0 input: prompt embeddings + gen_len mask embeddings.
        returns dict(x, L, n_pre, n_img, pos, size, orig, ids, itensor)"""
        from llava.constants import IMAGE_TOKEN_INDEX
        size, orig, itensor = self.image(image_path, target)
        ids = self.prompt_ids(question)
        n_pre = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0])
        res = self.model.prepare_inputs_labels_for_multimodal(
            ids, None, None, None, None, itensor, ["image"], image_sizes=[size])
        emb = res[4]
        L = emb.shape[1]
        n_img = L - (ids.shape[1] - 1)
        masks = self.wte(torch.full((1, gen_len), MASK_ID, device=self.device)).to(emb.dtype)
        x = torch.cat([emb, masks], dim=1)
        return {"x": x, "L": L, "n_pre": n_pre, "n_img": n_img, "pos": L, "size": size, "orig": orig,
                "ids": ids, "itensor": itensor, "gen_len": gen_len}

    # ------------------------------------------------------------------ forward
    def bias(self, total_len, L, mode):
        """None for full; boolean (1,1,T,T) allowed-mask for prefix (True = may attend)."""
        if mode == "full":
            return None
        q = torch.arange(total_len, device=self.device)[:, None]
        k = torch.arange(total_len, device=self.device)[None, :]
        allowed = (k < L) | (q >= L)
        return allowed[None, None]

    @torch.no_grad()
    def logits(self, x, L, mode="prefix"):
        """all-position logits of one step-0 forward, (T, V)"""
        out = self.inner(None, input_embeddings=x, attention_bias=self.bias(x.shape[1], L, mode))
        return out.logits[0]

    @torch.no_grad()
    def decision_logits(self, run, mode):
        """logits at the first response position. full: one bidirectional forward.
        prefix: the official KV-cache path (verified). The attention_bias route is ignored by the
        model (selfcheck: full == bias-prefix), so it is NOT used for prefix here."""
        if mode == "full":
            return self.logits(run["x"], run["L"], "full")[run["L"]]
        return self.logits_cache_path(run)[0]

    @torch.no_grad()
    def prompt_kv(self, run):
        """official prefix path, part 1: KV cache of the prompt alone (prompt never sees the response)"""
        return self.inner(None, input_embeddings=run["x"][:, :run["L"]], use_cache=True).attn_key_values

    @torch.no_grad()
    def response_logits(self, run, pkv):
        """official prefix path, part 2: response positions attending to the cached prompt.
        returns (gen_len, V); the decision position is index 0. pkv is not modified, so it can be reused."""
        return self.inner(None, input_embeddings=run["x"][:, run["L"]:], past_key_values=pkv).logits[0]

    @torch.no_grad()
    def logits_cache_path(self, run):
        """official prefix path: prompt KV first, then response positions only. returns (gen_len, V)"""
        emb = run["x"][:, :run["L"]]
        resp = run["x"][:, run["L"]:]
        pkv = self.inner(None, input_embeddings=emb, use_cache=True).attn_key_values
        out = self.inner(None, input_embeddings=resp, past_key_values=pkv)
        return out.logits[0]

    @torch.no_grad()
    def generate(self, image_path, question, max_new_tokens=64, block_length=64, step_ratio=0.5):
        """official LaViDa decoding (prefix_lm + KV cache, shifted schedule)"""
        size, _, itensor = self.image(image_path)
        ids = self.prompt_ids(question)
        out = self.model.generate(ids, images=itensor, image_sizes=[size], do_sample=False, temperature=0.1,
                                  max_new_tokens=max_new_tokens, block_length=block_length,
                                  step_ratio=step_ratio, tokenizer=self.tok, prefix_lm=True,
                                  verbose=False, schedule="shift")
        seq = out[0] if isinstance(out, (tuple, list)) else out
        return self.tok.batch_decode(seq, skip_special_tokens=True)[0].strip()