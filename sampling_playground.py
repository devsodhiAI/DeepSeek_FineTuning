"""
sampling_playground.py - SEE what temperature / top_p / top_k actually do.

Run:
    python sampling_playground.py
    python sampling_playground.py --prompt "Write a tagline for a mango ice cream" --temps 0 0.7 1.5 --n 3
    python sampling_playground.py --top-p 0.5
    python sampling_playground.py --top-k 5

How a model writes text: at each step it outputs a SCORE (logit) for every
token in its vocabulary (~150k). We turn scores into probabilities and pick one.
The sampling knobs only change HOW WE PICK:

    temperature T : probs = softmax(logits / T)
                    T < 1 sharpens (confident, repetitive), T > 1 flattens (creative, then random)
                    T = 0 -> "greedy": always take the top token -> deterministic
    top_k K       : only the K most likely tokens may be picked
    top_p P       : only the smallest set of tokens whose probs add up to P ("nucleus")

Why this runs on a LOCAL model: current Claude models don't accept these
params any more (API returns 400). Whoever runs the decoder owns the knobs.
"""

from __future__ import annotations

import argparse

import torch

from common_ft import DEVICE, load_base

p = argparse.ArgumentParser()
p.add_argument("--prompt", default="Write a one-line tagline for a new mango ice cream.")
p.add_argument("--temps", type=float, nargs="+", default=[0.0, 0.7, 1.5])
p.add_argument("--n", type=int, default=3, help="samples per temperature")
p.add_argument("--top-p", type=float, default=1.0)
p.add_argument("--top-k", type=int, default=0)
p.add_argument("--max-new", type=int, default=30)
args = p.parse_args()

tok, model = load_base()
msgs = [{"role": "user", "content": args.prompt}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(DEVICE)

# ---- Part 1: look inside one step: next-token probabilities at different T
with torch.no_grad():
    logits = model(**inputs).logits[0, -1].float()     # scores for the NEXT token only
print(f"\nPROMPT: {args.prompt}\n\nNext-token probabilities (top 8) at different temperatures:")
for T in [0.3, 1.0, 2.0]:
    probs = torch.softmax(logits / T, dim=-1)
    top = torch.topk(probs, 8)
    row = "  ".join(f"{tok.decode(i)!r}:{v:.2f}" for v, i in zip(top.values.tolist(), top.indices.tolist()))
    print(f"  T={T:<4} {row}")
print("  -> low T: the top token takes almost all probability. High T: the tail gets a real chance.")

# ---- Part 2: full generations
print(f"\nGenerations (top_p={args.top_p}, top_k={args.top_k or 'off'}):")
for T in args.temps:
    print(f"\n  temperature={T}")
    for _ in range(args.n):
        kw = dict(max_new_tokens=args.max_new, pad_token_id=tok.pad_token_id)
        if T == 0:
            kw["do_sample"] = False                     # greedy
        else:
            kw.update(do_sample=True, temperature=T, top_p=args.top_p, top_k=args.top_k or None)
        out = model.generate(**inputs, **kw)
        print("   -", tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
print("\nNotice: T=0 gives the SAME output every time. Use it for extraction/JSON, higher T for brainstorming.")
