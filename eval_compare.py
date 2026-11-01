"""
eval_compare.py - base model vs base+LoRA on the HELD-OUT test set.

Run:  python eval_compare.py
      python eval_compare.py --show 5      # print 5 example outputs side by side

Metrics (all computed by code - this task has exact answers, so no LLM judge needed):
    valid_json   - output parses as JSON with exactly the 4 keys
    intent/product/urgency/order_id accuracy - per field
    exact        - all 4 fields right
Decoding is GREEDY (temperature 0) so results are reproducible run to run.
"""

from __future__ import annotations

import argparse
import json
import re
import time

import torch

from common_ft import ADAPTER_DIR, DATA_DIR, DEVICE, chat_prompt, load_base

p = argparse.ArgumentParser()
p.add_argument("--show", type=int, default=3)
p.add_argument("--limit", type=int, default=60)
args = p.parse_args()

rows = [json.loads(l) for l in open(DATA_DIR / "test.jsonl", encoding="utf-8")][: args.limit]
KEYS = ["intent", "product", "urgency", "order_id"]


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)       # tolerate chatter around the JSON
    try:
        d = json.loads(m.group(0)) if m else None
        return d if isinstance(d, dict) and set(d) == set(KEYS) else None
    except json.JSONDecodeError:
        return None


@torch.no_grad()
def run(model, tok) -> tuple[dict, list[str]]:
    outs, score = [], {k: 0 for k in ["valid_json", *KEYS, "exact"]}
    t0 = time.time()
    for r in rows:
        enc = tok(chat_prompt(tok, r["input"]), return_tensors="pt").to(DEVICE)
        gen = model.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
        text = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        outs.append(text)
        pred, gold = parse(text), json.loads(r["output"])
        if pred:
            score["valid_json"] += 1
            hits = [pred.get(k) == gold[k] for k in KEYS]
            for k, h in zip(KEYS, hits):
                score[k] += h
            score["exact"] += all(hits)
    n = len(rows)
    score = {k: round(v / n, 2) for k, v in score.items()}
    score["sec_per_msg"] = round((time.time() - t0) / n, 2)
    return score, outs


tok, base = load_base()
print(f"evaluating BASE on {len(rows)} held-out messages...")
base_score, base_out = run(base, tok)

if not ADAPTER_DIR.exists():
    raise SystemExit(f"No adapter at {ADAPTER_DIR}. Run train_lora.py first.")
from peft import PeftModel  # noqa: E402
tuned = PeftModel.from_pretrained(base, ADAPTER_DIR)
print("evaluating BASE + LoRA...")
tuned_score, tuned_out = run(tuned, tok)

print(f"\n{'metric':14} {'base':>6} {'lora':>6}")
for k in base_score:
    print(f"{k:14} {base_score[k]:>6} {tuned_score[k]:>6}")

for r, b, t in list(zip(rows, base_out, tuned_out))[: args.show]:
    print(f"\nMSG:  {r['input']}\nGOLD: {r['output']}\nBASE: {b[:160]}\nLORA: {t[:160]}")
