# LoRA fine-tuning an open-weight LLM (Qwen 2.5 / DeepSeek-R1-Distill)

Teaching a small open-weight model to return strict, correct JSON for customer-support triage with LoRA (PEFT + Hugging Face Trainer), evaluated on a held-out set with field-level metrics. Also part of [genai-production-labs](https://github.com/devsodhiAI/genai-production-labs) (lab 04).

Base model is a knob: `Qwen/Qwen2.5-0.5B-Instruct` (default, results below) or `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` via `FT_BASE_MODEL`.

## Summary

> I took a small open-weight model that couldn't reliably follow "reply with exactly this JSON" and taught it the format with LoRA: frozen base weights plus tiny trainable low-rank matrices on the attention projections, under 1% of the parameters. I used the HF Trainer, masked the prompt tokens so the loss is only on the answer, and appended EOS so it learns to stop. I evaluated on a held-out test set that has no overlap with training, using greedy decoding and code metrics: valid-JSON rate, per-field accuracy, exact match, base vs fine-tuned. Fine-tuning is for behaviour and format; facts belong in RAG.

## Files (in order)

| File | What to learn |
|---|---|
| `common_ft.py` | Base model choice, bf16, chat templates |
| `sampling_playground.py` | Temperature / top-p / top-k, *seen* as probabilities and as outputs |
| `make_dataset.py` | Instruction data format, deterministic generation, train/test leakage guard |
| `train_lora.py` | LoRA config, label masking, EOS, collator/padding, Trainer, adapter saving, QLoRA path |
| `eval_compare.py` | Held-out eval, greedy decoding, code metrics, base vs tuned |

## Run (no API key)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # install CUDA torch first for GPU
python sampling_playground.py
python make_dataset.py
python train_lora.py
python eval_compare.py --show 5
```

## What it showed on this machine (2026-10-27, RTX 5050 8 GB, Qwen2.5-0.5B-Instruct)

**Sampling** (prompt: "tagline for a new mango ice cream"):
- Next-token probability of the top token: **1.00 at T=0.3, 0.74 at T=1.0, 0.03 at T=2.0**
- T=0 gave `"Sweetness in Every Bite"` both times, identically
- T=0.7 gave varied, sensible taglines
- T=1.5 gave gibberish mixing scripts and languages: the tail of the distribution takes over

**Fine-tune** (r=8, q/v projections, 240 examples, 2 epochs): **0.11% of parameters trained, 11 seconds, 3.2 GB peak GPU**. Loss 0.77 → 0.09.

| metric (60 unseen messages) | base | + LoRA |
|---|---|---|
| valid JSON | 1.00 | 1.00 |
| intent | 0.02 | **0.73** |
| product | 0.33 | **0.95** |
| urgency | 0.00 | 0.58 |
| order_id | 0.53 | 0.78 |
| exact (all 4 right) | 0.00 | **0.35** |

Takeaways:
- **Valid JSON was 1.00 for both — format isn't correctness.** The base model wrote nice JSON with made-up labels (`"intent": "Request"`) and **invented an order ID** (`"12345"`) when there was none. That's a hallucination you'd only catch with field-level evals.
- LoRA learned the label vocabulary and the "null when absent" rule.
- **Urgency stays weakest (0.58)** because my labels are partly rule-based on typo-sensitive keywords, so the labels themselves are noisy. Label quality caps model accuracy. Next step: clean the labelling rules, not train harder.
- 35% exact match isn't production-ready. Next: more varied data, more epochs or target modules, then re-evaluate (experiments below).

## Experiments (one knob at a time, re-run eval each time)

1. `python sampling_playground.py --temps 0.7 --top-p 0.3` vs `--top-p 0.95`. top-p cuts the crazy tail even at higher temperature.
2. `python train_lora.py --n-train 30` — how much data do you really need?
3. `--r 2` vs `--r 32` — capacity vs overfitting.
4. `--lr 2e-3` (too high: watch the loss spike) and `--lr 2e-5` (too low: barely learns).
5. `--epochs 6` — does test accuracy go *down* while training loss keeps falling? That's overfitting.
6. `--targets q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` — more modules: more parameters, better?
7. **The DeepSeek one:** `$env:FT_BASE_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"`, then run `eval_compare.py` before training (it'll say no adapter, but you'll see base outputs with `<think>`), then train and compare. Takes longer — 1.5B parameters.
8. **Break it on purpose:** in `train_lora.py`, change `[-100] * len(prompt_ids)` to `prompt_ids` (no masking). Retrain and compare.
9. QLoRA: `pip install bitsandbytes`, then `python train_lora.py --qlora`. Compare peak GPU memory and time.

