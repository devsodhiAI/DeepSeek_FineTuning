"""
train_lora.py - LoRA fine-tune with the Hugging Face Trainer.

Run:  python train_lora.py            (~2-5 min on an RTX 5050 with the 0.5B model)

LoRA in one breath:
    Full fine-tuning updates ALL weights (0.5B numbers -> big GPU, big files).
    LoRA FREEZES the model and adds tiny trainable matrices next to chosen
    layers: W' = W + (B @ A) * (alpha / r), where A is (r x d) and B is (d x r),
    with r small (8). We train <1% of the parameters; the adapter file is a few MB.
    QLoRA = the same, but the frozen base model is loaded in 4-bit to save even
    more memory (needs `pip install bitsandbytes`; see the --qlora flag).

KNOBS (CLI flags) - change ONE at a time, then run eval_compare.py:
    --r 8            LoRA rank: capacity of the adapter (try 2, 8, 32)
    --alpha 16       scaling; rule of thumb alpha = 2 * r
    --targets q_proj v_proj     which layers get adapters (try adding k_proj o_proj gate_proj up_proj down_proj)
    --lr 2e-4        LoRA likes higher LR than full fine-tuning (try 2e-5 and 2e-3)
    --epochs 2       too many -> overfits (memorises train, worse on test)
    --n-train 240    how many examples to use (try 30)
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

from common_ft import ADAPTER_DIR, BASE_MODEL, DATA_DIR, DEVICE, chat_prompt, load_base

p = argparse.ArgumentParser()
p.add_argument("--r", type=int, default=8)
p.add_argument("--alpha", type=int, default=16)
p.add_argument("--dropout", type=float, default=0.05)
p.add_argument("--targets", nargs="+", default=["q_proj", "v_proj"])
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--epochs", type=float, default=2)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--n-train", type=int, default=240)
p.add_argument("--max-len", type=int, default=160)
p.add_argument("--qlora", action="store_true", help="load base in 4-bit (needs bitsandbytes)")
args = p.parse_args()

# ------------------------------------------------------------------ model
if args.qlora:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.pad_token or tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
else:
    tok, model = load_base()

lora = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                  target_modules=args.targets, task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.print_trainable_parameters()      # e.g. "trainable: 0.54M || all: 494M || 0.11%"

# ------------------------------------------------------------------ data
rows = [json.loads(l) for l in open(DATA_DIR / "train.jsonl", encoding="utf-8")][: args.n_train]


def encode(row: dict) -> dict:
    """
    Tokenise prompt + answer, and MASK THE PROMPT in the labels (-100).
    Loss is then computed only on the answer tokens: we teach the model to
    WRITE the JSON, not to predict the customer's message. Forgetting this is
    a classic bug - the model wastes capacity learning to echo inputs.
    """
    prompt_ids = tok(chat_prompt(tok, row["input"]), add_special_tokens=False)["input_ids"]
    answer_ids = tok(row["output"] + tok.eos_token, add_special_tokens=False)["input_ids"]   # EOS teaches it to STOP
    ids = (prompt_ids + answer_ids)[: args.max_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[: args.max_len]
    return {"input_ids": ids, "labels": labels}


train = [encode(r) for r in rows]
print(f"{len(train)} examples, longest {max(len(x['input_ids']) for x in train)} tokens")


def collate(batch: list[dict]) -> dict:
    """Pad a batch to the same length. Padding positions: label -100 and attention 0."""
    n = max(len(x["input_ids"]) for x in batch)
    pad = tok.pad_token_id
    return {
        "input_ids": torch.tensor([x["input_ids"] + [pad] * (n - len(x["input_ids"])) for x in batch]),
        "labels": torch.tensor([x["labels"] + [-100] * (n - len(x["labels"])) for x in batch]),
        "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * (n - len(x["input_ids"])) for x in batch]),
    }


# ------------------------------------------------------------------ train
targs = TrainingArguments(
    output_dir=str(ADAPTER_DIR / "_checkpoints"),
    per_device_train_batch_size=args.batch,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    lr_scheduler_type="cosine", warmup_steps=3,   # ramp LR up for 3 steps, then decay smoothly (transformers 5 dropped warmup_ratio)
    logging_steps=5,
    save_strategy="no",
    report_to=[],
    bf16=DEVICE == "cuda",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)
trainer = Trainer(model=model, args=targs, train_dataset=train, data_collator=collate)

t0 = time.time()
result = trainer.train()
print(f"\ntrained in {time.time() - t0:.0f}s, final loss {result.training_loss:.4f}")
if DEVICE == "cuda":
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

model.save_pretrained(ADAPTER_DIR)       # saves ONLY the adapter (a few MB), not the base model
(ADAPTER_DIR / "train_args.json").write_text(json.dumps(vars(args), indent=2))
print(f"adapter saved to {ADAPTER_DIR}")
