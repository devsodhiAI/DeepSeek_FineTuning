"""
common_ft.py - shared settings for lab 04.

KNOB: FT_BASE_MODEL
    Qwen/Qwen2.5-0.5B-Instruct                (default - trains in a few minutes on an 8 GB GPU)
    deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (the DeepSeek distill. 3x bigger and
        slower, and it's a REASONING model: it writes <think>...</think> before answering,
        which wrecks strict JSON output until fine-tuned. A great thing to observe.)
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from pathlib import Path  # noqa: E402

import torch  # noqa: E402

BASE_MODEL = os.environ.get("FT_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
ADAPTER_DIR = HERE / "adapters" / BASE_MODEL.split("/")[-1]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# bf16 on GPU: half the memory of fp32, same range. CPU stays fp32.
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

SYSTEM = ("Extract the customer's request as JSON with keys: intent, product, urgency, order_id. "
          "Reply with JSON only.")


def load_base(model_name: str = BASE_MODEL):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE).to(DEVICE)
    return tok, model


def chat_prompt(tok, user_msg: str) -> str:
    """Use the model's OWN chat template. A wrong template = garbage outputs."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
