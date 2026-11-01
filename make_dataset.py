"""
make_dataset.py - build a small instruction-tuning dataset. Free, offline, deterministic.

Run:  python make_dataset.py            -> data/train.jsonl (240), data/test.jsonl (60)

TASK: messy customer WhatsApp message -> strict JSON
    {"intent": ..., "product": ..., "urgency": "low|medium|high", "order_id": "QZ12345" | null}

Why this task for fine-tuning? Fine-tuning teaches BEHAVIOUR/FORMAT, not facts.
A 0.5B model can't follow "JSON only, these 4 keys, these exact labels"
reliably from a prompt. After ~250 examples it can - and it then runs locally,
fast and nearly free, instead of calling a big API model for every message.

Real life: you'd label real messages (or have a big model label them and
humans spot-check = "distillation"). Here templates simulate that, with
Hinglish and typos mixed in so it's not too easy.

KNOB: N_TRAIN - try 30 vs 240 examples and compare eval scores.
"""

import json
import random

from common_ft import DATA_DIR

N_TRAIN, N_TEST = 240, 60
random.seed(7)   # deterministic: same dataset every run -> comparable experiments

PRODUCTS = ["phone", "laptop", "tv", "ac", "fridge", "washing machine", "earphones", "water purifier"]

TEMPLATES = {
    "order_status": ["where is my {p} order {oid}", "{oid} {p} not delivered yet??", "mera {p} kab aayega order {oid}",
                     "track order {oid} pls, it's a {p}", "any update on my {p}? order no {oid}"],
    "return_request": ["want to return the {p}, order {oid}", "{p} not as expected, how to send back",
                       "return karna hai {p} {oid}", "i dont like the {p} i got, refund?"],
    "warranty_claim": ["my {p} stopped working, still under warranty", "{p} not turning on after 3 months, claim warranty",
                       "{p} kharab ho gaya warranty mein hai", "warranty repair needed for {p} order {oid}"],
    "emi_query": ["can i buy {p} on emi", "no cost emi on {p} with hdfc card?", "{p} monthly instalment possible?",
                  "emi options for {p} pls"],
    "installation_booking": ["need installation for my new {p}", "when will technician come to install {p} {oid}",
                             "{p} installation book karna hai", "pls schedule {p} fitting"],
    "complaint": ["worst service, {p} delivered damaged!!! {oid}", "delivery guy was rude, {p} box torn",
                  "very unhappy with {p}, nobody replies", "{p} broken on arrival {oid} fix this NOW"],
}

URGENT_WORDS = ["!!!", "NOW", "worst", "broken", "??", "stopped", "kharab"]


def urgency(intent: str, msg: str) -> str:
    if intent == "complaint" or any(w in msg for w in URGENT_WORDS):
        return "high"
    if intent in ("order_status", "warranty_claim", "installation_booking"):
        return "medium"
    return "low"


def typo(s: str) -> str:
    """Randomly drop a letter - real users don't spell-check."""
    if len(s) > 10 and random.random() < 0.3:
        i = random.randrange(len(s))
        return s[:i] + s[i + 1:]
    return s


def example() -> dict:
    intent = random.choice(list(TEMPLATES))
    p = random.choice(PRODUCTS)
    oid = f"QZ{random.randint(10000, 99999)}"
    tpl = random.choice(TEMPLATES[intent])
    msg = typo(tpl.format(p=p, oid=oid))
    label = {"intent": intent, "product": p, "urgency": urgency(intent, msg),
             "order_id": oid if "{oid}" in tpl and oid in msg else None}
    return {"input": msg, "output": json.dumps(label)}


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    train = [example() for _ in range(N_TRAIN)]
    # DATA LEAKAGE guard: a test message that also appears in train would
    # inflate the score (the model memorised it). Keep test strictly unseen.
    seen = {r["input"] for r in train}
    test = []
    while len(test) < N_TEST:
        r = example()
        if r["input"] not in seen:
            test.append(r)
            seen.add(r["input"])
    rows = train + test
    for name, part in (("train", train), ("test", test)):
        with open(DATA_DIR / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
        print(f"{name}: {len(part)} examples")
    print("sample:", rows[0])
