#!/usr/bin/env python3
# make_jsonl_per_model.py
"""
Reads a CSV with a 'prompt' column (user inputs) and a single SYSTEM_PROMPT,
then for each model in MODELS:
  - applies the model's chat template (system + user),
  - writes a JSONL with lines like: {"prompt": "<templated text>"}
The JSONL files are ready for vLLM's "custom" dataset loader.

Edit the CONFIG section below and run:
  python make_jsonl_per_model.py
"""

# =======================
# CONFIG — EDIT THESE
# =======================
INPUT_CSV = "inputs.csv"  # CSV must contain a 'prompt' column
SYSTEM_PROMPT = "You are a helpful doctor. Keep answers concise."
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "openai/GPT-OSS-120B",
]
OUT_DIR = "out_jsonl"

# Optional knobs
ADD_GENERATION_PROMPT = True  # appends assistant-start marker for most chat templates
MAX_ROWS = None  # e.g., 100 to cap rows; or None for all
DROP_EMPTY = True  # skip blank/whitespace-only prompts
# =======================

from pathlib import Path
import json
import re
import pandas as pd
from transformers import AutoTokenizer


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)


def build_prompts_for_model(model_id: str, system_prompt: str, user_texts: list[str]) -> list[str]:
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if not hasattr(tok, "apply_chat_template"):
        raise RuntimeError(
            f"Tokenizer for {model_id} lacks apply_chat_template(). "
            "Use a chat-tuned model or add a manual template."
        )

    prompts = []
    for u in user_texts:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s and DROP_EMPTY:
            continue
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": s},
        ]
        out = tok.apply_chat_template(
            messages,
            tokenize=False,  # we want a raw string for vLLM 'prompt'
            add_generation_prompt=ADD_GENERATION_PROMPT,
        )
        prompts.append(out)
    return prompts


def main():
    # Load CSV
    df = pd.read_csv(INPUT_CSV)
    if "prompt" not in df.columns:
        raise SystemExit("ERROR: Input CSV must contain a 'prompt' column.")

    if MAX_ROWS is not None:
        df = df.head(MAX_ROWS)

    user_prompts = df["prompt"].astype(str).tolist()
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process each model
    for model_id in MODELS:
        print(f"[+] Processing model: {model_id}")
        prompts = build_prompts_for_model(
            model_id=model_id,
            system_prompt=SYSTEM_PROMPT,
            user_texts=user_prompts,
        )

        out_path = out_dir / f"{sanitize_filename(model_id)}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for p in prompts:
                f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")

        print(f"    -> Wrote {len(prompts)} lines to {out_path}")

    print("\nAll done. Reminder: when running vLLM bench, --model must equal your --served-model-name.")


if __name__ == "__main__":
    main()
