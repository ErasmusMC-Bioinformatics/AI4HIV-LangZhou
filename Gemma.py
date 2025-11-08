from sklearn.metrics import classification_report, matthews_corrcoef
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from transformers import pipeline, BitsAndBytesConfig

import argparse
from typing import Optional
import os
import time
import traceback

from experimental import run_unsloth, run_llama
from utilities import parse_gemma_output, build_dataset, build_dataset_with_add, send_email
from prompt import SYSTEM_PROMPT


def run(prompt: str, pipe):
    # Execute a single prompt and get parsed response from LLM

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    output = pipe(messages, max_new_tokens=131072)
    response = output[0]["generated_text"][-1]["content"]

    decision = parse_gemma_output(response)

    return decision, response


def evaluate(dataset: str | pd.DataFrame, pipe):
    # Execute a prompt dataset and evaluate against the provided labels

    if isinstance(dataset, str):
        dataset = pd.read_csv(dataset)

    predictions, outputs = [], []

    for idx, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        prompt = row['prompt']

        for _ in range(1):
            if isinstance(pipe, tuple):
                prediction, output = run_unsloth(prompt, pipe)
            elif isinstance(pipe, str):
                prediction, output, prob = run_llama(prompt, pipe)

                predictions.append({'prediction': prediction, 'prob': prob})
                outputs.append({'output': output})
            else:
                prediction, output = run(prompt, pipe)

    predictions = pd.DataFrame(predictions)
    outputs = pd.DataFrame(outputs)

    # Save prediction results to unique folder for future inspections
    save_name = f'experiment_{int(time.time())}'
    os.makedirs(save_name, exist_ok=True)

    predictions.to_csv(f'{save_name}/predictions.csv', index=False)
    outputs.to_csv(f'{save_name}/outputs.csv', index=False)

    labels = dataset['label'].to_numpy().astype(int)

    print(classification_report(labels, predictions['prediction'].to_numpy(), target_names=['Exclusion', 'Inclusion']))
    print(f"MCC: {matthews_corrcoef(labels, predictions['prediction'].to_numpy())}")


def main(backend: str, bit: Optional[int], dataset: str, add: bool = False,):
    assert backend in ['hf', 'unsloth', 'llama'], 'Invalid backend specified.'
    assert bit in [4, 8, 16] or bit is None, 'Invalid quantization configuration.'

    if add:
        dataset = build_dataset_with_add(dataset, 'med.csv', 'labtests.csv')
    else:
        dataset = build_dataset(dataset)

    if backend == 'hf':
        if bit == 4:
            q_config = BitsAndBytesConfig(load_in_4bit=True)
        elif bit == 8:
            q_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            q_config = BitsAndBytesConfig()

        pipe = pipeline(
            "text-generation",
            model="google/medgemma-27b-text-it",
            torch_dtype=torch.bfloat16,
            model_kwargs={"quantization_config": q_config}
        )
    elif backend == 'unsloth':
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/medgemma-27b-text-it-unsloth-bnb-4bit" if bit == 4 else 'unsloth/medgemma-27b-text-it',
            max_seq_length=131072,
            dtype=torch.bfloat16,
            load_in_4bit=True if bit == 4 else False,
        )

        FastLanguageModel.for_inference(model)

        tokenizer = get_chat_template(
            tokenizer,
            chat_template="gemma3",
        )

        pipe = (model, tokenizer)
    else:
        pipe = 'http://localhost:8000/v1/chat/completions'

    try:
        evaluate(dataset, pipe)
    except Exception:
        send_email(f"[FAILED]", traceback.format_exc())
        raise
    else:
        send_email(f"[COMPLETED]", "Experiment finished successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run evaluation experiment."
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        default="llama",
        help="Backend to use ['hf', 'unsloth', 'llama'], currently support llama.cpp (default), huggingface or unsloth (limited model options)"
    )
    parser.add_argument(
        "--bit",
        type=int,
        default=None,
        help="Quantization to use (4, 8). If omitted, defaults to BF16."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='dataset.csv',
        required=True,
        help="Dataset path."
    )
    parser.add_argument(
        "--add",
        action="store_true",
        default=False,
        help="Whether to use additional medication/labtest data, default False."
    )

    args = parser.parse_args()
    main(args.backend, args.bit, args.dataset, args.add)
