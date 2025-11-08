from huggingface_hub import login

from utilities import parse_gemma_output, build_dataset, calculate_cumulate_logprob
from prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_COMPLEX

import requests


login(token='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')    # This token has been expired


def run_unsloth(prompt: str, pipe):
    # Execute a single prompt and get parsed response from LLM
    model, tokenizer = pipe

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=131072,
        temperature=1.0,
        top_k=64,
        top_p=0.95,
    )

    gen_tokens = outputs[0][inputs["input_ids"].size(-1):]

    response = tokenizer.decode(
        gen_tokens,
        skip_special_tokens=True,
    )

    decision = parse_gemma_output(response)

    return decision, response


def run_llama(prompt: str, ip: str):
    # This method interacts with an active llama_cpp server (or any OpenAI API compatible service) to obtain the model response
    # For local testing, make sure a server is running first
    # Example command to start server (with max context window and full GPU offloading):
    # llama-server -hf unsloth/medgemma-27b-text-it-GGUF:Q4_K_M -c 131072 -ngl 63
    # Q4_K_M can be changed to other quantized versions

    headers = {"Content-Type": "application/json"}

    payload = {
        "model": "any",  # this value is ignored by llama-server
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_COMPLEX},
            {"role": "user", "content": prompt}
        ],
        "temperature": 1.0,
        "top_k": 64,
        "top_p": 0.95,
        "min_p": 0.0,
        "logprobs": True
    }

    try:
        resp = requests.post(ip, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        raw_response = data["choices"][0]["message"]["content"]
        decision = parse_gemma_output(raw_response)  # may raise ValueError

        prob = calculate_cumulate_logprob(data)
        return decision, raw_response, prob
    except:     # All errors are from too long input
        mid = len(prompt) // 2
        left_prompt, right_prompt = prompt[:mid], prompt[mid:]

        # Recursively process the two halves
        dec1, resp1, prob1 = run_llama(left_prompt, ip)
        dec2, resp2, prob2 = run_llama(right_prompt, ip)

        combined_decision = dec1 or dec2
        combined_resp = f"{resp1}\n{resp2}"
        combined_prob = 0.5 * (prob1 + prob2)

        return combined_decision, combined_resp, combined_prob


def main():
    run_llama('Who are you? Answer in short responses.', 'http://localhost:8000/v1/chat/completions')


if __name__ == '__main__':
    main()
