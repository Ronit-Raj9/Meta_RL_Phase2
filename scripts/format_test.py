"""scripts/format_test.py - the Section 1.3 existential go/no-go check.

Prompts a model with 10 hand-crafted syndromes (3 samples each) and counts
how many of the 30 outputs are parseable in our exact ``X_ERRORS=[...]
Z_ERRORS=[...]`` format. Above 30% means RL can be started directly; below
means SFT is mandatory.

Usage::

    python -m scripts.format_test                       # uses HF Inference API or Unsloth
    python -m scripts.format_test --backend unsloth     # local 4-bit
    python -m scripts.format_test --backend hf          # HF Inference API
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable

from qubit_medic.client.client import LocalDecoderClient
from qubit_medic.config import (
    CURRICULUM, MODEL_ID, SAMPLE_TEMPERATURE, SAMPLE_TOP_P,
)
from qubit_medic.prompts import parse_action


def _generate_unsloth(model_id: str, prompts: list[str], n_samples: int) -> list[str]:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    out = []
    for prompt in prompts:
        chat = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(chat, tokenize=False,
                                             add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        for _ in range(n_samples):
            gen = model.generate(
                **inputs, max_new_tokens=160,
                do_sample=True,
                temperature=SAMPLE_TEMPERATURE,
                top_p=SAMPLE_TOP_P,
                eos_token_id=tokenizer.eos_token_id,
            )
            comp = tokenizer.decode(
                gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            out.append(comp)
    return out


def _generate_hf(model_id: str, prompts: list[str], n_samples: int) -> list[str]:
    """Use HuggingFace's hosted Inference API. Requires HF_TOKEN."""
    import requests
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN env var required for --backend hf")
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    out = []
    for prompt in prompts:
        for _ in range(n_samples):
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": SAMPLE_TEMPERATURE,
                    "top_p": SAMPLE_TOP_P,
                    "max_new_tokens": 160,
                    "do_sample": True,
                },
                "options": {"wait_for_model": True},
            }
            r = requests.post(url, json=payload, headers=headers, timeout=120)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and "generated_text" in data[0]:
                out.append(data[0]["generated_text"][len(prompt):])
            else:
                out.append(json.dumps(data))
    return out


def _generate_dummy(prompts: list[str], n_samples: int) -> list[str]:
    """Offline/CI mode: fake responses (10% well-formed, rest noise).

    Used by tests; lets the format_test script run without network or GPU.
    """
    out = []
    for i, _ in enumerate(prompts):
        for j in range(n_samples):
            if (i + j) % 3 == 0:
                out.append("X_ERRORS=[1] Z_ERRORS=[]")
            elif (i + j) % 3 == 1:
                out.append("Step 1: I think the error happened on qubit 3...")
            else:
                out.append("```\nX_ERRORS=[2,5] Z_ERRORS=[1]\n```")
    return out


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["unsloth", "hf", "dummy"],
                        default="dummy")
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--syndromes", type=int, default=10)
    parser.add_argument("--samples-per", type=int, default=3)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(list(argv))

    client = LocalDecoderClient()
    prompts = []
    for i in range(args.syndromes):
        obs = client.reset(forced_level="L2_target", seed=20_000 + i)
        prompts.append(obs.prompt)

    print(f"backend={args.backend}  model={args.model}  "
          f"syndromes={args.syndromes}  samples_per={args.samples_per}")

    if args.backend == "unsloth":
        completions = _generate_unsloth(args.model, prompts, args.samples_per)
    elif args.backend == "hf":
        completions = _generate_hf(args.model, prompts, args.samples_per)
    else:
        completions = _generate_dummy(prompts, args.samples_per)

    n = len(completions)
    parseable = 0
    partial = 0
    for c in completions:
        parsed = parse_action(c, num_data_qubits=9)
        if parsed.parse_success:
            parseable += 1
        elif parsed.parse_partial:
            partial += 1

    rate = parseable / max(1, n)
    verdict = "ABOVE 30% - RL can start directly" if rate >= 0.30 \
              else "BELOW 30% - SFT warmup is MANDATORY"
    print(f"format compliance: {parseable}/{n} = {rate:.2%} "
          f"(+{partial}/{n} partial)")
    print(verdict)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "backend": args.backend, "model": args.model,
                "n": n, "parseable": parseable, "partial": partial,
                "rate": rate, "verdict": verdict,
                "completions": completions,
            }, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
