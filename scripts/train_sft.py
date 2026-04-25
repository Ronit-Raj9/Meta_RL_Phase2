"""scripts/train_sft.py - SFT warm-up phase (Section 6 of the plan).

Loads Qwen2.5-3B-Instruct in 4-bit via Unsloth, attaches a LoRA adapter
(rank 16, alpha 32, on q/k/v/o projections), and runs a single epoch of
supervised fine-tuning on ``data/sft_dataset.jsonl``.

Goal: take the base model from ~0% format compliance to ~95%+ so the GRPO
trainer has a non-zero probability of getting a parseable reward.

Designed to run on a Colab T4 in ~30 minutes. The Stim/PyMatching server
itself does *not* import any of these heavy ML deps - they live in
``requirements-train.txt``.

Run::

    pip install -r requirements-train.txt
    python -m scripts.train_sft \
        --dataset data/sft_dataset.jsonl \
        --output checkpoints/sft_warmup
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable


def _load_dataset(path: str):
    """Load the SFT JSONL into a HuggingFace Dataset."""
    from datasets import Dataset

    rows = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            rows.append({
                "prompt": rec["prompt"],
                "completion": rec["completion"],
                # Concatenate for SFTTrainer's "text" field with chat-style
                # markers so the model learns "after this prompt, produce
                # the formatted answer".
                "text": (
                    "<|im_start|>user\n"
                    f"{rec['prompt']}\n<|im_end|>\n"
                    "<|im_start|>assistant\n"
                    f"{rec['completion']}<|im_end|>"
                ),
            })
    return Dataset.from_list(rows)


def main(argv: Iterable[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="data/sft_dataset.jsonl")
    parser.add_argument("--output", type=str, default="checkpoints/sft_warmup")
    parser.add_argument("--model", type=str,
                        default=os.getenv("QUBIT_MEDIC_MODEL", "Qwen/Qwen2.5-3B-Instruct"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--report-to", type=str, default="none",
                        help="`wandb` or `none`. W&B requires WANDB_API_KEY.")
    args = parser.parse_args(list(argv))

    # Heavy imports are lazy so this module is importable without GPU deps.
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth not installed. Run `pip install -r requirements-train.txt`",
              file=sys.stderr)
        return 1
    import torch
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from peft import LoraConfig

    from qubit_medic.config import (
        LORA_ALPHA, LORA_R, LORA_TARGET_MODULES, MODEL_ID, PRIMARY_SEED,
        SFT_BATCH_SIZE, SFT_EPOCHS, SFT_GRAD_ACCUM, SFT_LR, SFT_MAX_SEQ_LEN,
    )

    epochs = args.epochs if args.epochs is not None else SFT_EPOCHS
    batch_size = args.batch_size if args.batch_size is not None else SFT_BATCH_SIZE
    grad_accum = args.grad_accum if args.grad_accum is not None else SFT_GRAD_ACCUM
    lr = args.lr if args.lr is not None else SFT_LR
    max_seq_len = args.max_seq_len if args.max_seq_len is not None else SFT_MAX_SEQ_LEN
    seed = args.seed if args.seed is not None else PRIMARY_SEED
    lora_r = args.lora_r if args.lora_r is not None else LORA_R
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else LORA_ALPHA
    model_id = args.model if args.model else MODEL_ID

    print(f"loading {model_id} via Unsloth (4-bit)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_len,
        load_in_4bit=True,
        dtype=None,  # Unsloth auto-selects bf16/fp16
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=list(LORA_TARGET_MODULES),
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    print(f"loading dataset from {args.dataset}")
    dataset = _load_dataset(args.dataset)
    print(f"  {len(dataset)} samples; first text len = {len(dataset[0]['text'])}")

    Path(args.output).mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=50,
        lr_scheduler_type="linear",
        bf16=torch.cuda.is_available()
            and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available()
            and not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        seed=seed,
        report_to=args.report_to,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        args=training_args,
        packing=False,
    )

    print("training...")
    trainer.train()

    print(f"saving adapters to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
