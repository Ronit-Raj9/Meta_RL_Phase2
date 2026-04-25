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
        --output checkpoints/sft_warmup \
        --report-to wandb \
        --wandb-run-name my-sft-experiment

W&B logging
-----------
* Every ``logging_steps`` steps: TRL's built-in train/loss curves.
* Every ``--sample-every`` steps: a sample-completion ``wandb.Table``
  showing prompt, model completion, parser X/Z output, and parse status.
* End of training: a ``run.summary`` dict with final parse-success rate
  and sample count. The LoRA adapter directory is uploaded as an
  artifact so downstream GRPO runs can pull it by reference.
"""
from __future__ import annotations

import argparse
import json
import os
import random
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


# --------------------------------------------------------------------------- #
# W&B sample-completion callback                                              #
# --------------------------------------------------------------------------- #


def _build_wandb_callback(model, tokenizer, dataset, sample_every: int,
                          n_samples: int, max_new_tokens: int):
    """Return a ``TrainerCallback`` that logs a generation table to W&B.

    Pulls ``n_samples`` random prompts from the held-out tail of the SFT
    dataset and runs greedy decoding on them every ``sample_every`` steps.
    The resulting ``wandb.Table`` columns are::

        step | prompt | gold_completion | model_completion |
        x_pred | z_pred | parse_success | parse_partial

    Plus a scalar ``train/sft_parse_success_rate`` so the rate appears on
    the line plot too.
    """
    from transformers import TrainerCallback

    from qubit_medic import wandb_utils
    from qubit_medic.prompts import parse_action

    if not wandb_utils.is_available():
        return None

    if len(dataset) < n_samples + 4:
        eval_indices = list(range(len(dataset)))
    else:
        eval_indices = list(range(len(dataset) - n_samples, len(dataset)))

    class _SampleCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):  # noqa: D401
            if state.global_step == 0 or state.global_step % sample_every != 0:
                return
            self._log(state.global_step)

        def on_train_end(self, args, state, control, **kwargs):  # noqa: D401
            self._log(state.global_step, table_name="sft/final_generations")

        def _log(self, step: int, table_name: str = "sft/generations"):
            try:
                from unsloth import FastLanguageModel
                FastLanguageModel.for_inference(model)
            except Exception:
                model.eval()  # type: ignore[attr-defined]

            rows: list[dict] = []
            n_success, n_partial = 0, 0
            for idx in eval_indices:
                rec = dataset[idx]
                chat_in = (
                    "<|im_start|>user\n"
                    f"{rec['prompt']}\n<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                inputs = tokenizer(chat_in, return_tensors="pt").to(model.device)
                try:
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    completion = tokenizer.decode(
                        out[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                except Exception as exc:  # pragma: no cover
                    completion = f"<gen-error: {exc}>"
                num_data = 9  # primary distance-3 surface code
                parsed = parse_action(completion, num_data_qubits=num_data)
                n_success += int(parsed.parse_success)
                n_partial += int(parsed.parse_partial and not parsed.parse_success)
                rows.append({
                    "step": step,
                    "prompt": rec["prompt"][:600],
                    "gold_completion": rec["completion"],
                    "model_completion": completion[:300],
                    "x_pred": ",".join(map(str, parsed.x_errors)),
                    "z_pred": ",".join(map(str, parsed.z_errors)),
                    "parse_success": parsed.parse_success,
                    "parse_partial": parsed.parse_partial,
                })

            n = max(1, len(rows))
            wandb_utils.log({
                "sft/parse_success_rate": n_success / n,
                "sft/parse_partial_rate": n_partial / n,
                "sft/parse_failure_rate": (n - n_success - n_partial) / n,
            }, step=step)
            wandb_utils.log_generation_table(
                rows, step=step, table_name=table_name,
                columns=["step", "prompt", "gold_completion", "model_completion",
                         "x_pred", "z_pred", "parse_success", "parse_partial"],
            )

            try:
                from unsloth import FastLanguageModel
                FastLanguageModel.for_training(model)
            except Exception:
                model.train()  # type: ignore[attr-defined]

    return _SampleCallback()


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


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
    parser.add_argument("--report-to", type=str, default="wandb",
                        help="`wandb`, `tensorboard`, `none`. Falls back to "
                             "`none` if wandb is requested but unavailable.")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name; auto-generated if omitted.")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="W&B group name to bundle SFT+GRPO+eval runs.")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=("sft",),
                        help="Extra W&B tags appended to the project defaults.")
    parser.add_argument("--wandb-notes", type=str, default=None,
                        help="Free-text notes pinned to the W&B run.")
    parser.add_argument("--sample-every", type=int, default=100,
                        help="Log a sample-completion table every N steps.")
    parser.add_argument("--sample-count", type=int, default=4,
                        help="Number of completions per sample-table snapshot.")
    parser.add_argument("--no-artifact", action="store_true",
                        help="Skip uploading the LoRA adapter dir as a W&B artifact.")
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

    from qubit_medic import wandb_utils
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

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- W&B init (no-op if unavailable / disabled) -------------------- #
    report_to = wandb_utils.derive_report_to(args.report_to)
    run_name = args.wandb_run_name or wandb_utils.make_run_name("sft")
    wandb_utils.init_run(
        run_name=run_name,
        job_type="sft",
        tags=args.wandb_tags,
        notes=args.wandb_notes,
        group=args.wandb_group,
        extra_config={
            "cli": {
                "epochs": epochs,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "lr": lr,
                "max_seq_len": max_seq_len,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "dataset_path": args.dataset,
                "model": model_id,
                "seed": seed,
                "report_to": report_to,
            },
        },
    )

    # ---- Load model + dataset ----------------------------------------- #
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
    wandb_utils.log({
        "sft/dataset_size": len(dataset),
        "sft/first_text_len": len(dataset[0]["text"]),
    })
    wandb_utils.log_generation_table(
        [
            {"split": "preview", "prompt": dataset[i]["prompt"][:600],
             "completion": dataset[i]["completion"]}
            for i in range(min(8, len(dataset)))
        ],
        step=0,
        table_name="sft/dataset_preview",
        columns=["split", "prompt", "completion"],
    )

    # ---- TrainingArguments ------------------------------------------- #
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
        report_to=report_to,
        run_name=run_name,
    )

    callbacks = []
    cb = _build_wandb_callback(model, tokenizer, dataset,
                               sample_every=args.sample_every,
                               n_samples=args.sample_count,
                               max_new_tokens=160)
    if cb is not None:
        callbacks.append(cb)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        args=training_args,
        packing=False,
        callbacks=callbacks,
    )

    print("training...")
    train_result = trainer.train()
    metrics = getattr(train_result, "metrics", {}) or {}
    if metrics:
        wandb_utils.update_summary({f"sft/final/{k}": v for k, v in metrics.items()
                                    if isinstance(v, (int, float))})

    print(f"saving adapters to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # ---- Upload adapter as W&B artifact -------------------------------- #
    if not args.no_artifact:
        wandb_utils.log_artifact(
            args.output,
            name=f"sft-adapter-{run_name}",
            artifact_type="model",
            description="SFT-warmed Qwen2.5-3B + LoRA adapter (Qubit-Medic).",
        )

    wandb_utils.finish_run()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
