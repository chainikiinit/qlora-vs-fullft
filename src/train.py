"""Обучение: полный fine-tuning, LoRA (fp16) или QLoRA (4-bit NF4) на одной GPU.

python -m src.train --config configs/base.yaml configs/method/qlora.yaml --set seed=1 train.lr=2e-4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from .config import ensure_dir, load_config, make_run_name
from .data import build_datasets, tokenize_rows


class ResourceCallback(TrainerCallback):
    """Пиковая память GPU и время обучения (без загрузки модели)."""

    def __init__(self) -> None:
        self.stats: dict = {}

    def on_train_begin(self, args, state, control, **kw):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self._t0 = time.time()

    def on_train_end(self, args, state, control, **kw):
        torch.cuda.synchronize()
        self.stats = {
            "peak_mem_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
            "peak_mem_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
            "wall_time_s": time.time() - self._t0,
        }


def build_model(cfg: dict):
    m, method, t = cfg["model"], cfg["method"], cfg["train"]
    kw = dict(attn_implementation=m.get("attn_implementation", "sdpa"))

    if method == "full":
        # fp32-веса + fp16 autocast (Trainer fp16=True): мастер-веса нужны для полного FT
        model = AutoModelForCausalLM.from_pretrained(m["name"], torch_dtype=torch.float32, **kw)
    elif method == "lora":
        model = AutoModelForCausalLM.from_pretrained(m["name"], torch_dtype=torch.float16, **kw)
    elif method == "qlora":
        q = cfg["quant"]
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=q["quant_type"],
            bnb_4bit_use_double_quant=q["double_quant"],
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            m["name"], quantization_config=bnb, torch_dtype=torch.float16, device_map={"": 0}, **kw
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=t["gradient_checkpointing"]
        )
    else:
        raise ValueError(f"Неизвестный метод: {method}")

    if method in ("lora", "qlora"):
        lc = cfg["lora"]
        r = lc["r"]
        peft_cfg = LoraConfig(
            r=r,
            lora_alpha=lc.get("alpha") or 2 * r,
            lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        if t["gradient_checkpointing"]:
            model.enable_input_require_grads()
        model = get_peft_model(model, peft_cfg)

    model.config.use_cache = False
    return model


def count_params(model) -> tuple[int, int]:
    if hasattr(model, "get_nb_trainable_parameters"):  # учитывает упаковку 4-bit
        return model.get_nb_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--force", action="store_true", help="перезапустить, даже если результат уже есть")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    run = make_run_name(cfg)
    res_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / run)
    out_dir = Path(cfg["paths"]["output_dir"]) / run

    if (res_dir / "train_metrics.json").exists() and not args.force:
        print(f"[skip] {run}: результат уже есть (используйте --force)")
        return
    if torch.cuda.device_count() > 1:
        print("ВНИМАНИЕ: видно несколько GPU. Для честных замеров памяти задайте CUDA_VISIBLE_DEVICES=0")

    (res_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    set_seed(cfg["seed"])

    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    train_rows, val_rows, report = build_datasets(cfg)
    d = cfg["data"]
    train_ds, drop_tr = tokenize_rows(train_rows, tok, d["prompt_template"], d["max_seq_len"])
    val_ds, drop_val = tokenize_rows(val_rows, tok, d["prompt_template"], d["max_seq_len"])
    report.update({"dropped_too_long_train": drop_tr, "dropped_too_long_val": drop_val})
    n_tokens = sum(len(x) for x in train_ds["input_ids"])

    model = build_model(cfg)
    trainable, total = count_params(model)
    print(f"[{run}] trainable={trainable:,} total={total:,} ({100 * trainable / total:.2f}%)")

    t = cfg["train"]
    args_tr = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=float(t["lr"]),
        lr_scheduler_type=t["scheduler"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        optim=t["optim"],
        fp16=(t["precision"] == "fp16"),
        bf16=(t["precision"] == "bf16"),
        gradient_checkpointing=t["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=t["logging_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_strategy="steps",
        save_steps=t["eval_steps"],
        save_total_limit=t["save_total_limit"],
        report_to=t.get("report_to", "none"),
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    res_cb = ResourceCallback()
    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8),
        callbacks=[res_cb],
    )

    last = get_last_checkpoint(str(out_dir)) if out_dir.exists() else None
    result = trainer.train(resume_from_checkpoint=last)
    final_eval = trainer.evaluate()
    trainer.save_model(str(out_dir))  # полная модель или адаптер LoRA
    tok.save_pretrained(out_dir)

    runtime = result.metrics["train_runtime"]
    metrics = {
        "run": run,
        "method": cfg["method"],
        "resumed_from_checkpoint": last is not None,  # при True время и память неполные
        "trainable_params": trainable,
        "total_params": total,
        "train_tokens_per_epoch": n_tokens,
        "tokens_per_s": n_tokens * t["epochs"] / runtime,
        "train_runtime_s": runtime,
        "final_train_loss": result.metrics.get("train_loss"),
        "final_val_loss": final_eval["eval_loss"],
        **res_cb.stats,
        "data": report,
    }
    (res_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (res_dir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
