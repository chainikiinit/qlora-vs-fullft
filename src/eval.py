"""Оценка через lm-evaluation-harness (автоматическое извлечение ответов, без LLM-судьи).

Обученная модель:  python -m src.eval --config configs/base.yaml configs/method/qlora.yaml --set seed=1
Исходная модель:   python -m src.eval --baseline --config configs/base.yaml --eval_config configs/eval_baseline.yaml

QLoRA оценивается "как обучалась": 4-bit база + адаптер (без слияния).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .config import ensure_dir, load_config, make_run_name


def build_model_args(method: str, base: str, out_dir: Path, dtype: str) -> str:
    if method == "baseline":
        parts = [f"pretrained={base}"]
    elif method == "full":
        parts = [f"pretrained={out_dir}"]
    elif method == "lora":
        parts = [f"pretrained={base}", f"peft={out_dir}"]
    elif method == "qlora":
        parts = [
            f"pretrained={base}",
            f"peft={out_dir}",
            "load_in_4bit=True",
            "bnb_4bit_quant_type=nf4",
            "bnb_4bit_use_double_quant=True",
            "bnb_4bit_compute_dtype=float16",
        ]
    else:
        raise ValueError(method)
    return ",".join(parts + [f"dtype={dtype}"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--eval_config", default="configs/eval.yaml")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import lm_eval
    from lm_eval.api.registry import get_model

    cfg = load_config(args.config, args.set)
    ev = yaml.safe_load(open(args.eval_config, encoding="utf-8"))
    base = cfg["model"]["name"]

    if args.baseline:
        method, run = "baseline", "baseline_" + base.split("/")[-1]
        cfg["method"] = "baseline"
    else:
        method, run = cfg["method"], make_run_name(cfg)

    res_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / run)
    out_dir = Path(cfg["paths"]["output_dir"]) / run
    if (res_dir / "eval.json").exists() and not args.force:
        print(f"[skip] {run}: eval.json уже есть")
        return
    if method == "baseline":
        (res_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))

    model_args = build_model_args(method, base, out_dir, ev["dtype"])
    print(f"[{run}] model_args: {model_args}")
    lm = get_model("hf").create_from_arg_string(model_args, {"batch_size": ev["batch_size"]})

    all_res = {}
    for t in ev["tasks"]:
        res = lm_eval.simple_evaluate(
            model=lm,
            tasks=[t["name"]],
            num_fewshot=t["num_fewshot"],
            limit=t.get("limit"),
            log_samples=False,
        )
        all_res[f"{t['name']}|{t['num_fewshot']}shot"] = res["results"]
        print(t["name"], t["num_fewshot"], res["results"].get(t["name"]))

    (res_dir / "eval.json").write_text(json.dumps(all_res, indent=2, default=str))


if __name__ == "__main__":
    main()
