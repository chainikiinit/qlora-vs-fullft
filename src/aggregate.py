"""Сводная таблица: results/*/ -> results/summary.csv и results/summary_by_config.csv (mean ± std по seed)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


def flatten_eval(ev: dict) -> dict:
    flat = {}
    for key, results in ev.items():
        task = key.split("|")[0]
        for metric, val in results.get(task, {}).items():
            if isinstance(val, (int, float)) and "stderr" not in metric:
                flat[f"{key}/{metric}"] = val
    return flat


def main(results_dir: str = "results") -> None:
    rows = []
    for d in sorted(Path(results_dir).iterdir()):
        cfg_p = d / "config.yaml"
        if not cfg_p.exists():
            continue
        cfg = yaml.safe_load(cfg_p.read_text())
        row = {
            "run": d.name,
            "method": cfg.get("method"),
            "dataset": cfg["data"]["train_dataset"],
            "train_size": cfg["data"].get("train_size"),
            "lora_r": (cfg.get("lora") or {}).get("r"),
            "lr": cfg.get("train", {}).get("lr"),
            "seed": cfg.get("seed"),
        }
        if (d / "train_metrics.json").exists():
            tm = json.loads((d / "train_metrics.json").read_text())
            for k in ("trainable_params", "peak_mem_allocated_gb", "wall_time_s", "tokens_per_s", "final_val_loss"):
                row[k] = tm.get(k)
        if (d / "eval.json").exists():
            row.update(flatten_eval(json.loads((d / "eval.json").read_text())))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(Path(results_dir) / "summary.csv", index=False)

    keys = ["method", "dataset", "train_size", "lora_r", "lr"]
    num = df.select_dtypes("number").columns.difference(keys + ["seed"])
    g = df.groupby(keys, dropna=False)[list(num)].agg(["mean", "std", "count"])
    g.to_csv(Path(results_dir) / "summary_by_config.csv")
    print(df.to_string(max_colwidth=40))


if __name__ == "__main__":
    main()
