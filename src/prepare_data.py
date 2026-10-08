"""Отчёт по данным: размеры, утечки, длины в токенах -> results/data_report.json.

python -m src.prepare_data --config configs/base.yaml [--set data.train_dataset=metamathqa]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from transformers import AutoTokenizer

from .config import ensure_dir, load_config
from .data import build_datasets, tokenize_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    train, val, report = build_datasets(cfg)
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    ds, dropped = tokenize_rows(train, tok, cfg["data"]["prompt_template"], 10**9)
    lens = np.array([len(x) for x in ds["input_ids"]])
    report.update(
        {
            "tokens_mean": float(lens.mean()),
            "tokens_p50": float(np.percentile(lens, 50)),
            "tokens_p95": float(np.percentile(lens, 95)),
            "tokens_max": int(lens.max()),
            "share_over_max_seq_len": float((lens > cfg["data"]["max_seq_len"]).mean()),
        }
    )
    out = ensure_dir(cfg["paths"]["results_dir"]) / f"data_report_{cfg['data']['train_dataset']}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
