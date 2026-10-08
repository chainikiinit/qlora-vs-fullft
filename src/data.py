"""Загрузка обучающих данных, фильтр утечек относительно GSM8K test, токенизация."""
from __future__ import annotations

import random
import re

from datasets import Dataset, load_dataset


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _ngrams(s: str, n: int = 10) -> set[str]:
    w = _norm(s).split()
    return {" ".join(w[i : i + n]) for i in range(max(len(w) - n + 1, 1))}


def _load_gsm8k_train(strip_calc: bool) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    for r in ds:
        sol = r["answer"]
        if strip_calc:
            sol = re.sub(r"<<[^>]*>>", "", sol)
        rows.append({"question": r["question"], "solution": sol})
    return rows


def _load_metamathqa() -> list[dict]:
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    rows = []
    for r in ds:
        # приводим финальную строку к формату GSM8K: "#### ответ"
        sol = re.sub(r"The answer is:\s*(.+)\s*$", r"#### \1", r["response"].strip())
        rows.append({"question": r["query"], "solution": sol})
    return rows


def load_train_rows(d: dict) -> list[dict]:
    name = d["train_dataset"]
    if name == "gsm8k":
        return _load_gsm8k_train(d.get("strip_calc_annotations", False))
    if name == "metamathqa":
        return _load_metamathqa()
    raise ValueError(f"Неизвестный датасет: {name}")


def build_datasets(cfg: dict):
    """Возвращает (train_rows, val_rows, report).

    Разбиение и подвыборка зависят только от data.data_seed, поэтому одинаковы
    для всех методов и всех seed обучения; подвыборки вложены (1k ⊂ 3k ⊂ ...).
    """
    d = cfg["data"]
    rows = load_train_rows(d)

    test = load_dataset("openai/gsm8k", "main", split="test")
    test_grams: set[str] = set()
    for q in test["question"]:
        test_grams |= _ngrams(q)
    before = len(rows)
    rows = [r for r in rows if not (_ngrams(r["question"]) & test_grams)]
    removed = before - len(rows)

    rng = random.Random(d["data_seed"])
    rng.shuffle(rows)
    val = rows[: d["val_size"]]
    pool = rows[d["val_size"] :]
    train = pool[: d["train_size"]] if d.get("train_size") else pool

    report = {
        "train_dataset": d["train_dataset"],
        "rows_before_leak_filter": before,
        "removed_by_leak_filter_10gram": removed,
        "n_train": len(train),
        "n_val": len(val),
    }
    return train, val, report


def tokenize_rows(rows: list[dict], tok, template: str, max_len: int):
    """Токенизация с маскированием промпта (loss только по решению)."""
    out, dropped = [], 0
    for r in rows:
        p_ids = tok(template.format(question=r["question"]), add_special_tokens=False)["input_ids"]
        s_ids = tok(" " + r["solution"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = p_ids + s_ids
        if len(ids) > max_len:
            dropped += 1
            continue
        out.append(
            {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": [-100] * len(p_ids) + s_ids,
            }
        )
    return Dataset.from_list(out), dropped
