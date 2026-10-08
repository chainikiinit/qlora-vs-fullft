# QLoRA против полного fine-tuning на математических задачах

Курсовая по машинному обучению, ИТМО, осень 2026.

## 1. Постановка задачи

TODO (2–3 предложения): что сравниваем, зачем, что считается успехом.

Сравнивается полное дообучение малой LLM с LoRA в 16 бит и QLoRA (4-bit NF4) на математических задачах (GSM8K) в условиях одной GPU Kaggle. Оцениваются:
качество решения (точное совпадение итогового числа), сохранение общих навыков (ARC, HellaSwag, MMLU) и затраты ресурсов (пиковая память, скорость).
Проверка ответов автоматическая, без LLM-судьи.

Гипотезы и их статус: [docs/hypotheses.md](docs/hypotheses.md).

## 2. Данные

См. [docs/data_card.md](docs/data_card.md): источники, лицензии, способ получения, размеры, проблемы, проверка утечек.

## 3. Структура репозитория

```
configs/
  base.yaml            общие настройки (модель, данные, гиперпараметры обучения)
  method/              full.yaml, lora.yaml, qlora.yaml (переопределяют base.yaml)
  eval.yaml            бенчмарки для дообученных моделей
  eval_baseline.yaml   то же + 5-shot GSM8K для исходной модели
src/
  config.py            слияние YAML и переопределения --set key=value
  data.py              загрузка, фильтр утечек, токенизация с маской промпта
  prepare_data.py      отчёт по данным
  train.py             full / LoRA / QLoRA, замер памяти и скорости
  eval.py              lm-evaluation-harness
  aggregate.py         сводные таблицы mean ± std по seed
scripts/run.sh         стадии экспериментов: baseline | lr | core | size | rank | meta
notebooks/             01_eda, 02_baseline, 03_results (с пояснениями между ячейками)
results/               конфиги и метрики прогонов (в git)
outputs/               веса и адаптеры (не в git)
docs/                  гипотезы, карточка данных
```

## 4. Установка и запуск

```bash
pip install -r requirements.txt
python -m src.prepare_data --config configs/base.yaml      # проверка данных и утечек
bash scripts/run.sh baseline                               # исходная модель
bash scripts/run.sh lr                                     # подбор lr по каждому методу
# впишите выбранные lr в configs/method/*.yaml
bash scripts/run.sh core                                   # 3 метода x 3 seed
bash scripts/run.sh size && bash scripts/run.sh rank && bash scripts/run.sh meta
python -m src.aggregate                                    # results/summary*.csv
```

Один прогон вручную:

```bash
python -m src.train --config configs/base.yaml configs/method/qlora.yaml --set seed=1 lora.r=32
python -m src.eval  --config configs/base.yaml configs/method/qlora.yaml --set seed=1 lora.r=32
```

### Kaggle
- Включите GPU и интернет в настройках ноутбука; для замеров нужна одна GPU (`CUDA_VISIBLE_DEVICES=0` уже в `scripts/run.sh`).
- Пути: `paths.output_dir=/kaggle/working/outputs`, результаты копируйте в репозиторий (`results/` маленькие).
- Сессия ограничена по времени: прогоны уже посчитанные пропускаются, обучение продолжается с последнего чекпоинта. При этом время и память такого прогона неполные (флаг `resumed_from_checkpoint` в метриках).
- Лимит рабочей директории ~20 ГБ: удаляйте `outputs/` полного FT после оценки.
- Если Qwen2.5 в fp16 даёт NaN loss, смените модель (например, `model.name=HuggingFaceTB/SmolLM2-360M`) или проверьте fp32 для проблемных слоёв.

### Ноутбуки
Запускайте по порядку из `notebooks/` или из корня репозитория (путь определяется автоматически).
- `01_eda.ipynb`: происхождение и состав данных, длины, дубликаты, проверка разбиения и утечек (нужен интернет для загрузки GSM8K).
- `02_baseline.ipynb`: правила без модели и исходная модель; нужен `results/baseline_*/eval.json` (`bash scripts/run.sh baseline`).
- `03_results.ipynb`: таблицы и графики по `results/*/`; ничего не пересчитывает, GPU не нужна. Графики сохраняются в `results/figures/`, таблица в `results/main_table.md`.

## 5. Методика

- Модель: Qwen2.5-0.5B (см. `configs/base.yaml`). Метод и скорость обучения подбираются отдельно, оптимизатор и gradient checkpointing одинаковы для всех методов.
- Разбиение train/val фиксируется `data.data_seed` и не зависит от seed обучения; подвыборки вложены. Тест (GSM8K test) в обучении и подборе не участвует.
- Из обучающих данных удаляются задачи с общими 10-граммами с GSM8K test.
- Выбор lr по `final_val_loss` на val: это приближение, а не точность на val.
- Основные конфигурации считаются с 3 seed, в таблицах mean ± std.

## 6. Результаты

TODO: таблицы и графики «качество vs память / время», сравнение с baseline, проверенные гипотезы.

## 7. Выводы и ограничения

TODO: что получилось, где не работает, что делать дальше. Обязательно отметить: возможное загрязнение бенчмарков у выбранной модели, один домен, одна модель размера 0.5B, доверительные интервалы GSM8K.

## 8. Источники

- Dettmers et al., QLoRA: Efficient Finetuning of Quantized LLMs, 2023 (arXiv:2305.14314)
- Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, 2021 (arXiv:2106.09685)
- Biderman et al., LoRA Learns Less and Forgets Less, 2024 (arXiv:2405.09673)
- Cobbe et al., Training Verifiers to Solve Math Word Problems (GSM8K), 2021 (arXiv:2110.14168)
- Yu et al., MetaMath: Bootstrap Your Own Mathematical Questions for LLMs, 2023 (arXiv:2309.12284)
- Документация: Hugging Face PEFT, bitsandbytes, lm-evaluation-harness

(номера и выходные данные перед защитой сверить по оригиналам)
