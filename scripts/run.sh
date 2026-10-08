#!/usr/bin/env bash
# Использование: bash scripts/run.sh <baseline|lr|core|size|rank|meta>
# Уже посчитанные прогоны пропускаются (results/<run>/train_metrics.json, eval.json).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # одна GPU: честные замеры памяти
export TOKENIZERS_PARALLELISM=false

BASE=configs/base.yaml
EVAL=configs/eval.yaml

train_eval() {  # train_eval <method> key=value ...
  local method=$1; shift
  python -m src.train --config $BASE configs/method/$method.yaml --set "$@"
  python -m src.eval  --config $BASE configs/method/$method.yaml --eval_config $EVAL --set "$@"
}

case "${1:-}" in
  baseline)  # исходная модель без дообучения
    python -m src.eval --baseline --config $BASE --eval_config configs/eval_baseline.yaml ;;

  lr)  # подбор скорости обучения отдельно для каждого метода, один seed, только обучение;
       # выбор по final_val_loss (результаты в results/*/train_metrics.json)
    declare -A LRS=( [full]="5.0e-6 1.0e-5 2.0e-5" [lora]="1.0e-4 2.0e-4 5.0e-4" [qlora]="1.0e-4 2.0e-4 5.0e-4" )
    for m in full lora qlora; do
      for lr in ${LRS[$m]}; do
        python -m src.train --config $BASE configs/method/$m.yaml --set seed=0 train.lr=$lr
      done
    done ;;

  core)  # ядро: 3 метода x 3 seed, весь GSM8K train, r=16
    for seed in 0 1 2; do
      for m in full lora qlora; do train_eval $m seed=$seed; done
    done ;;

  size)  # влияние размера обучающей выборки (7k = core)
    for n in 1000 3000; do
      for m in full lora qlora; do train_eval $m seed=0 data.train_size=$n; done
    done ;;

  meta)  # большая выборка из MetaMathQA
    for m in full lora qlora; do
      train_eval $m seed=0 data.train_dataset=metamathqa data.train_size=25000 train.epochs=1
    done ;;

  rank)  # влияние ранга LoRA (16 = core)
    for r in 8 32 128; do
      for m in lora qlora; do train_eval $m seed=0 lora.r=$r; done
    done ;;

  *) echo "Использование: bash scripts/run.sh <baseline|lr|core|size|rank|meta>"; exit 1 ;;
esac

python -m src.aggregate
