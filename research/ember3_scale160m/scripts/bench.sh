#!/bin/bash
# Round-9 benchmark battery — runs on CT103 from /data/ember3.
set -u
export HF_HOME=/data/models/ember3-olmoe/hf-cache
export HF_HUB_DISABLE_XET=1
export TMPDIR=/data/models/ember3-olmoe/tmp

cd /data/ember3
TASKS_MAIN=arc_easy,arc_challenge,hellaswag,piqa,winogrande,boolq,lambada_openai

run_one () {
  local NAME="$1" PATHSPEC="$2"
  ./venv/bin/lm_eval --model hf --model_args "pretrained=${PATHSPEC},dtype=bfloat16" \
    --tasks "$TASKS_MAIN" --batch_size 16 \
    --output_path "/data/ember3/results_bench/${NAME}_main" \
    && echo "BATTERY-DONE-${NAME}-main"
  ./venv/bin/lm_eval --model hf --model_args "pretrained=${PATHSPEC},dtype=bfloat16" \
    --tasks triviaqa --limit 2000 --batch_size 16 \
    --output_path "/data/ember3/results_bench/${NAME}_tqa" \
    && echo "BATTERY-DONE-${NAME}-tqa"
}

run_one frozen  allenai/OLMoE-1B-7B-0924
run_one experts /data/models/ember3-olmoe/adapted/experts
run_one full    /data/models/ember3-olmoe/adapted/full
