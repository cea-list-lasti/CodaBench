#!/usr/bin/env bash
#SBATCH--job-name=run
#SBATCH--partition=gpu40G
#SBATCH--nodes=1
#SBATCH--gres=gpu:1
#SBATCH--ntasks-per-node=1
#SBATCH--cpus-per-task=8

MAIN_DIR="/home/data/dataset/BLOOM/CodeGeeX"
PROMPT_PATH="/home/data/dataset/BLOOM/dev/jerome/humaneval/human-eval/data/HumanEval.jsonl"
PROMPT_FILE=$PROMPT_PATH/tests/test_humaneval.txt

TEMPERATURE=0.6
NUM_GENERATIONS=10
NUM_EXP=3

# import model configuration
source "$MAIN_DIR/configs/codegeex_13b.sh"
export CUDA_HOME=/usr/local/cuda-11.1/
export CUDA_VISIBLE_DEVICES=0

echo "NUM_GENERATIONS ${NUM_GENERATIONS}"

for ((i=1; i<=NUM_EXP; i++)); do
        echo "RUN ${i}"
        SEED=$((1 + RANDOM % 10000))
        echo "SEED ${SEED}"
        for TEMPERATURE in 0.2 0.6; do
                echo "TEMPERATURE ${TEMPERATURE}"
                CMD="python inference.py \
                        --prompt-file $PROMPT_PATH \
                        --tokenizer-path $MAIN_DIR/codegeex/tokenizer \
                        --micro-batch-size 1 \
                        --out-seq-length 1024 \
                        --temperature $TEMPERATURE \
                        --top-p 0.95 \
                        --seed $SEED  \
                        --num-generations $NUM_GENERATIONS \
                        $MODEL_ARGS"

                echo "$CMD"
                eval "$CMD"
        done
done