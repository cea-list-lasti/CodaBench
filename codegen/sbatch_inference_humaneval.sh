#!/usr/bin/env bash
#SBATCH--job-name=run
#SBATCH--partition=gpu40G
#SBATCH--nodes=1
#SBATCH--gres=gpu:1
#SBATCH--ntasks-per-node=1
#SBATCH--cpus-per-task=8

PROMPT_PATH="/home/data/dataset/BLOOM/dev/jerome/humaneval/human-eval/data/HumanEval.jsonl"
TOKENIZER_PATH="/home/data/dataset/BLOOM/CodeGen25/codegen2-7B"
MODEL_PATH="/home/data/dataset/BLOOM/CodeGen25/codegen2-7B"
NUM_GENERATIONS=1

CMD="python inference_humaneval.py \
        --num-generations $NUM_GENERATIONS \
        --tokenizer-path $TOKENIZER_PATH \
        --model-path $MODEL_PATH \
        --prompt-file $PROMPT_PATH"
        
echo "$CMD"
eval "$CMD"




