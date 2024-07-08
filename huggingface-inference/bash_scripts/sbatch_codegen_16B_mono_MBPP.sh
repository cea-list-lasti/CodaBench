#!/usr/bin/env bash
#SBATCH--job-name=run
#SBATCH--partition=gpu80G
## SBATCH--reservation=root_44
#SBATCH--nodes=1
#SBATCH--gres=gpu:2
#SBATCH--ntasks-per-node=1
#SBATCH--cpus-per-task=8


# PROMPT_PATH="/home/data/dataset/BLOOM/dev/jerome/humaneval/human-eval/data/HumanEval.jsonl"
PROMPT_PATH="/home/data/dataset/BLOOM/dev/jerome/humaneval/human-eval/data/Mbpp_test_split.jsonl"
TOKENIZER_PATH="/home/data/dataset/BLOOM/CodeGen25/codegen-16B-mono"
MODEL_PATH="/home/data/dataset/BLOOM/CodeGen25/codegen-16B-mono"
DO_SAMPLE=True
TEMPERATURE=0.2
MAX_NEW_TOKENS=384
NUM_GENERATIONS=1
BATCH_SIZE=8
BENCHMARK="MBPP"

echo "GPUs used"
echo $CUDA_VISIBLE_DEVICES

cd ..
CMD="python inference.py \
        --tokenizer-path $TOKENIZER_PATH \
        --model-path $MODEL_PATH \
        --num-generations $NUM_GENERATIONS \
        --temperature $TEMPERATURE \
        --max-new-tokens $MAX_NEW_TOKENS \
        --batch-size $BATCH_SIZE \
        --benchmark $BENCHMARK \
        --do-sample $DO_SAMPLE \
        --prompt-file $PROMPT_PATH"
        
echo "$CMD"
eval "$CMD"



 
