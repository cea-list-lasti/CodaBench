#!/usr/bin/env bash
#SBATCH--job-name=run
## SBATCH--partition=gpu40G
#SBATCH--reservation=root_44
#SBATCH--nodes=1
#SBATCH--gres=gpu:4
#SBATCH--ntasks-per-node=1
#SBATCH--cpus-per-task=8

PROMPT_PATH="/home/data/dataset/BLOOM/dev/jerome/humaneval/human-eval/data/HumanEval.jsonl"
TOKENIZER_PATH="/home/data/dataset/BLOOM/CodeGen25/codegen2-7B"
MODEL_PATH="/home/data/dataset/BLOOM/CodeGen25/codegen2-7B"
DO_SAMPLE=True
TEMPERATURE=0.6
MAX_NEW_TOKENS=384
NUM_GENERATIONS=1
BATCH_SIZE=8
BENCHMARK="HumanEval"

echo "GPUs used"
echo $CUDA_VISIBLE_DEVICES

cd ..
CMD="python inference.py \
        --tokenizer-path $TOKENIZER_PATH \
        --model-path $MODEL_PATH \
        --num-generations $NUM_GENERATIONS \
        --temperature $TEMPERATURE \
        --max-new-tokens $MAX_NEW_TOKENS \
        --do-sample $DO_SAMPLE \
        --benchmark $BENCHMARK \
        --batch-size $BATCH_SIZE \
        --prompt-file $PROMPT_PATH"
        
echo "$CMD"
eval "$CMD"




