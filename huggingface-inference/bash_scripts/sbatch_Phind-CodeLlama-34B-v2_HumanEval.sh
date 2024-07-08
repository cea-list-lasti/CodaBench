#!/usr/bin/env bash
#SBATCH--job-name=run
## SBATCH--partition=gpu40G
#SBATCH--reservation=root_44
#SBATCH--nodes=1
#SBATCH--gres=gpu:4
#SBATCH--ntasks-per-node=1
#SBATCH--cpus-per-task=8

PROMPT_PATH="/home/data/dataset/BLOOM/dev/jerome/humaneval/human-eval/data/HumanEval.jsonl"
TOKENIZER_PATH="/home/data/dataset/BLOOM/Phind-CodeLlama-34B-v2"
MODEL_PATH="/home/data/dataset/BLOOM/Phind-CodeLlama-34B-v2"
DO_SAMPLE=True
TEMPERATURE=0.1
MAX_NEW_TOKENS=384
NUM_GENERATIONS=50
BATCH_SIZE=50
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
        --benchmark $BENCHMARK \
        --batch-size $BATCH_SIZE \
        --do-sample $DO_SAMPLE \
        --prompt-file $PROMPT_PATH"
        
echo "$CMD"
eval "$CMD"




