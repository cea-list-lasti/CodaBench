#!/usr/bin/env bash
#SBATCH--job-name=run
## SBATCH--partition=gpu40G
#SBATCH--reservation=root_44
#SBATCH--nodes=1
#SBATCH--gres=gpu:4

PROMPT_PATH="/home/data/dataset/BLOOM/dev/jerome/humaneval/human-eval/data/HumanEval.jsonl"
TOKENIZER_PATH="/home/data/dataset/BLOOM/codellama/source/CodeLlama-34b-hf"
MODEL_PATH="/home/data/dataset/BLOOM/codellama/source/CodeLlama-34b-hf"
DO_SAMPLE=False
TEMPERATURE=0
MAX_NEW_TOKENS=384
NUM_GENERATIONS=200
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




