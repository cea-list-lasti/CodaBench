#!/usr/bin/env bash
#SBATCH--job-name=run
## SBATCH--partition=gpu80G
#SBATCH--reservation=root_44
#SBATCH--nodes=1
#SBATCH--gres=gpu:8

PROMPT_PATH="/home/data/dataset/BLOOM/dev/jerome/humaneval/human-eval/data/Mbpp_test_split.jsonl"
TOKENIZER_PATH="/home/data/dataset/huggingface/LLMs/codellama/CodeLlama-34b-hf"
MODEL_PATH="/home/data/dataset/huggingface/LLMs/codellama/CodeLlama-34b-hf"
DO_SAMPLE=True
TEMPERATURE=0.8
MAX_NEW_TOKENS=384
NUM_GENERATIONS=20
BATCH_SIZE=16    
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
        --benchmark $BENCHMARK \
        --batch-size $BATCH_SIZE \
        --do-sample $DO_SAMPLE \
        --prompt-file $PROMPT_PATH"
        
echo "$CMD"
eval "$CMD"




