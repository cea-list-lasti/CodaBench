#!/usr/bin/env bash
#SBATCH --job-name=geexinf
# #SBATCH --partition=classicgpu
# #SBATCH --partition=gpu40G
# #SBATCH --partition=gpu80G
#SBATCH--partition=prismgpup
#SBATCH --reservation=root_44
#SBATCH --nodes=1
#SBATCH --nodelist=node45
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --time=7-00:00:00

# Set the mem 100G * number of gpu to avoid processes killed by OOM
# Edit the variables below to adapt to your configuration

set -o errexit
set -o pipefail
set -o nounset

# NB_GPUS must have the same value x as the gres=gpu:x
# and mem=x00G SBATCH parameters
NB_GPUS=8
MICRO_BATCH_SIZE=25
MAIN_DIR="/home/data/dataset/CodaBench/CodeGeeX"
SIF_IMAGE="${MAIN_DIR}/codegeex_latest.sif"
HUMAN_EVAL_DIR="/home/data/dataset/CodaBench/human-eval"
PROMPT_PATH="${HUMAN_EVAL_DIR}/data/HumanEval.jsonl"

TEMPERATURE=0.2
NUM_GENERATIONS=$((200/${NB_GPUS}/${MICRO_BATCH_SIZE}))

export CUDA_HOME=/usr/local/cuda-11.1/
# export CUDA_VISIBLE_DEVICES=0

echo "NUM_GENERATIONS ${NUM_GENERATIONS}"
echo "CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"

SCRATCH_DIR=/sscratch/CodaBench
install -d ${SCRATCH_DIR}

echo "Syncing"
rsync ${SIF_IMAGE} ${SCRATCH_DIR}
SIF_IMAGE=${SCRATCH_DIR}/codegeex_latest.sif

MODEL_DIR=$MAIN_DIR/Models
install -d ${SCRATCH_DIR}/Models
rsync -a ${MODEL_DIR} ${SCRATCH_DIR}
MODEL_DIR=${SCRATCH_DIR}/Models
echo "Syncing DONE"

# import model configuration
source "$MAIN_DIR/CodeGeeX-bench/codegeex_13b.sh"

RUN_ID=$(python -c "import uuid; print(str(uuid.uuid4()))")
echo "RUN_ID ${RUN_ID}"
# Start the timer
start_time=$(date +%s.%N)

for ((i=0; i<${NB_GPUS}; i++))
do
        echo "RUN on GPU ${i}"
        SEED=$((1 + RANDOM % 1000000))
        echo "SEED ${SEED}"
        echo "TEMPERATURE ${TEMPERATURE}"
        SINGULARITYENV_CUDA_VISIBLE_DEVICES=${i} singularity exec --nv --writable-tmpfs \
                --bind "${MAIN_DIR}":"${MAIN_DIR}" \
                --bind "${SCRATCH_DIR}":"${SCRATCH_DIR}" \
                --bind "${HUMAN_EVAL_DIR}":"${HUMAN_EVAL_DIR}" \
                ${SIF_IMAGE} bash -c "cd ${MAIN_DIR} &&  \
                CUDA_VISIBLE_DEVICES=${i} python -u \
                $MAIN_DIR/CodeGeeX-bench/inference.py \
                --benchmark humaneval \
                --prompt-file $PROMPT_PATH \
                --tokenizer-path $MAIN_DIR/CodeGeeX/codegeex/tokenizer \
                --micro-batch-size ${MICRO_BATCH_SIZE} \
                --out-seq-length 256 \
                --temperature $TEMPERATURE \
                --top-p 0.95 \
                --num-generations $NUM_GENERATIONS \
                --seed $SEED  \
                --run-id ${RUN_ID} \
                $MODEL_ARGS" &
#                         -m cProfile -o prof${i}.log \
done

wait

# End the timer
end_time=$(date +%s.%N)

# Calculate the elapsed time
elapsed_time=$(echo "$end_time - $start_time" | bc)

# Print the elapsed time
echo "Time elapsed: $elapsed_time seconds"

# rm -Rf ${SCRATCH_DIR}
