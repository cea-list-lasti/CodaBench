#!/usr/bin/env bash
#SBATCH --job-name=geexinf
# #SBATCH --partition=classicgpu
# #SBATCH --partition=gpu40G
# #SBATCH --partition=gpu80G
#SBATCH--partition=prismgpup
#SBATCH --reservation=root_44
#SBATCH --nodes=1
#SBATCH --nodelist=node45
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
# #SBATCH --mem-per-gpu=100G
#SBATCH --time=1-00:00:00

# Set the mem 100G * number of gpu to avoid processes killed by OOM

set -o errexit
set -o pipefail
set -o nounset

echo "Begin on machine: $(hostname)"

#### Edit the variables below to adapt to your configuration
#### Adapt also the Slurm variables above, particularly gres to select a
#### number of GPUs and mem as explained

# The number of Optuna trials to run
NUM_TRIALS=4
# Where all CodeGeeX data are
MAIN_DIR="/home/data/dataset/CodaBench/CodeGeeX"
# The CodeGeeX Singularity image
SIF_IMAGE="${MAIN_DIR}/codegeex_latest.sif"
# The HumanEval Singularity image
EVAL_SIF_IMAGE="/home/data/dataset/CodaBench/humaneval-eval.sif"
# Where HumanEval data are
HUMAN_EVAL_DIR="/home/data/dataset/CodaBench/human-eval"
# The HumanEval prompts file
PROMPT_PATH="${HUMAN_EVAL_DIR}/data/HumanEval.jsonl"

# Set MICRO_BATCH_SIZE depending on the GPU memory available. For CodeGeeX with
# 80GB, a value of 25 is correct
MICRO_BATCH_SIZE=25

# Set the temperature to reproduce paper values. You can also activate the trial
# parameter temperature in optimize.py to optimize it
TEMPERATURE=0.2

# This is the number of code generation to do for each example. Its value is
# used to compute pass@k values. It is usualy 200 in papers
NUM_GENERATIONS=200

#### END of variables to adapt

# export CUDA_VISIBLE_DEVICES=0

echo "NUM_GENERATIONS ${NUM_GENERATIONS}"
echo "CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"

SCRATCH_DIR=/sscratch/CodaBench
install -d ${SCRATCH_DIR}

echo "Syncing"
rsync ${SIF_IMAGE} ${SCRATCH_DIR}
SIF_IMAGE=${SCRATCH_DIR}/codegeex_latest.sif

rsync ${EVAL_SIF_IMAGE} ${SCRATCH_DIR}
EVAL_SIF_IMAGE=${SCRATCH_DIR}/humaneval-eval.sif

MODEL_DIR=$MAIN_DIR/Models
install -d ${SCRATCH_DIR}/Models
rsync -a ${MODEL_DIR} ${SCRATCH_DIR}
MODEL_DIR=${SCRATCH_DIR}/Models
echo "Syncing DONE"

# Start the timer
start_time=$(date +%s.%N)

python optimize.py --trials 4 \
        --main-dir ${MAIN_DIR} --model-dir ${MODEL_DIR} \
        --scratch-dir ${SCRATCH_DIR} \
        --sif-image ${SIF_IMAGE} --eval-sif-image ${EVAL_SIF_IMAGE}\
        --human-eval-dir ${HUMAN_EVAL_DIR} \
        --prompt-path ${PROMPT_PATH} \
        --temperature ${TEMPERATURE} --num-generations ${NUM_GENERATIONS} \
        --micro-batch-size ${MICRO_BATCH_SIZE} # \
#         --skip-generate --uuid "c6ceb184-f09d-4a18-9e7e-754b882f004d"


# End the timer
end_time=$(date +%s.%N)

# Calculate the elapsed time
elapsed_time=$(python -c "print(int($end_time - $start_time))")
# Print the elapsed time
echo "Time elapsed: $elapsed_time seconds"

# rm -Rf ${SCRATCH_DIR}
