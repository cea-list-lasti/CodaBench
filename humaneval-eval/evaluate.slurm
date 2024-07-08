#!/usr/bin/env bash
#SBATCH --job-name=humaneval
#SBATCH --partition=cpufat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=164
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00

# Edit the variables below to adapt to your configuration

set -o errexit
set -o pipefail
set -o nounset

# Set the path where is your results file
OUTPUT_PATH="/home/data/dataset/CodaBench/CodeGeeX/output"

# And the name of the outputfile
OUTPUT_FILE="output_5c00e9da-f90e-4fc6-980c-9d3ace7e00e4.jsonl"


MAIN_DIR="/home/data/dataset/CodaBench"
SIF_IMAGE="${MAIN_DIR}/humaneval-eval.sif"
HUMAN_EVAL_DIR="/home/data/dataset/CodaBench/human-eval"
PROMPT_PATH="${HUMAN_EVAL_DIR}/data/HumanEval.jsonl"


# Start the timer
start_time=$(date +%s.%N)

singularity run \
        --nv --writable-tmpfs \
        --bind "${MAIN_DIR}":"${MAIN_DIR}" \
        --bind "${HUMAN_EVAL_DIR}":"${HUMAN_EVAL_DIR}" \
        --bind "${OUTPUT_PATH}":/output \
        ${SIF_IMAGE} \
        ${PROMPT_PATH} \
        ${OUTPUT_PATH}/${OUTPUT_FILE} \
        --n-workers ${SLURM_CPUS_PER_TASK}
# End the timer
end_time=$(date +%s.%N)

# Calculate the elapsed time
elapsed_time=$(echo "$end_time - $start_time" | bc)

# Print the elapsed time
echo "Time elapsed: $elapsed_time seconds"
