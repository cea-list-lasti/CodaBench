#!/bin/bash

#SBATCH --job-name=bigcoeh
#SBATCH --mail-type=start,end,fail
#SBATCH --mail-user=gael.de-chalendar@cea.fr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

# #SBATCH --partition=amd
# #SBATCH --partition=classicgpu
#SBATCH --partition=gpu40G
# #SBATCH --partition=gpu80G
# #SBATCH --partition=gpup100
# #SBATCH --partition=gpup5000short
# #SBATCH --partition=gpuv100
# #SBATCH --partition=lasti
# #SBATCH --partition=prismgpup
# #SBATCH --partition=gpu-test
# #SBATCH --reservation=root_52

#SBATCH --gres=gpu:6

#SBATCH --time=3-00:00:00

#SBATCH --mem=150G

echo "Begin on machine: `hostname`"

#set -o nounset
set -o errexit
set -o pipefail

MODEL_ID="starcoder2-3b"
MODEL="bigcode/starcoder2-3b"

source huggingface_env.sh

source config.sh
BATCH_SIZE=5
PRECISION="fp16" # Possible values are "", "fp32", "fp16" and "bf16"
NUM_GPU=6
MAX_LENGTH_GENERATION=1024

source functions.sh

# Done: "multiple-py" "multiple-d"
# Failure:  (RuntimeError: probability tensor contains either `inf`, `nan` or element < 0)
TASKS=("multiple-cpp" "multiple-cs" "multiple-java" "multiple-js" "multiple-sh" "multiple-ts" "multiple-rs" "multiple-go" "multiple-php" "multiple-lua" "multiple-pl" "multiple-r" "multiple-rb" "multiple-rkt" "multiple-scala" "multiple-swift" "multiple-jl")

for TASK in "${TASKS[@]}"; do
    run_generate "$TASK"
done
