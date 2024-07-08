#!/bin/bash

#SBATCH --job-name=bcehesm
#SBATCH --mail-type=start,end,fail
#SBATCH --mail-user=gael.de-chalendar@cea.fr

# Nombre de machine ou NODES typiquement=1 sauf
#SBATCH --nodes=1

# Nombre de processus en general=1 (a mémoire distribues type miprun)
#SBATCH --ntasks=1

#SBATCH --partition=gpuv100
# #SBATCH --partition=allcpu
# #SBATCH --partition=cpu
# #SBATCH --partition=cpufat

#SBATCH --cpus-per-task=10

#SBATCH --time=7-00:00:00

# StarCoder allocates 60GB/model
#SBATCH --mem=100G

echo "Begin on machine: `hostname`"

#set -o nounset
set -o errexit
set -o pipefail

MODEL_ID="starcoder2-7b"
MODEL="bigcode/starcoder2-7b"

source config.sh
source huggingface_env.sh

SINGULARITY_IMAGE="/home/data/dataset/CodaBench/bigcode-evaluation-harness-multiple.sif"

source functions.sh

# FAIL
# DONE "multiple-py" "multiple-cpp" "multiple-js" "multiple-java" "multiple-cs" "multiple-go" "multiple-lua" "multiple-php" "multiple-pl" "multiple-r" "multiple-rb" "multiple-rkt" "multiple-rs" "multiple-sh" "multiple-ts" "multiple-scala" "multiple-d"
# TODO
TASKS=("multiple-jl" "multiple-swift")


for TASK in "${TASKS[@]}"; do
    run_eval "$TASK"
done
