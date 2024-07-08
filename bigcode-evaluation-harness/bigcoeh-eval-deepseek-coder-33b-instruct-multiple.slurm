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

MODEL="deepseek-ai/deepseek-coder-33b-instruct"
MODEL_ID="deepseek-coder-33b-instruct"

source huggingface_env.sh

source config.sh

SINGULARITY_IMAGE="/home/data/dataset/CodaBench/bigcode-evaluation-harness-multiple.sif"

source functions.sh

# FAIL "multiple-d" "multiple-jl" "multiple-py"
# DONE "multiple-cpp" "multiple-js" "multiple-java" "multiple-cs" "multiple-go" "multiple-lua" "multiple-php" "multiple-pl" "multiple-r" "multiple-rb" "multiple-rkt"
# TODO "multiple-sh" "multiple-ts" "multiple-scala" "multiple-swift"
TASKS=( "multiple-rs")

for TASK in "${TASKS[@]}"; do
    run_eval ${TASK}
done
