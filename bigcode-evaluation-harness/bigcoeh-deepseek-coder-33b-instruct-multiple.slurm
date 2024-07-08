#!/bin/bash

#SBATCH --job-name=bigcoeh
#SBATCH --mail-type=start,end,fail
#SBATCH --mail-user=gael.de-chalendar@cea.fr

# Nombre de machine ou NODES typiquement=1 sauf
#SBATCH --nodes=1

# Nombre de processus en general=1 (a mémoire distribues type miprun)
#SBATCH --ntasks=1

# #SBATCH --partition=amd
# #SBATCH --partition=classicgpu
# #SBATCH --partition=gpu40G
# #SBATCH --partition=gpu80G
# #SBATCH --partition=gpup100
# #SBATCH --partition=gpup5000short
# #SBATCH --partition=gpuv100
# #SBATCH --partition=lasti
# #SBATCH --partition=prismgpup
# #SBATCH --partition=gpu-test
#SBATCH --reservation=root_52

#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=6

#SBATCH --time=3-00:00:00
# #SBATCH --time=0-01:00:00

# StarCoder allocates 60GB/model
#SBATCH --mem=1000G

echo "Begin on machine: `hostname`"

#set -o nounset
set -o errexit
set -o pipefail

MODEL="deepseek-ai/deepseek-coder-33b-instruct"
MODEL_ID="deepseek-coder-33b-instruct"

DEST="/home/data/dataset/CodaBench/BigCodeEvaluationHarness/generations"
source huggingface_env.sh

TEMPERATURE=0.2
N_SAMPLES=200
BATCH_SIZE=5

source functions.sh

# Done: "multiple-cpp" "multiple-cs" "multiple-java" "multiple-js" "multiple-go" "multiple-lua" "multiple-php" "multiple-pl" "multiple-r" "multiple-rb" "multiple-rkt" "multiple-rs"
# Failure: "multiple-py" "multiple-d" "multiple-jl" (RuntimeError: probability tensor contains either `inf`, `nan` or element < 0)
TASKS=( "multiple-sh" "multiple-ts" "multiple-scala" "multiple-swift")

for TASK in "${TASKS[@]}"; do
    run_generate "${TASK}"
done
