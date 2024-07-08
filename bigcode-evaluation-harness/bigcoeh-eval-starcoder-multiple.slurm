#!/bin/bash

#SBATCH --job-name=bcehesm
#SBATCH --mail-type=start,end,fail
#SBATCH --mail-user=gael.de-chalendar@cea.fr

# Nombre de machine ou NODES typiquement=1 sauf
#SBATCH --nodes=1

# Nombre de processus en general=1 (a mémoire distribues type miprun)
#SBATCH --ntasks=1

# #SBATCH --partition=allcpu
# #SBATCH --partition=cpu
#SBATCH --partition=cpufat

#SBATCH --cpus-per-task=256

#SBATCH --time=1-00:00:00
# #SBATCH --time=0-01:00:00

# StarCoder allocates 60GB/model
#SBATCH --mem=1200G

echo "Begin on machine: `hostname`"

#set -o nounset
set -o errexit
set -o pipefail

MODEL_ID="starcoder"
MODEL="bigcode/starcoder"

source config.sh
source huggingface_env.sh

SINGULARITY_IMAGE="/home/data/dataset/CodaBench/bigcode-evaluation-harness-multiple.sif"

TEMPERATURE=0.2
N_SAMPLES=200

source functions.sh

# "multiple-jl"
# "multiple-cpp" "multiple-cs" "multiple-d" "multiple-go" "multiple-java"
# "multiple-js" "multiple-lua" "multiple-php" "multiple-pl" "multiple-py"
# "multiple-r" "multiple-rb" "multiple-rkt" "multiple-rs"
TASKS=("multiple-scala" "multiple-sh" "multiple-swift" "multiple-ts")

for TASK in "${TASKS[@]}"; do
    run_eval "$TASK"
done
