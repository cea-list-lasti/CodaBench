#!/bin/bash

#SBATCH --job-name=bigcoehe
#SBATCH --mail-type=start,end,fail
#SBATCH --mail-user=gael.de-chalendar@cea.fr

# Nombre de machine ou NODES typiquement=1 sauf
#SBATCH --nodes=1

# Nombre de processus en general=1 (a mémoire distribues type miprun)
#SBATCH --ntasks=1

# #SBATCH --partition=allcpu
# #SBATCH --partition=cpu
# #SBATCH --partition=cpufat
#SBATCH --partition=gpuv100

#SBATCH --cpus-per-task=48

#SBATCH --time=0-01:00:00

# StarCoder allocates 60GB/model
#SBATCH --mem=100G

echo "Begin on machine: `hostname`"

#set -o nounset
set -o errexit
set -o pipefail

MODEL_ID="dolphincoder-starcoder2-15b"
MODEL="cognitivecomputations/dolphincoder-starcoder2-15b"

TASK="humaneval"

source huggingface_env.sh
export HF_HOME=/home/data/dataset/huggingface
export HF_HUB_CACHE=${HF_HOME}/cache/
export HF_DATASETS_CACHE=${HF_HUB_CACHE}/datasets
export HF_DATASETS_OFFLINE=1
# TRANSFORMERS_CACHE is deprecated but if it is absent, data are searched
# on internet instead of using the cache
export TRANSFORMERS_CACHE=$HF_HUB_CACHE

source config.sh

SINGULARITY_IMAGE="/home/data/dataset/CodaBench/bigcode-evaluation-harness-3.sif"

source functions.sh

run_eval ${TASK}
