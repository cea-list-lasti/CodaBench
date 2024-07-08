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
#SBATCH --partition=gpu40G
# #SBATCH --partition=gpu80G
# #SBATCH --partition=gpup100
# #SBATCH --partition=gpup5000short
# #SBATCH --partition=gpuv100
# #SBATCH --partition=lasti
# #SBATCH --partition=prismgpup
# #SBATCH --partition=gpu-test
# #SBATCH --reservation=root_52

#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4

#SBATCH --time=2-00:00:00
# #SBATCH --time=0-01:00:00

# StarCoder allocates 60GB/model
#SBATCH --mem=300G

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

NUM_GPU=4
LOAD_IN_NBIT="--load_in_8bit"
# PRECISION=""

source functions.sh

run_generate ${TASK}
