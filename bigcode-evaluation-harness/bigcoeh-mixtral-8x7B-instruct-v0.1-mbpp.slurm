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
#SBATCH --cpus-per-task=8

#SBATCH --time=2-00:00:00
# #SBATCH --time=0-01:00:00

# StarCoder allocates 60GB/model
#SBATCH --mem=1000G

echo "Begin on machine: `hostname`"

#set -o nounset
set -o errexit
set -o pipefail


MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_ID="mixtral8x7b_instruct"
TASK="mbpp"

source huggingface_env.sh

# MBPP (Austin et al., 2021)
# we use temperature sampling (with temperature 0.5) to generate 80 samples of
# code

source config.sh
TEMPERATURE=0.5
N_SAMPLES=80
BATCH_SIZE=20

source functions.sh

#     --load_in_4bit \
#     --max_memory_per_gpu auto \
#     --save_every_k_tasks 20 \
run_generate ${TASK}
