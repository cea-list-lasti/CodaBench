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

#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=8

#SBATCH --time=1-00:00:00
# #SBATCH --time=0-01:00:00

# StarCoder allocates 60GB/model
#SBATCH --mem=200G

echo "Begin on machine: `hostname`"

#set -o nounset
set -o errexit
set -o pipefail


MODEL_ID="starcoder2-3b"
MODEL="bigcode/starcoder2-3b"
TASK="mbpp"

source huggingface_env.sh

source config.sh

# MBPP (Austin et al., 2021)
# we use temperature sampling (with temperature 0.5) to generate 80
# samples of code
TEMPERATURE=0.5
N_SAMPLES=80
BATCH_SIZE=20
MAX_LENGTH_GENERATION=2048

source functions.sh

run_generate ${TASK}
