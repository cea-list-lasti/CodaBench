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
#SBATCH --partition=prismgpup
# #SBATCH --partition=gpu-test
# #SBATCH --reservation=root_52

#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=8

#SBATCH --time=2-00:00:00

# StarCoder allocates 60GB/model
#SBATCH --mem=200G

echo "Begin on machine: `hostname`"

#set -o nounset
set -o errexit
set -o pipefail


MODEL_ID="deepseek-coder-6.7b-instruct"
MODEL="deepseek-ai/deepseek-coder-6.7b-instruct"
TASK="mbpp"

source huggingface_env.sh

source config.sh
# LOAD_IN_NBIT="--load_in_8bit" # Possible values are: "", "--load_in_4bit" and "--load_in_8bit"
PRECISION="fp16" # Possible values are "", "fp32", "fp16" and "bf16"
NUM_GPU=5

# MBPP (Austin et al., 2021)
# we use temperature sampling (with temperature 0.5) to generate 80
# samples of code
TEMPERATURE=0.5
N_SAMPLES=80
BATCH_SIZE=20
MAX_LENGTH_GENERATION=2048

source functions.sh

run_generate ${TASK}
