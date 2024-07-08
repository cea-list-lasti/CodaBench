#!/bin/bash

#SBATCH --job-name=bigcoeh
#SBATCH --mail-type=start,end,fail
#SBATCH --mail-user=gael.de-chalendar@cea.fr

#SBATCH --nodes=1
#SBATCH --ntasks=1

# #SBATCH --partition=amd
#SBATCH --partition=classicgpu
# #SBATCH --partition=gpu40G
# #SBATCH --partition=gpu80G
# #SBATCH --partition=gpup100
# #SBATCH --partition=gpup5000short
# #SBATCH --partition=gpuv100
# #SBATCH --partition=lasti
# #SBATCH --partition=prismgpup
# #SBATCH --partition=gpu-test
# #SBATCH --reservation=root_52

#SBATCH --gres=gpu:7
#SBATCH --cpus-per-task=5

#SBATCH --time=2-00:00:00
# #SBATCH --time=0-01:00:00

# StarCoder allocates 60GB/model
#SBATCH --mem=100G

echo "Begin on machine: `hostname`"

#set -o nounset
set -o errexit
set -o pipefail

MODEL_ID="starcoder2-3b"
MODEL="bigcode/starcoder2-3b"
TASK="humaneval"

source huggingface_env.sh

source config.sh
TEMPERATURE=0.2
PRECISION="fp16" # Possible values are "fp32", "fp16" and "bf16"

source functions.sh

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
run_generate ${TASK}
