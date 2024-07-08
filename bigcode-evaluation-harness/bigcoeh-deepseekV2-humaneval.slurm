#!/bin/bash

#SBATCH --job-name=bigcoeh
#SBATCH --mail-type=start,end,fail
#SBATCH --mail-user=pauline.auda@cea.fr

#SBATCH --nodes=1
#SBATCH --ntasks=8

#SBATCH --reservation root_64

#SBATCH -w node43

#SBATCH --partition=gpup5000short,gpu40G,gpup100,gpu80G,gpuv100,gpup6000,gpup5000short,prismgpup

#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8

#SBATCH --time=2-00:00:00
# #SBATCH --time=0-01:00:00

#SBATCH --exclusive

#SBATCH --mem=1024G

echo "Begin on machine: `hostname`"

#set -o nounset
set -o errexit
set -o pipefail

MODEL_ID="DeepSeek-V2"
MODEL="bigcode/DeepSeek-V2"
TASK="humaneval"

source huggingface_env.sh

source config.sh
TEMPERATURE=0.2
PRECISION="fp16" # Possible values are "fp32", "fp16" and "bf16"

source functions.sh

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
run_generate ${TASK}

