#!/bin/bash

#SBATCH --job-name=bigcoeh
#SBATCH --mail-type=start,end,fail
#SBATCH --mail-user=pauline.auda@cea.fr

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --reservation root_64

#SBATCH -w node43

#SBATCH --partition=gpup5000short,gpu40G,gpup100,gpu80G,gpuv100,gpup6000,gpup5000short,prismgpup

#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8


#SBATCH --time=2-00:00:00

#SBATCH --exclusive


echo "Begin on machine: `hostname`"

#set -o nounset
set -o errexit
set -o pipefail

MODEL_ID="merged_mistral"
MODEL="bigcode/merged_mistral"

TASK="humaneval"

source config.sh
source huggingface_env.sh

SINGULARITY_IMAGE="/home/data/dataset/CodaBench/bigcode-evaluation-harness-3.sif"

source functions.sh

run_eval ${TASK}
