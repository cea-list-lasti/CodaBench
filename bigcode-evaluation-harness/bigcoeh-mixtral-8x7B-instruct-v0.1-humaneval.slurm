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

#SBATCH --time=2-00:00:00
# #SBATCH --time=0-01:00:00

# StarCoder allocates 60GB/model
#SBATCH --mem=1000G

echo "Begin on machine: `hostname`"

#set -o nounset
set -o errexit
set -o pipefail

DEST="/home/data/dataset/CodaBench/BigCodeEvaluationHarness/generations"
source huggingface_env.sh

MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_ID="mixtral8x7b_instruct"
TASK="humaneval"

TEMPERATURE=0.2
N_SAMPLES=200
BATCH_SIZE=10

source functions.sh

run_generate ${TASK}
