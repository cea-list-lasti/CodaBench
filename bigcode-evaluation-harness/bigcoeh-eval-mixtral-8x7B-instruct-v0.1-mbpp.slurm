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
#SBATCH --partition=cpufat

#SBATCH --cpus-per-task=64

#SBATCH --time=2-00:00:00
# #SBATCH --time=0-01:00:00

# StarCoder allocates 60GB/model
#SBATCH --mem=500G

echo "Begin on machine: `hostname`"

#set -o nounset
set -o errexit
set -o pipefail

MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_ID="mixtral8x7b_instruct"
TASK="mbpp"

source huggingface_env.sh
source config.sh

SINGULARITY_IMAGE="/home/data/dataset/CodaBench/bigcode-evaluation-harness-3.sif"

TEMPERATURE=0.0
N_SAMPLES=20

source functions.sh

run_eval ${TASK}
