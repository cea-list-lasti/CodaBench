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

#SBATCH --cpus-per-task=256

#SBATCH --time=2-00:00:00
# #SBATCH --time=0-01:00:00

# StarCoder allocates 60GB/model
#SBATCH --mem=1200G

echo "Begin on machine: `hostname`"

#set -o nounset
set -o errexit
set -o pipefail

MODEL_ID="starcoder"
MODEL="bigcode/starcoder"
TASK="mbpp"

source config.sh
source huggingface_env.sh

SINGULARITY_IMAGE="/home/data/dataset/CodaBench/bigcode-evaluation-harness-3.sif"

TEMPERATURE=0.0
N_SAMPLES=20

# MBPP (Austin et al., 2021)
# We use greedy decoding (temperature set to 0.0) to generate a single
# approximate most likely generation,
source functions.sh

#     --bind /home/data/dataset/CodaBench/BigCodeEvaluationHarness/bigcode-evaluation-harness/bigcode_eval/base.py:/app/lm_eval/base.py \
#     --bind /home/data/dataset/CodaBench/BigCodeEvaluationHarness/bigcode-evaluation-harness/bigcode_eval/base.py:/app/build/lib/lm_eval/base.py \
run_eval ${TASK}
