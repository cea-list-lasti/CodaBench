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
#SBATCH --partition=gpuh100p
# # #SBATCH --partition=gpu80G
# #SBATCH --partition=gpup100
# #SBATCH --partition=gpup5000short
# #SBATCH --partition=gpuv100
# #SBATCH --partition=lasti
# #SBATCH --partition=prismgpup
# #SBATCH --partition=gpu-test
# #SBATCH --reservation=root_52

#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=4

#SBATCH --time=2-00:00:00
# #SBATCH --time=0-01:00:00

# StarCoder allocates 60GB/model
#SBATCH --mem=300G

echo "Begin on machine: `hostname`"

#set -o nounset
set -o errexit
set -o pipefail

MODEL_ID="starcoder2-15b-instruct"
MODEL="bigcode/starcoder2-15b-instruct-v0.1"
TASK="humaneval"

source huggingface_env.sh

sleep 6000
# https://github.com/bigcode-project/starcoder2-self-align/tree/main/evaluation
# CUDA_VISIBLE_DEVICES=0 python -m evaluation.text2code \
#   --model_key $MODEL_KEY \
#   --model_name_or_path $MODEL \
#   --save_path $SAVE_PATH \
#   --dataset $DATASET \
#   --temperature 0.0 \
#   --top_p 1.0 \
#   --max_new_tokens 512 \
#   --n_problems_per_batch 16 \
#   --n_samples_per_problem 1 \
#   --n_batches 1

#     prefix: Optional[str] = field(
#         default="",
#         metadata={
#             "help": "Prefix to add to the prompt. For example InCoder needs prefix='<| file ext=.py |>\n'"
#         },
#     )
#     do_sample: Optional[bool] = field(
#         default=True,
#         metadata={"help": "Sample from the language model's output distribution."},
#     )
#     temperature: Optional[float] = field(
#         default=0.2, metadata={"help": "Sampling temperature used for generation."}
#     )
#     top_k: Optional[int] = field(
#         default=0, metadata={"help": "Top-k parameter used for generation."}
#     )
#     top_p: Optional[float] = field(
#         default=0.95, metadata={"help": "Top-p parameter used for nucleus sampling."}
#     )
#     n_samples: Optional[int] = field(
#         default=1,
#         metadata={"help": "Number of completions to generate for each sample."},
#     )
#     eos: Optional[str] = field(
#         default="<|endoftext|>", metadata={"help": "end of sentence token."}
#     )
#     seed: Optional[int] = field(
#         default=0, metadata={"help": "Random seed used for evaluation."}
#     )


source config.sh
NUM_GPU=3

source functions.sh

run_generate ${TASK}
