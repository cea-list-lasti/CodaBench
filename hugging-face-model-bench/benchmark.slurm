#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=prismgpup,gpu80G
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=lasti
#SBATCH --time=3-00:00:00
#SBATCH --job-name=CodeLlama-7b-hf
#SBATCH --output=/home/users/jtourille/slurm/CodeLlama-7b-hf.o.log
#SBATCH --error=/home/users/jtourille/slurm/CodeLlama-7b-hf.e.log

. /home/users/$USER/.bashrc
conda activate hfmodelbench
export PYTHONUNBUFFERED=1


MODEL_DIR=/home/data/dataset/huggingface/LLMs/codellama
MODEL_NAME=CodeLlama-7b-hf
SIF_FILE=/home/users/$USER/text-generation-inference-1.0.3.sif
PORT=8089

HUMANEVAL_TEST_FILE=humaneval_python.jsonl
OUTPUT_FILE=${MODEL_NAME}-output-python.jsonl

# Start container
singularity instance run --nv \
  --mount type=bind,src=${MODEL_DIR},dst=/data \
  "${SIF_FILE}" \
  codellama \
  --model-id=/data/${MODEL_NAME} \
  --disable-custom-kernels \
  -p ${PORT} \
  --shard-uds-path=/home/users/"$USER"/codellama-tmp

# Wait for model to load
codellama temporize -h localhost -p ${PORT} -t 300

codellama evaluate-on-humaneval \
  --humaneval-test-file "${HUMANEVAL_TEST_FILE}" \
  -o "${OUTPUT_FILE}" \
  -t 0.8 \
  -n 200 \
  --host localhost \
  --port ${PORT}

singularity instance stop --all