#!/usr/bin/env bash
#!/bin/bash

#SBATCH --job-name=hexe
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

#SBATCH --time=1-00:00:00
# #SBATCH --time=0-01:00:00

# StarCoder allocates 60GB/model
#SBATCH --mem=1200G

SIF_IMAGE=codegeex_latest.sif
LANGUAGE=java # choose between ["python", "go", "js", "cpp", "java"]
TEST_FILE=data/fake_generated_files/humaneval_java.jsonl # Path to your generated test file
TMP_DIR=/tmp
NCPUS=$(nproc) # Set this variable to 32 when evaluating code written in go to avoid the error `Too many open files`

filename_wo_dir=$(basename -- "$TEST_FILE")
filename_wo_extension="${filename_wo_dir%.*}"
dirname=$(dirname ${TEST_FILE})
log_file=${filename_wo_extension}.log

singularity exec --writable-tmpfs \
    --bind "${dirname}":/mnt/data \
    ${SIF_IMAGE} bash -c "cd /workspace/CodeGeeX && \
        scripts/evaluate_humaneval_x.sh /mnt/data/${filename_wo_dir} \
            ${LANGUAGE} ${NCPUS} > /mnt/data/${log_file}"
