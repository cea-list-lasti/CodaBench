#!/bin/bash

run_eval() {
    local TASK=$1
    echo "Evaluating ${MODEL_ID} on task ${TASK}"
    install -d "${HOME}/tmp"

#    srun sprofile start
    singularity exec \
        --pwd /app \
        --bind $DEST:$DEST \
        --bind $MODELS:$MODELS \
        --bind $HF_HOME:$HF_HOME \
        --bind "${HOME}/tmp":/tmp \
        $SINGULARITY_IMAGE \
        python3 main.py \
        --model $MODELS/${MODEL} \
        --tasks ${TASK} \
        --allow_code_execution  \
        --temperature ${TEMPERATURE} \
        --n_samples ${N_SAMPLES} \
        --load_generations_path "$DEST/generations_${MODEL_ID}_${TASK}_${TASK}.json" \
        --metric_output_path "$DEST//generations_${MODEL_ID}_${TASK}_${TASK}_eval.json"
    wait
 #   srun sprofile stop
}


run_generate() {
    local TASK=$1
    echo "Generating for ${MODEL_ID} on task ${TASK}"

    cd /home/data/dataset/CodaBench/BigCodeEvaluationHarness/bigcode-evaluation-harness

  #  srun sprofile start
    accelerate launch --multi_gpu --num_processes=${NUM_GPU} --num_machines=1 --mixed_precision=no --dynamo_backend=no main.py \
        --model ${MODELS}/${MODEL} \
        --tasks ${TASK} \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P} \
        --precision ${PRECISION} ${LOAD_IN_NBIT} \
        --do_sample True \
        --n_samples ${N_SAMPLES} \
        --batch_size ${BATCH_SIZE} \
        --max_length_generation "${MAX_LENGTH_GENERATION}" \
        --generation_only \
        --save_generations \
        --save_generations_path "${DEST}/generations_${MODEL_ID}_${TASK}.json"
    wait
   # srun sprofile stop
}

