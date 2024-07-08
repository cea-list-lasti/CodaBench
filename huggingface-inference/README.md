Inference script for huggingface models


# Usage

Edit environment variables before launching the script :

- PROMPT_PATH path to dataset (ex : HumanEval.jsonl)
- TOKENIZER_PATH path to tokenizer "/home/data/dataset/BLOOM/CodeGen25/codegen2-7B"
- MODEL_PATH path to model "/home/data/dataset/BLOOM/CodeGen25/codegen2-7B"
- DO_SAMPLE  enable sampling mode
- TEMPERATURE temperature to set if DO_SAMPLE is True
- MAX_NEW_TOKEN number of generated tokens
- NUM_GENERATIONS number of generations for each sample
- BATCH_SIZE number of prompts passed to the model at a time.

# Generations

By default, generation is written to ```output/{model_name}/{run_id}.jsonl``` with run_id a random generated id.
 