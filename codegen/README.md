Inference script for codegen7 and 16b

# Usage

Edit environment variables before launching the script :

- PROMPT_PATH : path to dataset
- TOKENIZER_PATH : path to HF tokenizer
- MODEL_PATH : path to model
- NUM_GENERATIONS : number of generations per question

# Generations

Generation is outputed to ```output/{run_id}.jsonl``` with run_id a random generated id.
