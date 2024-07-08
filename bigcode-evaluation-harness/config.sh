# WARNING: Do not modify this file. It is shared with other scripts.
#          If necessary, change variables values after sourcing it


DEST="/home/data/dataset/CodaBench/BigCodeEvaluationHarness/generations"
TEMPERATURE=0.2
N_SAMPLES=200
BATCH_SIZE=10
MAX_LENGTH_GENERATION=512
LOAD_IN_NBIT="" # Possible values are: "", "--load_in_4bit" and "--load_in_8bit"
PRECISION="fp32" # Possible values are "", "fp32", "fp16" and "bf16"
NUM_GPU=8
TOP_P=0.95
