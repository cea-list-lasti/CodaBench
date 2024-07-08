import datasets
import os
import sys

#os.environ["HF_HOME"] = "/home/gael/Projets/CodaBench/huggingface_datasets"
#os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"]

print(f"Loading dataset ", 
                file=sys.stderr)
datasets.load_dataset("codeparrot/self-instruct-starcoder", 
                              trust_remote_code=True)

