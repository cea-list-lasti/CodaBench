import datasets
import os
import sys

#os.environ["HF_HOME"] = "/home/gael/Projets/CodaBench/huggingface_datasets"
#os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"]

for bench in ["humaneval", "mbpp"]:
    for lang in ["cpp", "cs", "d", "go", "java", "jl", "js", "lua", "php", "pl",
                 "py", "r", "rb", "rkt", "rs", "scala", "sh", "swift", "ts"]:
        print(f"Loading dataset nuprl/MultiPL-E {bench}-{lang}", 
                file=sys.stderr)
        datasets.load_dataset("nuprl/MultiPL-E", f"{bench}-{lang}",
                              trust_remote_code=True)

