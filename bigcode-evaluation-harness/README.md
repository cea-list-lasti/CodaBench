# CodaBench Big Code Evaluation Harness

The slurm files here use https://github.com/bigcode-project/bigcode-evaluation-harness to evaluate models on several benchmarks.

It works in a conda environment with Python 3.10.13. The `requirements.txt` file is updated with:
```
transformers>=4.37.1
accelerate>=0.26.1
datasets>=2.16.1
evaluate>=0.4.1
pyext==0.7
mosestokenizer==1.0.0
huggingface_hub>=0.20.3
fsspec==2023.9.2
```

## Singularity container images

On FactoryAI, in `/home/data/dataset/CodaBench`:
* `bigcode-evaluation-harness-3.sif`: the image for tests execution for all tasks except MultiPL-E.
* `bigcode-evaluation-harness-multiple.sif`: the image for MultiPL-E tests execution.


## Running evaluations


Datasets and models installation in FactoryAI from your PC requires to have the same version of the `datasets` package on both machines. Currently (at time of writing):

```bash
pip3 install datasets==2.16.1
```

To make models available on FactoryAI:

```bash
install -d ${HOME}/Projets/CodaBench/huggingface_models
sshfs gdechalendar@132.167.191.35:/home/data/dataset/huggingface/LLMs/ ${HOME}/Projets/CodaBench/huggingface_models
export HF_MODELS_CACHE=${HOME}/Projets/CodaBench/huggingface_models
MODEL=deepseek-ai/deepseek-coder-33b-instruct
huggingface-cli download \
    ${MODEL} \
    --local-dir huggingface_models/${MODEL} \
    --local-dir-use-symlinks False --resume-download
```

To make datasets available on FactoryAI:

```bash
install -d ${HOME}/Projets/CodaBench/huggingface_datasets
sshfs gdechalendar@132.167.191.35:/home/data/dataset/huggingface/cache ${HOME}/Projets/CodaBench/huggingface_datasets
export HF_HUB_CACHE=${HOME}/Projets/CodaBench/huggingface_datasets
export HF_DATASETS_CACHE=$HF_HUB_CACHE
python3 -c 'import datasets; datasets.load_dataset(path="openai_humaneval",name=None)'
```

To run a model on a task, use as example one of the existing slurm files and to execute the tests and compute the results, use the corresponding eval file.

In all cases, set environement variables like that to use our cached models (see the `env.sh` file for up to date values):
```bash
# For the cache itself for datasets
export DATASETS=/home/data/dataset/huggingface/cache/
export TRANSFORMERS_CACHE=$DATASETS
export HF_HUB_CACHE=$DATASETS
export HF_DATASETS_OFFLINE=1
# TRANSFORMERS_CACHE is deprecated but if it is absent, data are searched
# on internet instead of using the cache
TRANSFORMERS_CACHE=$DATASETS

# For models
MODELS=/home/data/dataset/huggingface/LLMs
HF_MODELS_CACHE=$MODELS
```
