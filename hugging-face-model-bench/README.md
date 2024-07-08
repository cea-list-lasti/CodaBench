# Hugging Face Model Bench

## 1. Requirements

Create a Python 3.9+ environment and install the dependencies via [Poetry](https://python-poetry.org/).

```shell
$ conda create -n newsdigester python=3.9
$ conda activate newsdigester
$ poetry install
```

Pull the last version of the [Text Generation Inference](https://github.com/huggingface/text-generation-inference) 
image from Docker Hub and convert it to the singularity format. You can install [Apptainer](https://apptainer.org/) 
by following the [documentation](https://apptainer.org/docs/admin/main/installation.html#install-unprivileged-from-pre-built-binaries).
Install it also in your user directory on FactoryIA (it will be used later to start the container).

```shell
$ docker pull ghcr.io/huggingface/text-generation-inference:1.3.4
$ apptainer build text-generation-inference-1.3.4.sif docker-daemon://ghcr.io/huggingface/text-generation-inference:1.3.4
```

## 2. Evaluate a model

Download the model with `huggingface-cli`.

```shell
$ huggingface-cli login
# Enter your access token (you can generate one on Hugging Face Hub)
$ huggingface-cli download \
  codellama/CodeLlama-7b-hf \ # Name of the model (as found on Hugging Face Hub) 
  --local-dir ./codellama/CodeLlama-7b-hf \ # Path where the model will be written 
  --local-dir-use-symlinks False # Skip caching (see documentation for further information)
```