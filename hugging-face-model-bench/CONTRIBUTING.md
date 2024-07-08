# Contributing

The project dependencies are managed with [poetry](https://python-poetry.org/). To prepare your development environment, create a Python 
virtual environment with your preferred tool (e.g. [conda](https://docs.conda.io/en/latest/),  [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [pyenv](https://github.com/pyenv/pyenv)). Then install the 
project by invoking the poetry install command:

```shell script
$ conda create -n newsdigester python=3.9 # Python version must be >= 3.9
$ conda activate newsdigester
$ cd <project-directory>
$ poetry install
```

Poetry ensures that every developer in the project will have the same Python environment. It also resolves dependency 
conflicts automatically.

If you want to add a library, use the add function of poetry. If the library is needed only for the development phase, 
add the `--group dev` flag. More information can be found in the poetry documentation.
```shell script
$ conda activate newsdigester
$ cd <project-directory>
$ poetry add <library> [--group dev]
```

## Code formatting

Code consistent formatting is ensured by using [ruff](https://github.com/astral-sh/ruff). Please configure your workflow to ensure that every file 
that you commit to the repository has been processed with ruff. 

## Code quality

Code quality is ensured by using [ruff](https://github.com/astral-sh/ruff). Please configure your workflow to ensure that every file that you 
commit to the repository has been processed with ruff. 

## Pre-commit hooks

For those who forget to pass their code through ruff, we provide a [pre-commit](https://pre-commit.com/) configuration 
(`.pre-commit-config.yaml`). In short, pre-commit will pass the staged code that you try to commit in `ruff`. 
It will yell at you if you didn't do it before this point. You will have to process the files with ruff before 
trying to commit them again. This workflow ensures that every file committed to the repository is correctly 
formatted and respect project quality levels. Feel free to use it.