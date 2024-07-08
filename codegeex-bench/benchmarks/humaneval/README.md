# HumanEval

## Launch as a python script ( not recommended )

```
python human_eval/evaluate_functional_correctness.py mydata_path.jsonl --problem_file=data/HumanEval.jsonl
```

Generations path can be a glob

Use ```combine.py``` to combine multiple jsonl

## Launch as docker image

```
cd to/folder
docker build -t humaneval .
docker run -e DATA_PATH=data/generations/codegeex/T0.2_preprompt_python/combined_data.jsonl humaneval
```

## Launch as Singularity

### Step 1
Docker save
Needs to be done locally if docker is not avaible on cluster
```
docker save humaneval -o humaneval.tar
scp tar to factory...
```

### Step 2 

Build a singularity container from the saved docker
```
singularity build humaneval.sif docker-archive://humaneval.tar
```

### Step 3
Run singularity container.
```
singularity run --env DATA_PATH=output_361_T0.6_5ab33ccc-4203-48b3-b8c4-189a76461b94.jsonl humaneval.sif
```


## Todo 
separate preprocessing from humaneval ( stop words, imports )
