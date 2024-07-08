# CodeGeeX

If you want to just run the inference of the CodeGeeX model, follow the instructions in the `inference` section. If you want to run both inference and evaluation and eventually optimize parameters, read the `Optuna based optimization` section.

## Inference

This project is used to generate code with CodeGeeX before using the result for benchmarking using HumanEval.

You must set some variables in `inference.slurm`:

  * `SIF_IMAGE` the path to the  CodeGeeX singularity image
  * `MAIN_DIR` the path to the dir containing code, models and results
  * `HUMAN_EVAL_DIR` the path to the HumanEval benchmark data

The file `codegeex_13b.sh` contains the parameters related to the model. In particular, you must set the `CHECKPOINT_PATH` which points to the selected model checkpoint file.

Adapt the `inference.slurm` file to your local settings and then generate the code with (if you use a slurm based cluster):
```bash
$ sbatch inference.slurm
```

If you are instead on a local computer, run:
```bash
$ bash inference.slurm
```

After completion, the result will be a jsonl file located in `$MAIN_DIR/ouput` .

## Optuna based optimization

Run `optimize.slurm` on a cluster (after editing it to adapt to your
configuration) to use Optuna to try various parameters and several trials on
each for inference and evaluation
