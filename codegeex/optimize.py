import argparse
import json
import logging
import multiprocessing
import optuna
import os
import pty
import random
import select
import subprocess
import sys
import uuid

from math import log10
from no_buffering import OutStream
from pathlib import Path
from typing import List

optuna.logging.set_verbosity(optuna.logging.DEBUG)
optuna.logging.disable_default_handler()
_logger = optuna.logging.get_logger("optuna")
_logger.setLevel(logging.DEBUG)
_logger.addHandler(logging.StreamHandler(sys.stdout))


def inference(inference_command: List[str]):
    try:
        _logger.debug(f"inference starting command {inference_command}")
        completed_process = subprocess.run(inference_command,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           check=True, text=True)
        stdout_data = completed_process.stdout
        _logger.debug(stdout_data)
    except subprocess.CalledProcessError as e:
        _logger.error(f"Error executing the command. code: {e.returncode}, "
                      f"out: {e.output}, err: {e.stderr}")
        raise
    # out_r, out_w = pty.openpty()
    # err_r, err_w = pty.openpty()
    # infer_proc = subprocess.Popen(
    #     inference_command,
    #     stdout=out_w,
    #     stderr=err_w,
    #     universal_newlines=True)
    # os.close(out_w) # if we do not write to process, close these.
    # os.close(err_w)
    # fds = {OutStream(out_r), OutStream(err_r)}
    # while fds:
    #     # Call select(), anticipating interruption by signals.
    #     while True:
    #         try:
    #             rlist, _, _ = select.select(fds, [], [])
    #             break
    #         except InterruptedError as e:
    #             raise e
    #             # continue
    #         except ValueError as e:
    #             _logger.error(f"{model_uuid}: Exception in training output collect: {e}\n"
    #                           f"{model_uuid}: Command was: {os.pathsep}{inference_command}")
    #             return 0
    #     for f in rlist:
    #         lines, readable = f.read_lines()
    #         if f.fileno() == out_r:
    #             for line in lines:
    #                 _logger.debug(line)
    #         elif f.fileno() == err_r:
    #             for line in lines:
    #                 _logger.error(line)
    #         if not readable:
    #             _logger.debug(f"{model_uuid}: remove a file descriptor")
    #             # This OutStream is finished.
    #             fds.remove(f)
    # _logger.debug(f"{model_uuid}: waiting for infer_proc to finish")
    #
    # infer_proc.wait()
    #
    # if infer_proc.returncode != 0:
    #     _logger.error(f"{model_uuid}: Inference subprocess failed with return code: {infer_proc.returncode}")
    #     _logger.error(f"{model_uuid}: Command was: {os.pathsep}{' '.join(inference_command)}")
    #     _logger.error(f"{model_uuid}: Stderr: \n{infer_proc.stderr}")
    #     return 0


def objective(trial, nv: bool,
              cuda_visible_devices: List[int],
              sif_image: str,
              eval_sif_image: str,
              num_generations: int,
              main_dir: str,
              scratch_dir: str,
              model_dir: str,
              human_eval_dir: str,
              prompt_path: str,
              micro_batch_size: int,
              temperature: float,
              n_trials: int,
              model_uuid: str,
              skip_generate: bool):
    _logger.debug(f"{model_uuid}: objective start trial {trial.number} at {temperature} on GPU {cuda_visible_devices}")
    if not model_uuid:
        model_uuid = str(uuid.uuid4())
    params = {}
    params["temperature"] = trial.suggest_float("temperature", 0.0, 1.0)
    temperature = params["temperature"]
    # params["hidden-dim"] = trial.suggest_int("hiddenDim", 256, 512)
    # params["hidden-dim"] = trial.suggest_categorical("hiddenDim", [32, 64, 128, 256, 512])
    # params["batch-size"] = trial.suggest_int("batchSize", 2, 6)
    # params["seq-len"] = trial.suggest_categorical("seqLen", [32, 64, 128, 256, 512])
    # params["maxepoch"] = trial.suggest_int("max_epochs", 50, 500)
    # params["max-epochs-without-improvement"] = trial.suggest_int(
    #     "max_epochs_without_improvement", 5, 15)
    # params["input-dropout"] = trial.suggest_float("input_dropout", 0.01, 0.99)
    # params["learning-rate"] = trial.suggest_float("lr", 1e-4, 1e-2)
    # params["weight-decay"] = trial.suggest_float("wd", 1e-6, 1e-4)

    _logger.debug(f"{model_uuid}: params are {params}")
    trial.set_user_attr("uuid", model_uuid)

    if not skip_generate:
        # Inference
        inference_commands = []
        for gpu_id in cuda_visible_devices:
            inference_command = ["singularity", "exec"]
            if nv:
                inference_command.append('--nv')

            os.environ["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            seed = random.randint(1, 1000000000)

            inference_command.append('--writable-tmpfs')

            checkpoint_path = f"{model_dir}/codegeex_13b.pt"

            model_args = ""
            inference_command.extend(
                ["--bind", f"{main_dir}:{main_dir}",
                "--bind", f"{scratch_dir}:{scratch_dir}",
                "--bind", f"{human_eval_dir}:{human_eval_dir}",
                sif_image, "bash", "-c", f'cd {main_dir} &&  \
                        CUDA_VISIBLE_DEVICES={gpu_id} python -u \
                        {main_dir}/CodeGeeX-bench/inference.py \
                        --benchmark humaneval \
                        --prompt-file {prompt_path} \
                        --tokenizer-path {main_dir}/CodeGeeX/codegeex/tokenizer \
                        --micro-batch-size {micro_batch_size} \
                        --out-seq-length 256 \
                        --temperature {temperature} \
                        --top-p 0.95 \
                        --num-generations {int(num_generations/len(cuda_visible_devices)/micro_batch_size)} \
                        --seed {seed}  \
                        --run-id {model_uuid} \
                        --num-layers 39 \
                        --hidden-size 5120 \
                        --num-attention-heads 40 \
                        --max-position-embeddings 2048 \
                        --attention-softmax-in-fp32 \
                        --load {checkpoint_path} \
                        --layernorm-epsilon 1e-5 \
                        --fp16 \
                        --ws-encoding-start-id 10 \
                        --ws-encoding-length 10 \
                        --make-vocab-size-divisible-by 52224 \
                        --seq-length 2048'])
            inference_commands.append(inference_command)
            _logger.debug(f"{model_uuid}: Inference command is:\n{' '.join(inference_command)}")

        # Create a multiprocessing pool to execute the function on GPUs
        with multiprocessing.Pool() as pool:
            pool.map(inference, inference_commands)

    # TODO retrieve the values of the 3 variables below from the correct place
    # Set the path where is your results file
    OUTPUT_PATH="/home/data/dataset/CodaBench/CodeGeeX/output"
    # And the name of the outputfile
    OUTPUT_FILE=f"output_{model_uuid}.jsonl"
    SLURM_CPUS_PER_TASK=32

    test_command = [
        "singularity", "run", "--nv", "--writable-tmpfs",
        "--bind", f"{main_dir}:{main_dir}",
        "--bind", f"{human_eval_dir}:{human_eval_dir}",
        "--bind", f"{OUTPUT_PATH}:/output",
        eval_sif_image,
        prompt_path,
        f"{OUTPUT_PATH}/{OUTPUT_FILE}",
        "--n-workers", str(SLURM_CPUS_PER_TASK)
        ]

    # Execute the subprocess and capture its stdout
    try:
        _logger.debug(f"Starting evaluation command: {test_command}")
        completed_process = subprocess.run(test_command,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           check=True, text=True)
        stdout_data = completed_process.stdout
    except subprocess.CalledProcessError as e:
        _logger.error(f"Error executing the command: code: {e.returncode}")
        _logger.error(f"out: {e.output}, err: {e.stderr}")
        return 0.0
    _logger.debug(f"Got eval data: {stdout_data}")
    eval_file_name = f"{OUTPUT_PATH}/{OUTPUT_FILE}_eval.json"
    try:
        with open(eval_file_name, "r") as eval_file:
            res = json.load(eval_file)
    except OSError as e:
        _logger.error(f"Failed to open evaluation result file {eval_file_name}: {e}")
        return 0.0
    pass_at_1 = res["pass@1"]
    score = pass_at_1
    # score = (pass_at_1+pass_at_10+pass_at_100)/3.0
    _logger.info(f"{model_uuid}: objective trial {trial.number} on GPU {cuda_visible_devices} "
                 f"test result: pass@1: {pass_at_1}, "
                 f"score: {score},")
                 # f"speed: {speed} tokens/s"
    trial.set_user_attr("pass_at_1", pass_at_1)
    # trial.set_user_attr("pass_at_10", pass_at_10)
    # trial.set_user_attr("pass_at_100", pass_at_100)
    return score


def study(study_name: str,
          load_if_exists: bool,
          n_trials: int,
          nv: bool,
          gpu_id: int,
          sif_image: str,
          eval_sif_image: str,
          main_dir: str,
          scratch_dir: str,
          model_dir: str,
          human_eval_dir: str ,
          prompt_path: str,
          micro_batch_size: str,
          temperature: str,
          num_generations: int,
          model_uuid: str,
          skip_generate: bool):
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(
            f"codegeex-{model_uuid}.journal"),
        )
    _logger.info(f"study {study_name} storing into {storage}, "
          f"load_if_exists: {load_if_exists}, "
          f"trials: {n_trials}, using nvidia: {nv} on GPU {gpu_id}")

    study = optuna.create_study(study_name=study_name,
                                direction="maximize",
                                storage=storage,
                                load_if_exists=load_if_exists,
                                sampler=optuna.samplers.RandomSampler(),
                                pruner=optuna.pruners.MedianPruner())
    # reduce the number of trials to run by the number already stored in the database
    n_trials -=  len(study.trials)
    if n_trials <= 0:
        _logger.info(f"study {study_name}: no trial to run; "
              f"there is already {len(study.trials)} in the database.")
        return

    study.optimize(lambda trial: objective(
        trial,
        nv,
        gpu_id,
        sif_image,
        eval_sif_image,
        num_generations=num_generations,
        main_dir=main_dir,
        scratch_dir=scratch_dir,
        model_dir=model_dir,
        human_eval_dir=human_eval_dir,
        prompt_path=prompt_path,
        micro_batch_size=micro_batch_size,
        temperature=temperature,
        n_trials=n_trials,
        model_uuid=model_uuid,
        skip_generate=skip_generate),
                    callbacks=[
                        optuna.study.MaxTrialsCallback(
                            n_trials,
                            states=(optuna.trial.TrialState.COMPLETE,
                            optuna.trial.TrialState.PRUNED))],
        show_progress_bar=True)
    # pruned_trials = study.get_trials(deepcopy=False,
    #                                  states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False,
                                       states=[optuna.trial.TrialState.COMPLETE])

    _logger.info(f"Study {study_name} statistics: ")
    _logger.info(f"  Number of finished trials: {len(study.trials)}")
    # _logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    _logger.info(f"  Number of complete trials: {len(complete_trials)}")
    trial = study.best_trial
    _logger.info(f"Best trial: {trial.value}\n"
                    "  Params: \n" +
                    "\n".join([f"    {key}: {value}" for key, value in trial.params.items()]) +
                    "  Attributes:\n" +
                    "\n".join([f"    {key}: {value}" for key, value in trial.user_attrs.items()]))
    # TODO save data in best models database


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Script to run a function on specified GPUs")
    parser.add_argument('--trials', type=int, default=100, help="Number of Optuna trials")
    parser.add_argument('--main-dir', required=True, help="Main directory")
    parser.add_argument('--scratch-dir', required=True, help="Scratch directory")
    parser.add_argument('--model-dir', required=True, help="Model directory")
    parser.add_argument('--sif-image', required=True, help="SIF image for generation")
    parser.add_argument('--eval-sif-image', required=True, help="SIF image for evaluation")
    parser.add_argument('--human-eval-dir', required=True, help="Human evaluation directory")
    parser.add_argument('--prompt-path', required=True, help="Prompt path")
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--skip-generate", action='store_true')
    parser.add_argument("--uuid", type=str)

    args = parser.parse_args()

    if args.skip_generate and args.uuid:
        model_uuid = args.uuid
    else:
        model_uuid = ""
    # Get the value of CUDA_VISIBLE_DEVICES environment variable
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')

    if not cuda_visible_devices:
        _logger.error("CUDA_VISIBLE_DEVICES environment variable is not set.")
        cuda_visible_devices = []
    else:
        # Split the visible GPU IDs into a list
        cuda_visible_devices = [int(gpu_id) for gpu_id in cuda_visible_devices.split(',')]



        # "--num-generations",
        # default=1,
        # "--num-layers",
        # default=39,
        # "--pre-prompt",
        # "--seed",
        # default=12,
        # "--output-folder",
        # default="output",
        # "--benchmark",
        # default="humaneval",
        # "--hidden-size",
        # default=5120,
        # "--num-attention-heads",
        # default=40,
        # "--padded-vocab-size",
        # default=52224,
        # "--max-position-embeddings",
        # default=2048,
        # "--temperature",
        # default=1.0,
        # "--greedy",
        # "--top-p",
        # default=0.0,
        # "--top-k",
        # default=0,
        # "--out-seq-length",
        # default=2048,
        # "--prompt-file",
        # "--tokenizer-path",
        # "--load",
        # "--run-id",
        # "--state-dict-path",
        # "--micro-batch-size",
        # default=1,
        # "--quantize",


        study("codegeex",  # study_name=
              True,  # load_if_exists=
              args.trials,  # n_trials=
              True,  # nv=
              cuda_visible_devices,  # gpu_id=
              args.sif_image,  # sif_image=
              args.eval_sif_image,  # sif_image=
              args.main_dir,  # main_dir=
              args.scratch_dir,  # scratch_dir=
              args.model_dir,  # model_dir=
              args.human_eval_dir ,  # human_eval_dir=
              args.prompt_path,  # prompt_path=
              args.micro_batch_size,  # micro_batch_size=
              args.temperature,  # temperature=
              args.num_generations,  # num_generations=
              model_uuid,
              args.skip_generate)  # model_uuid=

        _logger.debug("All subprocesses have completed.")

if __name__ == "__main__":
    main()
