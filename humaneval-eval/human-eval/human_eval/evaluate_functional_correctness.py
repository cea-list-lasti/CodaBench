import fire
import sys
import glob
from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness

def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 6,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    # print("runnning script "+sample_file)
    k = list(map(int, k.split(",")))
    # print("number of files", len(glob.glob(sample_file)))
    for file in glob.glob(sample_file):
        results = evaluate_functional_correctness(file, k, n_workers, timeout, problem_file)
        print("results >",results)


def main():
    fire.Fire(entry_point)


sys.exit(main())
