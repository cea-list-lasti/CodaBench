import argparse
import codegeex
import json
import numpy
import os
import os.path
import random
import sys
import time
import torch
import uuid

from filelock import FileLock
from typing import Dict

from codegeex.torch import CodeGeeXModel
from codegeex.tokenizer import CodeGeeXTokenizer
from codegeex.quantization import quantize
from codegeex.benchmark.utils import cleanup_code

def model_provider(args):
    """Build the model."""
    model = CodeGeeXModel(
        args.hidden_size,
        args.num_layers,
        args.num_attention_heads,
        args.padded_vocab_size,
        args.max_position_embeddings
    )
    return model


def add_code_generation_args(parser):
    group = parser.add_argument_group(title="code generation")
    group.add_argument(
        "--num-generations",
        type=int,
        default=1,
    )
    group.add_argument(
        "--num-layers",
        type=int,
        default=39,
    )
    group.add_argument(
        "--pre-prompt",
        type=str,
        default="",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=12,
    )
    group.add_argument(
        "--output-folder",
        type=str,
        default="output",
        # target="output_folder"
    )
    # humaneval or mbpp
    group.add_argument(
        "--benchmark",
        type=str,
        default="humaneval",
        help=("The type of benchmark is used to prepare the prompt sent to the "
              "model.")
    )
    group.add_argument(
        "--hidden-size",
        type=int,
        default=5120,
    )
    group.add_argument(
        "--num-attention-heads",
        type=int,
        default=40,
    )
    group.add_argument(
        "--padded-vocab-size",
        type=int,
        default=52224,
    )
    group.add_argument(
        "--max-position-embeddings",
        type=int,
        default=2048,
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    group.add_argument(
        "--greedy",
        action="store_true",
        default=False,
        help="Use greedy sampling.",
    )
    group.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="Top p sampling.",
    )
    group.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top k sampling.",
    )
    group.add_argument(
        "--out-seq-length",
        type=int,
        default=2048,
        help="Size of the output generated text.",
    )
    group.add_argument(
        "--prompt-file",
        type=str,
        default="./test_prompt.txt",
    )
    group.add_argument(
        "--tokenizer-path",
        type=str,
        default="./tokenizer",
    )
    group.add_argument(
        "--load",
        type=str,
    )
    group.add_argument(
        "--run-id",
        type=str,
        default=str(uuid.uuid4())
    )
    group.add_argument(
        "--state-dict-path",
        type=str,
    )
    group.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
    )
    group.add_argument(
        "--quantize",
        action="store_true",
    )
    return parser


def save_dict_to_jsonl(file_path: str, data: Dict[str, str]) -> None:
    """ Append the data in json format to the given jsonl file

    Args:
        file_path (str): path to the file to append to
        data (Dict[str, str]): the dictionary to append to the file
    """
    file_path_lock = f"{file_path}.lock"

    lock = FileLock(file_path_lock)
    with lock:
        # Initialize an empty list to store the dictionaries
        json_data = []

        # Open the JSONL file and read line by line
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    # Load each line as a JSON object and append it to the list
                    try:
                        f_data = json.loads(line)
                        json_data.append(f_data)
                    except json.JSONDecodeError:
                        # Handle invalid JSON data in the file, if necessary
                        print(f"Skipping invalid JSON data: {line}")
        for f_data in json_data:
            if f_data["task_id"] == data["task_id"]:
                f_data["completion"] += data["completion"]
                break
        else:
            json_data.append(data)
        with open(file_path, 'w', encoding='UTF-8') as file:
            for f_data in json_data:
                json.dump(f_data, file)
                file.write('\n')


def read_jsonl(file_path):
    """ read a jsonl file
    Args:
        file_path (str): jsonl file path

    Returns:
        [dict]: data contained in json as list of dicts
    """
    data = []
    with open(file_path, 'r', encoding='UTF-8') as file:
        lines = file.read().splitlines()
        for line in lines:
            item = json.loads(line.strip())
            data.append(item)
    return data


def generate():
    """ main method 

    Raises:
        ValueError: _description_
    """
    sys.stdout.flush()
    num_of_gpus = torch.cuda.device_count()
    print(num_of_gpus)
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sys.stdout.flush()
    parser = argparse.ArgumentParser()
    parser = add_code_generation_args(parser)
    args, _ = parser.parse_known_args()
    print(f"run_id {args.run_id} on gpu {available_gpus}")

    def _set_random_seed(seed_):
        """Set random seed for reproducability."""
        if seed_ is not None and seed_ > 0:
            # Ensure that different pipeline MP stages get different seeds.
            # seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
            random.seed(seed_)
            numpy.random.seed(seed_)
            torch.manual_seed(seed_)
            # if torch.cuda.device_count() > 0:
            #     mpu.model_parallel_cuda_manual_seed(seed)
        else:
            raise ValueError("Seed ({}) should be a positive integer.".format(seed_))

    # setting random seed if given as paremeter
    if args.seed:
        _set_random_seed(args.seed)
    # f'_{args.seed}'
    # f'_T{args.temperature}'
    file_path = (f'{args.output_folder}/output'
                 f'_{args.run_id}.jsonl')
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    print(f"Will append results to {file_path}")
    sys.stdout.flush()

    # loading tokenizer and model
    tokenizer = CodeGeeXTokenizer(
        tokenizer_path=args.tokenizer_path,
        mode="codegeex-13b")
    state_dict = torch.load(args.load, map_location="cpu")
    state_dict = state_dict["module"]
    model = model_provider(args)
    model.load_state_dict(state_dict)
    model.eval()
    model.half()

    if args.quantize:
        model = quantize(model, weight_bit_width=8, backend="torch")

    model.cuda()
    torch.cuda.synchronize()

    print("loaded model")
    sys.stdout.flush()

    prompts_data = read_jsonl(args.prompt_file)
    outputs = []

    preprompt=""
    benchmark = args.benchmark
    for _, prompt_data in enumerate(prompts_data):
        language = "python"
        if benchmark == "humaneval":
            preprompt = "# language: Python\n"
            task_id = prompt_data['task_id']
            prompt = preprompt+prompt_data['prompt']+'\n'
        elif benchmark == "humanevalx":
            task_id = prompt_data['task_id']
            if "CPP" in task_id:
                preprompt = "// language: C++\n"
                language = "cpp"
            elif "Python" in task_id:
                preprompt= "# language: Python\n"
                language = "python"
            elif "Go" in task_id:
                preprompt= "// language: Go\n"
                language = "go"
            elif "Java" in task_id:
                preprompt= "// language: Java\n"
                language = "javascript"
            elif "JavaScript" in task_id:
                preprompt= "// language: JavaScript\n"
                language = "js"
            else:
                raise Exception("unsupported language for humanevalx")
            prompt = preprompt+prompt_data['prompt']+'\n'
            
        elif benchmark == "mbpp":
            test_str = "\n".join(prompt_data['test_list'])
            preprompt = "# language: Python\n\n"
            mbpp_prompt = f"You are an expert Python programmer, and here is your task: {prompt_data['text']}. Your code should pass these tests: {test_str} \n\n"
            print("mbpp prompt", mbpp_prompt)
            task_id = prompt_data['task_id']
            prompt = mbpp_prompt
        elif benchmark == "mbpp_one_shot":
            # one shot example ( ex 2)
            preprompt = "# language: Python\n\n"
            mbpp_prompt = preprompt+"Write a function to find the similar elements from the given two tuple lists. \n"
            tests = ["assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)","assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)","assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"]
            test_str = "\n".join(tests)
            mbpp_prompt+= f"Your code should pass these tests: {test_str} \n\n"        
            mbpp_prompt += "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res)\n\n"
            mbpp_prompt += prompt_data['text']+"\n"
            test_str = "\n".join(prompt_data['test_list'])
            mbpp_prompt+= f"Your code should pass these tests: {test_str} \n\n"
            task_id = prompt_data['task_id']
            prompt = mbpp_prompt
            print("====================================================================")
            print("prompt "+prompt, flush=True)
            
            
        elif benchmark== "mbpp_three_shot":
            # adding task id 2,3,4 in prompt
            preprompt = "# language: Python\n\n"
            mbpp_prompt = preprompt+"Write a function to find the similar elements from the given two tuple lists. \n"
            tests = ["assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)","assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)","assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"]
            test_str = "\n".join(tests)
            mbpp_prompt+= f"Your code should pass these tests: {test_str} \n\n"        
            mbpp_prompt += "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res)\n\n"
          
            mbpp_prompt +=  "Write a python function to identify non-prime numbers.\n"
            tests = ["assert is_not_prime(2) == False", "assert is_not_prime(10) == True", "assert is_not_prime(35) == True"]
            test_str = "\n".join(tests)
            mbpp_prompt+= f"Your code should pass these tests: {test_str} \n\n"        
            mbpp_prompt+= "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result\n\n" 
            
            mbpp_prompt +=  "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.\n"
            tests = ["assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ", "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ", "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]"]
            test_str = "\n".join(tests)
            mbpp_prompt+= f"Your code should pass these tests: {test_str} \n\n"        
            mbpp_prompt+= "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums\n\n"
            
            mbpp_prompt += prompt_data['text']+"\n"
            test_str = "\n".join(prompt_data['test_list'])
            mbpp_prompt+= f"Your code should pass these tests: {test_str} \n\n"
            task_id = prompt_data['task_id']
            prompt = mbpp_prompt
            
            print("====================================================================")
            print("prompt "+prompt, flush=True)
            
        else:
            raise NotImplementedError
        
        if not prompt:
            continue
        if prompt == "stop":
            return
        try:
            gen = []
            for i in range(args.num_generations):
                t0 = time.perf_counter()
                # generated_code: List[str]
                generated_code = codegeex.generate(
                    model,
                    tokenizer,
                    prompt,
                    out_seq_length=args.out_seq_length,
                    seq_length=args.max_position_embeddings,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    micro_batch_size=args.micro_batch_size,
                    backend="megatron",
                    verbose=True,
                )
                generated_code = [cleanup_code(code,
                                              language_type=language,
                                              dataset=benchmark) for code in generated_code]
                gen += generated_code
                sys.stdout.flush()

            output = dict(task_id=f"{task_id}", completion=gen)
            save_dict_to_jsonl(file_path, output)
            # outputs.append(output)
            t1 = time.perf_counter()
        except (ValueError, FileNotFoundError) as e:
            continue
    os.remove(f"{file_path}.lock")


def main():
    """ main method

    Raises:
        ValueError: _description_
    """
    return generate()


if __name__ == "__main__":
    main()
