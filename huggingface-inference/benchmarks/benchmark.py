from utils import read_jsonl, batch
import time
from typings import BenchmarkConfig
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM
from generation.generate import Generation

class Benchmark(ABC):
    @abstractmethod
    def run_benchmark(self, tokenizer : AutoTokenizer, model : AutoModelForCausalLM, generation_strategy : Generation, args : BenchmarkConfig):
        pass
   
class HumanEvalBenchmark(Benchmark):
    def run_benchmark(self, tokenizer : AutoTokenizer, model : AutoModelForCausalLM, generation_strategy : Generation, args : BenchmarkConfig):
        args.max_new_tokens = 384
        args.max_length= None
 
        prompts_data = read_jsonl(args.prompt_file)
        outputs = []

        # batch of code questions
        for i,prompt_batch in enumerate(batch(prompts_data, args.batch_size)):
            batch_gens = []
            print(f"{i}=======================================================", flush=True)
            for j in range(args.num_generations):
                print(f"{j}----------------------------------", flush=True)
                start = time.time()
                completions = generation_strategy.generate([p["prompt"] for p in prompt_batch],
                                                           tokenizer,
                                                           model,
                                                           args)
                batch_gens.append(completions)
                end = time.time()
                print(f"time (s) {end-start}", flush=True)
            for gen in batch_gens:
                for i,_ in enumerate(gen):
                    outputs.append(dict(task_id=f"{prompt_batch[i]['task_id']}", completion=gen[i]))
        
        return outputs



class MBPPBenchmark(Benchmark):
        
    def run_benchmark(self, tokenizer : AutoTokenizer, model : AutoModelForCausalLM, generation_strategy : AutoModelForCausalLM, args : BenchmarkConfig):
        args.max_new_tokens = 500
        args.max_length= None

        prompts_data = read_jsonl(args.prompt_file)

        three_shot_prompts = [
                                {   "text": "Write a function to find the similar elements from the given two tuple lists.",
                                    "test_list": ["assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)", 
                                                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)", 
                                                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"],
                                    "code" : "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res)"
                                },
                                {   "text": "Write a python function to identify non-prime numbers.",
                                    "test_list": ["assert is_not_prime(2) == False",
                                                "assert is_not_prime(10) == True",
                                                "assert is_not_prime(35) == True"],
                                    "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res)"
                                },
                                {
                                    "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
                                    "test_list": ["assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]",
                                                "assert heap_queue_largest( [25,35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]", 
                                                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]"],
                                    "code" : "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums"
                                }   
                            ]

        outputs = []
        
        
        # prepare preprompt ( 3 shot)
        def format_pre_prompt(prompt):
            text = prompt["text"]
            if "code" in prompt:
                code = prompt["code"]
            else:
                code = ""
            test_str = '# '+"\n# ".join(prompt["test_list"])
            return f"""# You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n{test_str}\n# Solution : \n{code}\n\n"""
        mbpp_preprompt = ""
        for prompt in three_shot_prompts:
            mbpp_preprompt+= format_pre_prompt(prompt)

        def format_mbpp_prompt(prompt_data):
            text = mbpp_preprompt
            sample_text = prompt_data['text']
            sample_test_str = '# '+"\n# ".join(prompt_data["test_list"])
            text += f"""# You are an expert Python programmer, and here is your task: {sample_text} Your code should pass these tests:\n\n{sample_test_str}\n# Solution : \n"""
            return text
            
        # batch of code questions
        for i,prompt_batch in enumerate(batch(prompts_data, args.batch_size)):
            batch_gens = []
            # if i<25:
            #     continue
            print(f"{i}=======================================================", flush=True)
            for j in range(args.num_generations):
                print(f"{j}----------------------------------", flush=True)
                start = time.time()
                print("preprompt -----------------------", flush=True)
                print(format_mbpp_prompt(prompt_batch[0]))
                print("preprompt -----------------------", flush=True)
                completions = generation_strategy.generate([format_mbpp_prompt(p) for p in prompt_batch],
                                                           tokenizer,
                                                           model,
                                                           args)
                print("completion -----------------------", flush=True)
                print(completions[0], flush=True)
                print("completion -----------------------", flush=True)

                batch_gens.append(completions)
                end = time.time()
                print(f"time (s) {end-start}", flush=True)
            for gen in batch_gens:
                for i,_ in enumerate(gen):
                    outputs.append(dict(task_id=f"{prompt_batch[i]['task_id']}", completion=gen[i]))
        
        return outputs