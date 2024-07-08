from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import uuid
import time
from utils import str2bool, save_to_jsonl
import os
from generation.generate import HFBaseGeneration, PhindGeneration
from benchmarks.benchmark import HumanEvalBenchmark, MBPPBenchmark

def argparser():
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--tokenizer-path', type=str, help='path to HF tokenizer')
    parser.add_argument('--model-path', type=str, help='path to HF model')
    parser.add_argument('--benchmark', type=str, help='benchmark name')
    parser.add_argument('--prompt-file', type=str, default='./test_prompt.txt', help='path to the prompt file')
    parser.add_argument('--batch-size', type=int, default=1, help='number of prompts to pass at each inference')
    parser.add_argument('--num-generations', type=int, default=1, help='num generations per prompt sample')
    parser.add_argument('--max-new-tokens', type=int, default=384, help='Number of tokens allowed to generate, not counting in the prompt')
    parser.add_argument('--do-sample', type=str2bool, default=True, help='Sampling or greedy decoding.')
    parser.add_argument('--temperature', type=float, default=0.2, help='Sampling temperature.')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    global_start_time = time.time()

    RUN_ID = str(uuid.uuid4())
    args = argparser()
    print('RUN_ID', RUN_ID)
    print(args)
    model_name = os.path.basename(args.model_path)
    benchmark = args.benchmark
    
    if not os.path.exists(os.path.join('output',benchmark, model_name)):
        raise FileNotFoundError(f'Please create output folder output/{benchmark}/{model_name}')
    else:
        print(f'Running model {model_name} on benchmark {benchmark} ...')
    
    # padding side left important when batching
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, padding_side='left', model_max_length=2048)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, revision='main', device_map='auto')
    if not os.path.exists(args.model_path):
        raise NotImplementedError(f'Unrecognized model : {model_name}')
    
    # define generation strategy
    if model_name == 'Phind-CodeLlama-34B-v2':
        generation_strategy = PhindGeneration()
    else:
        generation_strategy = HFBaseGeneration()
    
    # define benchmark
    if benchmark == 'HumanEval':
        benchmark_strategy = HumanEvalBenchmark()
    elif benchmark == 'MBPP':
        benchmark_strategy = MBPPBenchmark()
    else:
        raise NotImplementedError
        
    # run inference
    outputs = benchmark_strategy.run_benchmark(tokenizer, model, generation_strategy, args)
    
    # save
    file_path = os.path.join('output', benchmark, model_name, f'T{str(args.temperature)}_{RUN_ID}.jsonl')
    save_to_jsonl(file_path, outputs)
    global_end_time = time.time()
    print(f'Total ellapsed time (s) {global_end_time - global_start_time}', flush=True)

