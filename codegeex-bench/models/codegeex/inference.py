import time
import sys
import argparse
import json
import torch
import codegeex
from codegeex.torch import CodeGeeXModel
from codegeex.tokenizer import CodeGeeXTokenizer
from codegeex.quantization import quantize
import uuid
import numpy
import random


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
        "--seed",
        type=int,
        default=12,
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

def save_to_jsonl(file_path, data):
    with open(file_path, 'w') as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')
            
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
        for line in lines:
            item = json.loads(line.strip())
            data.append(item)
    return data
        
def main():
    run_id = str(uuid.uuid4())
    print("run uuid", run_id)
    
    parser = argparse.ArgumentParser()
    parser = add_code_generation_args(parser)
    args, _ = parser.parse_known_args()
    
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


    if(args.seed):
        print(f"setting seed {args.seed}")
        _set_random_seed(args.seed)

    print("args",args)
    print("Loading tokenizer ...")
    tokenizer = CodeGeeXTokenizer(
        tokenizer_path=args.tokenizer_path, 
        mode="codegeex-13b")
    
   
    print("Loading state dict ...")
    state_dict = torch.load(args.load, map_location="cpu")
    state_dict = state_dict["module"]

    print("Building CodeGeeX model ...")
    print("args", args)
    sys.stdout.flush()
    model = model_provider(args)
    model.load_state_dict(state_dict)
    model.eval()
    model.half()
    if args.quantize:
        model = quantize(model, weight_bit_width=8, backend="torch")
    model.cuda()
    torch.cuda.synchronize()
    
    prompts_data = read_jsonl(args.prompt_file)
    print("number of prompts", len(prompts_data))
    
    outputs = []
    preprompt = "# language: Python\n"
    # preprompt = ""
    print("preprompt", preprompt)
    # out_seq_lengths = [args.out_seq_length]
    for i,prompt_data in enumerate(prompts_data):   
        task_id = prompt_data['task_id']   
        prompt = preprompt+prompt_data['prompt']
        if not prompt:
            print('Query should not be empty!')
            continue
        if prompt == "stop":
            return
        try:
            gen = []
            for i in range(args.num_generations):
                t0 = time.perf_counter()
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
                gen+= generated_code
                
            outputs.append(dict(task_id=f"{task_id}", completion=gen))
            t1 = time.perf_counter()
            print("Total generation time:", t1 - t0)
        except (ValueError, FileNotFoundError) as e:
            print(e)
            continue
    
    file_path = f'output/seeds/output_{args.seed}_T{args.temperature}_{run_id}.jsonl'
    # Call the function to save the data
    save_to_jsonl(file_path, outputs)



        
print("Generation finished.")


if __name__ == "__main__":
    main()