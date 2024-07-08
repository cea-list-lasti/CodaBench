import sys

from human_eval.data import write_jsonl, read_problems

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.multiprocessing as mp
from torch.multiprocessing import Pool

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder2-3b")

def generate_code(prompt, max_length=200):
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate code
    outputs = model.generate(inputs["input_ids"], max_length=max_length, num_return_sequences=1)

    # Decode the generated code
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_code

def process_prompts(task_id, prompt, num_generations, device):
    # Set the device for the current process
    torch.cuda.set_device(device)
    model.to(device)

    results = []
    for _ in range(num_generations):
        result = generate_code(prompt)
        results.append((task_id, result))

    return results

def chunk_dict(d, n):
    """Split a dictionary into n roughly equal parts."""
    items = list(d.items())
    chunk_size = len(items) // n
    chunks = [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]
    return chunks

if __name__ == '__main__':
    # Dictionary of task_id to prompt
    tasks = read_problems()

    # Number of code generations per task
    num_generations = 200

    # Number of GPUs
    num_gpus = 10

    # Split the tasks dictionary into roughly equal chunks for each GPU
    task_chunks = chunk_dict(tasks, num_gpus)

    # Start the multiprocessing pool with the number of GPUs
    with Pool(num_gpus) as p:
        # Assign each chunk of tasks to a different GPU
        results = p.starmap(
            process_prompts,
            [(task_id, prompt, num_generations, i)
             for i, task_chunk in enumerate(task_chunks)
             for task_id, prompt in task_chunk.items()])

    # Flatten the list of results
    results = [item for sublist in results for item in sublist]

    # Print or save the results
    for task_id, result in results:
        print(f"{task_id}: {result}")
    write_jsonl("samples.jsonl", results)
