from human_eval.data import write_jsonl, read_problems

problems = read_problems()
print("problems", problems)


# place the LLM call here
def generate_one_completion(prompt : str):
    print("prompt", prompt)
    return "if name='rkd': pass"
    
num_samples_per_task = 10
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
print("samples", samples)
write_jsonl("samples.jsonl", samples)