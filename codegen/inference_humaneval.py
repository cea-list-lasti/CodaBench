from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
import uuid
import time



def save_to_jsonl(file_path, data):
    """ save a list of dict to

    Args:
        file_path (_type_): _description_
        data (_type_): _description_
    """
    with open(file_path, 'w', encoding='UTF-8') as file:
        for item in data:
            json.dump(item, file)
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

def argparser():
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument(
        "--num-generations",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="./test_prompt.txt",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str
    )
    parser.add_argument(
        "--model-path",
        type=str
    )
    args = parser.parse_args()
    return args


def generate(text):
    """ generate text based on prompt
    Args:
        text (_type_): _description_

    Returns:
        _type_: _description_
    """
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=256)
    generated = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(generated)
    return generated

def generate2(text):
    inputs = tokenizer(text, return_tensors="pt")
    sample = model.generate(**inputs, max_length=256)
    generated = tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])
    print(f"generated ++++ \n {generated} \n\n")
    return generated

def generate3(text, temperature=0.8):
    inputs = tokenizer(text, return_tensors="pt")
    sample = model.generate(**inputs,do_sample=True,temperature=temperature, top_p=0.95, max_length=256, pad_token_id=tokenizer.eos_token_id,)
    generated = tokenizer.decode(sample[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])
    print(f"generated ++++ \n {generated} \n\n")
    return generated


if __name__=="__main__":
    run_id = str(uuid.uuid4())
    print("run_id", run_id)
    args = argparser()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, revision="main")
    prompts_data = read_jsonl(args.prompt_file)

    outputs = []
    for i,prompt_data in enumerate(prompts_data):
        print(f"{i}=================", flush=True)
        start = time.time()
        print(f"prompt ++++ \n {prompt_data['prompt']} \n\n")
        completion = generate3(prompt_data["prompt"], temperature=0.8)
        outputs.append(dict(task_id=f"{prompt_data['task_id']}", completion=completion))
        end = time.time()
        print(f"time (s) {end-start}")

    file_path = f'output/{run_id}.jsonl'
    # Call the function to save the data
    save_to_jsonl(file_path, outputs)
