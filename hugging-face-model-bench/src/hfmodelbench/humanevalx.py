import json
import re

from huggingface_hub import InferenceClient
from loguru import logger


def evaluate_on_humanevalx(
    test_file: str = None,
    target_file: str = None,
    host: str = None,
    port: int = None,
    n: int = 200,
    inference_parameters: dict = None,
):
    client = InferenceClient(model=f"http://{host}:{str(port)}")

    with open(test_file, encoding="UTF-8") as input_file:
        for line in input_file:
            if re.match("^$", line):
                continue

            payload = json.loads(line)
            prompt = payload.get("prompt")
            task_id = payload.get("task_id")

            logger.info(f"Processing task {task_id}")
            results = []
            for _ in range(1, n + 1):
                generated = client.text_generation(prompt, **inference_parameters)

                results.append(
                    {
                        "task_id": task_id,
                        "generation": generated,
                        "prompt": prompt,
                    }
                )

            with open(target_file, "a+", encoding="UTF-8") as output_file:
                for item in results:
                    output_file.write(f"{json.dumps(item)}\n")
