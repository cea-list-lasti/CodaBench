FROM python:3.9

ENV DATA_PATH="humaneval/human-eval/data/generations/codegeex/seeds"

# Install only requirements first
COPY ./human-eval/requirements.txt ./human-eval/requirements.txt
RUN pip install -r human-eval/requirements.txt


# Set the working directory
WORKDIR /app
COPY ./human-eval ./human-eval
COPY ./src ./src

# Install the human-eval package
RUN pip install -e human-eval

# evaluate
WORKDIR /app/human-eval
EXPOSE 8000

CMD python "/app/human-eval/human_eval/evaluate_functional_correctness.py" "$DATA_PATH" "--problem_file=/app/human-eval/data/HumanEval.jsonl"