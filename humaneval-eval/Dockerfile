FROM python:3.10

# RUN git clone https://github.com/openai/human-eval.git

# Set the working directory
WORKDIR /app
COPY ./human-eval ./human-eval

RUN pip install --upgrade pip && pip install -r human-eval/requirements.txt


COPY ./src ./src

# Install the human-eval package
RUN pip install -e human-eval

# evaluate
WORKDIR /app/human-eval
EXPOSE 8000

# To run, bind-mount problems and test dirs and pass the problems and test file
# as parameters
ENTRYPOINT [ "python", \
             "/app/human-eval/human_eval/evaluate_functional_correctness.py", \
             "--problem_file"]
