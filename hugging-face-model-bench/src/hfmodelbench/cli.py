import json
import os.path
import sys

import click
from loguru import logger

from hfmodelbench.humanevalx import evaluate_on_humanevalx
from hfmodelbench.utils import check_whether_api_is_ready, load_inference_parameters


@click.group()
@click.option("--debug", is_flag=True)
def cli(debug):
    if not debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO")


@cli.command("temporize")
@click.option(
    "-h",
    "--host",
    help="Hostname of the machine where codellama is running",
    required=False,
    type=str,
    default="localhost",
    show_default=True,
)
@click.option(
    "-p",
    "--port",
    help="Port of the machine on which codellama is running",
    type=int,
    required=False,
    default=8080,
    show_default=True,
)
@click.option(
    "-t",
    "--timeout",
    help="Number of seconds to wait for the API to complete loading",
    required=False,
    type=int,
    default=300,
    show_default=True,
)
def temporize_cli(host: str = None, port: int = None, timeout: int = None):
    check_whether_api_is_ready(host=host, port=port, timeout=timeout)


@cli.command("arguments")
@click.option(
    "--configuration-file", help="Model configuration file", type=str, required=True
)
def arguments_cli(configuration_file: str = None):
    configuration_file = os.path.abspath(configuration_file)

    if not os.path.isfile(configuration_file):
        raise FileNotFoundError("Configuration file does not exist")

    with open(configuration_file, encoding="UTF-_8") as input_file:
        payload = json.load(input_file)

    print(" ".join([f"{k} {v}" for k, v in payload.get("tgi").items()]))


@cli.command("evaluate-on-humaneval")
@click.option(
    "--humaneval-test-file", help="Path to humaneval test file", required=True, type=str
)
@click.option(
    "-o",
    "--output-file",
    help="File where generations will be written",
    required=True,
    type=str,
)
@click.option("--parameter-file", help="Model parameter file", type=str, required=True)
@click.option(
    "-n",
    help="Number of samples to generate by problem",
    default=200,
    type=int,
    show_default=True,
    required=False,
)
@click.option(
    "-h",
    "--host",
    help="Hostname of the machine where codellama is running",
    required=False,
    type=str,
    default="localhost",
    show_default=True,
)
@click.option(
    "-p",
    "--port",
    help="Port of the machine on which codellama is running",
    type=int,
    required=False,
    default=8080,
    show_default=True,
)
def evaluate_on_humaneval_cli(
    humaneval_test_file: str = None,
    output_file: str = None,
    parameter_file: str = None,
    n: int = None,
    host: str = None,
    port: int = None,
):
    humaneval_test_file = os.path.abspath(humaneval_test_file)
    output_file = os.path.abspath(output_file)
    parameter_file = os.path.abspath(parameter_file)

    if not os.path.isfile(humaneval_test_file):
        raise FileNotFoundError("HumanEval file not found")

    if os.path.isfile(output_file):
        raise FileExistsError("Output file does already exist")

    if not os.path.isfile(parameter_file):
        raise FileNotFoundError("Parameter file does not exist")

    inference_parameters = load_inference_parameters(parameter_file=parameter_file)

    evaluate_on_humanevalx(
        test_file=humaneval_test_file,
        target_file=output_file,
        n=n,
        host=host,
        port=port,
        inference_parameters=inference_parameters,
    )


if __name__ == "__main__":
    cli()
