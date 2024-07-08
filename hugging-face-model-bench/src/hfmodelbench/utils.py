import json
import time

import pendulum
import requests
from loguru import logger
from requests.exceptions import ConnectionError


def load_inference_parameters(parameter_file: str = None):
    with open(parameter_file, encoding="UTF-8") as input_file:
        payload = json.load(input_file)

    parameters = payload.get("inference", {})

    return parameters


def check_whether_api_is_ready(
    host: str = "localhost", port: int = 8089, timeout: int = 300
):
    full_url = f"http://{host}:{port}/health"

    start = pendulum.now()
    end = pendulum.now()
    duration = end - start
    reachable = False

    while not reachable and duration.seconds < timeout:
        try:
            r = requests.get(full_url)
        except ConnectionError:
            end = pendulum.now()
            duration = end - start
            logger.info('REST API is unreachable, waiting 10"')
            time.sleep(10)
            continue

        if r.status_code != 200:
            end = pendulum.now()
            duration = end - start
            logger.info('REST API is unreachable, waiting 10"')
            time.sleep(10)
            continue

        reachable = True
        logger.info("REST API is reachable, quitting")
