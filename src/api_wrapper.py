import time
import json
import timeit
import logging
import requests
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.Formatter.converter = time.localtime
logging.basicConfig(
    filename='open_source.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

models = {
    'llama8b': 'llama3.1',
    'gemma9b': 'gemma2',
    'mistral7b': 'mistral',
    'mistral-nemo12b': 'mistral-nemo',
    'mixtral7b': 'mixtral:8x7b',
    'gemma27b': 'gemma2:27b',
    'llama70b': 'llama3.1:70b',
}

BASE_URL = 'http://localhost:11434'
ENDPOINT = '/api/generate'
MAX_WORKERS = 4

def initialize() -> str:
    """
    Make sure the ollama server is running and return the URL to use for requests.

    :return: String: URL to use for requests
    """

    try:
        response = requests.get(BASE_URL)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print('ollama is not running on localhost:11434')
        logging.error(f'Failed to connect to ollama server: {e}')
        raise e

    return BASE_URL + ENDPOINT

def process_response(response, llm_name: str):
    """
    Process the response from the ollama server.

    :param response: Response from the server
    :param llm_name: String: Name of the model
    :return: Parsed JSON response or empty dictionary if parsing fails
    """
    try:
        return json.loads(response['response'])
    except (json.JSONDecodeError, KeyError) as e:
        print(f'Error decoding JSON: {e}')
        logging.error(f'Error decoding JSON for model {llm_name}: {e}. Raw model output: {response["response"]}', exc_info=True)
        return {}

def query_ollama(
        url: str,
        model_code: str,
        user_prompt: str,
        system_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 500,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 0.9
) -> Dict:

    """
    Query the ollama server with the given parameters.

    :param url: String: URL to use for requests
    :param model_code: String: Model code to use for the request
    :param user_prompt: String: User prompt to send to the server
    :param system_prompt: String: System prompt to send to the server
    :param temperature: Float: Temperature to use for the request
    :param max_tokens: Int: Maximum number of tokens to generate
    :param top_p: Float: Top-p value to use for the request
    :param top_k: Int: Top-k value to use for the request
    :param repeat_penalty: Float: Repeat penalty to use for the request
    :return: Dict: Response from the server
    """

    options = {
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p,
        'top_k': top_k,
        'repeat_penalty': repeat_penalty,
    }

    payload = {
        'model': model_code,
        'user_prompt': user_prompt,
        'system_prompt': system_prompt,
        'options': options,
        'format': 'json',
        'stream': False,
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f'Failed to query ollama server: {e}')
        raise e

def worker(
        url: str,
        model_code: str,
        user_prompt: str,
        system_prompt: str,
        options: Optional[Dict] = None
) -> tuple[Dict, float]:

    """
    Worker function for the ollama pipeline.

    :param url: String: URL to use for requests
    :param model_code: String: Model code to use for the request
    :param user_prompt: String: User prompt to send to the server
    :param system_prompt: String: System prompt to send to the server
    :param options: Dict: Additional options to send to the server
    :return: Tuple: Processed response and processing time in seconds
    """

    user_prompt = str(user_prompt)
    start_time = timeit.default_timer()

    if not options:
        response = query_ollama(url, model_code, user_prompt, system_prompt)
    else:
        temperature = options.get('temperature', 0.3)
        max_tokens = options.get('max_tokens', 500)
        top_p = options.get('top_p', 0.9)
        top_k = options.get('top_k', 40)
        repeat_penalty = options.get('repeat_penalty', 0.9)
        response = query_ollama(url, model_code, user_prompt, system_prompt, temperature, max_tokens, top_p, top_k, repeat_penalty)

    end_time = timeit.default_timer()
    return process_response(response, model_code), end_time - start_time

def send_batched_ollama_queries(
        url: str,
        inputs: list,
        model_code: str,
        system_prompt: str,
        options: Optional[Dict] = None,
        max_workers: int = MAX_WORKERS
) -> Tuple[List[Dict], float]:

    """
    Send batched queries to the ollama server.

    :param url: String: URL to use for requests
    :param inputs: List: List of input data
    :param model_code: String: Model code to use for the request
    :param system_prompt: String: System prompt to send to the server
    :param options: Dict: Additional options to send to the server
    :param max_workers: Int: Maximum number of workers to use
    :return:
    """

    start_time = timeit.default_timer()
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, url, model_code, input_data, system_prompt, options) for input_data in inputs]
        for future in as_completed(futures):
            results.append(future.result())

    end_time = timeit.default_timer()
    logging.info(f'Batch processing completed in {end_time - start_time:.2f} seconds.')

    return results, end_time - start_time

# ----------------------------------- conform with given API -----------------------------------

def prompt(
        run_id: str,
        pkg: str,
        task: str,
        model: str,
        system_msg: str,
        user_msg: str,
        max_tokens: int = 2048,
        examples: Optional[List[Tuple[str, str]]] = None
) -> tuple[str, float]:

    """
    Prompt the ollama server with the given parameters.

    :param run_id: String: ID of the run
    :param pkg: String: Name of the package
    :param task: String: Name of the task
    :param model: String: Model code to use for the request
    :param system_msg: String: System message to send to the server
    :param user_msg: String: User message to send to the server
    :param max_tokens: Int: Maximum number of tokens to generate
    :param examples: **UNUSED** List: List of examples to send to the server
    :return: Tuple: Response from the server and the processing time of the request
    """

    url = initialize()
    response, processing_time = worker(url, model, user_msg, system_msg, {'max_tokens': max_tokens})

    if run_id is not None and pkg is not None:
        # log the processing time and response
        with open(f"output/{run_id}/{model}-responses/times-{task}.csv", "a") as f:
            f.write(f"{pkg},{processing_time}\n")
        with open(f"output/{run_id}/{model}-responses/{task}/{pkg}.json", "w") as f:
            f.write(json.dumps(response, indent=4))

    return json.dumps(response), processing_time

# ----------------------------------- new for batching with ollama -----------------------------------

def prompt_batched(
        run_id: str,
        pkg: str,
        task: str,
        model: str,
        system_msg: str,
        user_msgs: list[str],
        max_tokens: int = 2048,
        examples: Optional[List[Tuple[str, str]]] = None
) -> tuple[str, float]:

    """
    Prompt the ollama server with multiple user messages in a batch.

    :param run_id: String: ID of the run
    :param pkg: String: Name of the package
    :param task: String: Name of the task
    :param model: String: Model code to use for the request
    :param system_msg: String: System message to send to the server
    :param user_msgs: List: List of user messages to send to the server
    :param max_tokens: Int: Maximum number of tokens to generate
    :param examples: **UNUSED** List: List of examples to send to the server
    :return: Tuple: Response from the server and the processing time of the request
    """

    url = initialize()
    responses, processing_time = send_batched_ollama_queries(url, user_msgs, model, system_msg, {'max_tokens': max_tokens})

    if run_id is not None and pkg is not None:
        # log the processing time and response
        with open(f"output/{run_id}/{model}-responses/times-{task}.csv", "a") as f:
            for i, response in enumerate(responses):
                f.write(f"{pkg}_{i},{processing_time}\n")
        for i, response in enumerate(responses):
            with open(f"output/{run_id}/{model}-responses/{task}/{pkg}_{i}.json", "w") as f:
                f.write(json.dumps(response, indent=4))

    return json.dumps(responses), processing_time
