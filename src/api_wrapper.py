import json
import timeit
import asyncio
import logging
import tiktoken
from ollama import AsyncClient, Options
from typing import Dict, Optional, List, Tuple, Literal, Union

models = {
    'llama3': 'llama3:instruct',
    'llama8b': 'llama3.1:latest',
    'llama8b-instruct': 'llama3.1:8b-instruct-q4_0',
    'llama8b-instruct-fp': 'llama3.1:8b-instruct-fp16',
    'llama70b-instruct': 'llama3.1:70b-instruct-q3_K_L',
    'gemma2b': 'gemma2:2b',
    'gemma9b': 'gemma2:latest',
    'gemma27b': 'gemma2:27b',
    'mistral7b': 'mistral:latest',
    'mistral-nemo12b': 'mistral-nemo:latest',
    'mixtral8x7b': 'mixtral:8x7b',
}

TEMPERATURE = 0.5
MAX_TOKENS = 500
TOP_P = 0.9
TOP_K = 40
REPEAT_PENALTY = 1.1
CONTEXT_WINDOW = 8192


def process_response(response, llm_name: str, json_format: bool = True) -> Union[Dict, str]:
    """
    Process the response from the ollama server.

    :param response: Response from the server
    :param llm_name: String: Name of the model
    :param json_format: Bool: Whether to return the response as JSON
    :return: Parsed JSON response or empty dictionary if parsing fails
    """
    try:
        logging.debug(f'Ouput from model {llm_name}: {response["response"].strip()}')
        if json_format:
            return json.loads(response['response'])
        else:
            return response['response']
    except (json.JSONDecodeError, KeyError) as e:
        print(f'Error decoding JSON: {e}')
        logging.error(f'Error decoding JSON for model {llm_name}: {e}. Raw model output: {response["response"]}', exc_info=True)
        return {}


async def query_ollama(
        model_code: str,
        user_prompt: str,
        system_prompt: str,
        ollama_client: AsyncClient,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        top_p: float = TOP_P,
        top_k: int = TOP_K,
        repeat_penalty: float = REPEAT_PENALTY,
        context_window: int = CONTEXT_WINDOW,
        output_format: Literal['json', ''] = '',
        keep_alive: int = None
) -> Union[Dict, str]:
    """
    Query the ollama server with the given parameters.

    :param model_code: String: Model code to use for the request
    :param user_prompt: String: User prompt to send to the server
    :param system_prompt: String: System prompt to send to the server
    :param ollama_client: AsyncClient: Ollama client to use for the request
    :param temperature: Float: Temperature to use for the request
    :param max_tokens: Int: Maximum number of tokens to generate
    :param top_p: Float: Top-p value to use for the request
    :param top_k: Int: Top-k value to use for the request
    :param repeat_penalty: Float: Repeat penalty to use for the request
    :param context_window: Int: Context window length in tokens to use for the request
    :param output_format: String: Output format to use for the request
    :param keep_alive: Int: Keep alive value in seconds to use for the request
    :return: Dict or String: Response from the server
    """

    options = {
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p,
        'top_k': top_k,
        'repeat_penalty': repeat_penalty,
        'num_ctx': context_window,
    }

    options = Options(**options)

    # calculate the token count of the prompt
    encoding = tiktoken.get_encoding('cl100k_base')
    user_prompt_token_count = len(encoding.encode(user_prompt))
    # calculate the token count of the system prompt
    system_prompt_token_count = len(encoding.encode(system_prompt))

    logging.debug(
        f'Querying model {model_code} with:\nUser prompt: {user_prompt}\nSystem prompt: {system_prompt[:100]}...\nOptions: {options}\nOutput format: {output_format if output_format else "default"}')
    logging.debug(f'User prompt token count: {user_prompt_token_count}')
    logging.debug(f'System prompt token count: {system_prompt_token_count}')

    try:
        response = process_response(
            await ollama_client.generate(model=model_code, prompt=user_prompt, system=system_prompt, format=output_format, options=options, keep_alive=keep_alive),
            llm_name=model_code,
            json_format=output_format == 'json'
        )
        return response
    except Exception as e:
        logging.error(f'Error querying model {model_code}: {e}', exc_info=True)
        return {}


def prompt(
        run_id: str,
        pkg: str,
        task: str,
        model: str,
        ollama_client: AsyncClient,
        system_msg: str,
        user_msg: str,
        options: Optional[Dict] = None,
        json_format: bool = True,
        examples: Optional[List[Tuple[str, str]]] = None
) -> Tuple[Union[Dict, str], float]:
    """
    Prompt the ollama server with the given parameters.

    :param run_id: String: ID of the run
    :param pkg: String: Name of the package
    :param task: String: Name of the task
    :param model: String: Model code to use for the request
    :param ollama_client: AsyncClient: Ollama client to use for the request
    :param system_msg: String: System message to send to the server
    :param user_msg: String: User message to send to the server
    :param options: Dict: Additional options to send to the server
    :param json_format: Bool: Whether to return the response as JSON
    :param examples: **UNUSED** List: List of examples to send to the server
    :return: Tuple: Response from the server and the processing time of the request
    """

    user_prompt = str(user_msg)
    loop = asyncio.get_event_loop()

    start_time = timeit.default_timer()

    if not options:
        response = loop.run_until_complete(query_ollama(model, user_prompt, system_msg, ollama_client, output_format='json' if json_format else ''))
    else:
        temperature = options.get('temperature', TEMPERATURE)
        max_tokens = options.get('max_tokens', MAX_TOKENS)
        top_p = options.get('top_p', TOP_P)
        top_k = options.get('top_k', TOP_K)
        repeat_penalty = options.get('repeat_penalty', REPEAT_PENALTY)
        context_window = options.get('num_ctx', CONTEXT_WINDOW)

        response = loop.run_until_complete(query_ollama(
            model,
            user_prompt,
            system_msg,
            ollama_client,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            context_window=context_window,
            output_format='json' if json_format else ''
        ))

    end_time = timeit.default_timer()
    processing_time = end_time - start_time
    logging.info(f'Processing time for model {model}: {processing_time}')

    if run_id is not None and pkg is not None:
        # log the processing time and response
        with open(f"output/{run_id}/{model}_responses/times-{task}.csv", "a") as f:
            f.write(f"{pkg},{processing_time}\n")
        with open(f"output/{run_id}/{model}_responses/{task}/{pkg}.json", "w") as f:
            f.write(json.dumps(response, indent=4))

    if json_format:
        return json.dumps(response), processing_time
    else:
        return response, processing_time

def load_model(ollama_client:AsyncClient, model: str):
    """
    Load the specified model by sending an empty query to the server.

    :param ollama_client: AsyncClient: Ollama client to use for the request
    :param model: String: Model code to load
    """

    loop = asyncio.get_event_loop()
    loop.run_until_complete(query_ollama(model, "", "", ollama_client))

def unload_model(ollama_client:AsyncClient, model: str):
    """
    Unload the specified model by sending an empty query to the server with a keep_alive value of 0.

    :param ollama_client: AsyncClient: Ollama client to use for the request
    :param model: String: Model code to unload
    """

    loop = asyncio.get_event_loop()
    loop.run_until_complete(query_ollama(model, "", "", ollama_client, keep_alive=0))
