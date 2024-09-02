import json
import timeit
import asyncio
import logging
import tiktoken
from src import util
from ollama import AsyncClient, Options
from typing import Dict, Optional, List, Tuple, Literal, Union

models = {
    'llama3': 'llama3:instruct',
    'llama8b': 'llama3.1:8b-instruct-q4_0',
    'llama8b-fp16': 'llama3.1:8b-instruct-fp16',
    'llama70b': 'llama3.1:70b-instruct-q3_K_L',
    'gemma9b': 'gemma2:9b-instruct-q4_0',
    'gemma27b': 'gemma2:27b-instruct-q4_0',
    'mistral7b': 'mistral:instruct',
    'mistral-nemo12b': 'mistral-nemo:12b-instruct-2407-q4_0',
    'mixtral8x7b': 'mixtral:instruct',
    'mixtral8x22b': 'mixtral:8x22b-instruct',
    'mistral-large': 'mistral-large:123b-instruct-2407-q4_0',
}

TEMPERATURE = 0.4
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
        f'Querying model {model_code} with:\nUser prompt: {user_prompt}\nSystem prompt: {system_prompt[:100]}...\nOptions: {options}\n'
        f'Output format: {output_format if output_format else "default"}\n'
        f'User prompt token count: {user_prompt_token_count}\nSystem prompt token count: {system_prompt_token_count}')

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


async def query_ollama_batched(
        model_code: str,
        user_prompts: List[str],
        system_prompt: str,
        ollama_client: AsyncClient,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        top_p: float = TOP_P,
        top_k: int = TOP_K,
        repeat_penalty: float = REPEAT_PENALTY,
        context_window: int = CONTEXT_WINDOW,
        output_format: Literal['json', ''] = '',
        keep_alive: int = None,
        concurrent_requests: int = 4
) -> List[Union[Dict, str]]:
    """
    Query the ollama server with the given parameters using a thread pool to handle multiple requests concurrently.

    :param model_code: String: Model code to use for the request
    :param user_prompts: List[String]: List of user prompts to send to the server
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
    :param concurrent_requests: Int: Number of concurrent requests to send to the server
    :return: List[Dict or String]: List of responses from the server
    """

    request_options = {
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p,
        'top_k': top_k,
        'repeat_penalty': repeat_penalty,
        'num_ctx': context_window,
    }

    options = Options(**request_options)

    async def send_request(prompt: str) -> Union[Dict, str]:
        # calculate the token count of the prompt
        encoding = tiktoken.get_encoding('cl100k_base')
        prompt_token_count = len(encoding.encode(prompt))
        system_prompt_token_count = len(encoding.encode(system_prompt))

        logging.debug(
            f'Querying model {model_code} with:\nUser prompt: {prompt}\nSystem prompt: {system_prompt[:100]}...\nOptions: {options}\n'
            f'Output format: {output_format if output_format else "default"}\n'
            f'User prompt token count: {prompt_token_count}\nSystem prompt token count: {system_prompt_token_count}')

        try:
            response = process_response(
                await ollama_client.generate(model=model_code, prompt=prompt, system=system_prompt, format=output_format, options=options, keep_alive=keep_alive),
                llm_name=model_code,
                json_format=output_format == 'json'
            )
            return response
        except Exception as e:
            logging.error(f'Error querying model {model_code}: {e}', exc_info=True)
            return {}

    async def worker(task_queue: asyncio.Queue, result_list: List[Union[Dict, str]]):
        while True:
            task_prompt = await task_queue.get()
            if task_prompt is None:
                break
            result = await send_request(task_prompt)
            result_list.append(result)
            task_queue.task_done()

    task_queue = asyncio.Queue()
    for prompt in user_prompts:
        task_queue.put_nowait(prompt)

    result_list = []
    workers = [asyncio.create_task(worker(task_queue, result_list)) for _ in range(concurrent_requests)]

    await task_queue.join()

    for _ in range(concurrent_requests):
        task_queue.put_nowait(None)

    await asyncio.gather(*workers)

    return result_list


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
        util.add_to_file(f"output/{run_id}/{model}_responses/times-{task}.csv", f"{pkg},{processing_time}\n")
        util.write_to_file(f"output/{run_id}/{model}_responses/{task}/{pkg}.json", json.dumps(response, indent=4))

    if json_format:
        return json.dumps(response), processing_time
    else:
        return response, processing_time

def prompt_batched(
        run_id: str,
        pkg: str,
        task: str,
        model: str,
        ollama_client: AsyncClient,
        system_msg: str,
        user_msgs: List[str],
        options: Optional[Dict] = None,
        json_format: bool = True,
        examples: Optional[List[Tuple[str, str]]] = None
) -> Tuple[List[Union[Dict, str]], float]:
    """
    Prompt the ollama server with the given parameters using a thread pool to handle multiple requests concurrently.

    :param run_id: String: ID of the run
    :param pkg: String: Name of the package
    :param task: String: Name of the task
    :param model: String: Model code to use for the request
    :param ollama_client: AsyncClient: Ollama client to use for the request
    :param system_msg: String: System message to send to the server
    :param user_msgs: List[String]: List of user messages to send to the server
    :param options: Dict: Additional options to send to the server
    :param json_format: Bool: Whether to return the response as JSON
    :param examples: **UNUSED** List: List of examples to send to the server
    :return: Tuple: List of responses from the server and the processing time of the request
    """

    loop = asyncio.get_event_loop()

    start_time = timeit.default_timer()

    if not options:
        responses = loop.run_until_complete(query_ollama_batched(model, user_msgs, system_msg, ollama_client, output_format='json' if json_format else ''))
    else:
        temperature = options.get('temperature', TEMPERATURE)
        max_tokens = options.get('max_tokens', MAX_TOKENS)
        top_p = options.get('top_p', TOP_P)
        top_k = options.get('top_k', TOP_K)
        repeat_penalty = options.get('repeat_penalty', REPEAT_PENALTY)
        context_window = options.get('num_ctx', CONTEXT_WINDOW)

        responses = loop.run_until_complete(query_ollama_batched(
            model,
            user_msgs,
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
        for i, response in enumerate(responses):
            util.write_to_file(f"output/{run_id}/{model}_responses/{task}/{pkg}_{i}.json", json.dumps(response, indent=4))
        util.add_to_file(f"output/{run_id}/{model}_responses/times-{task}.csv", f"{pkg},{processing_time}\n")

    if json_format:
        return [json.dumps(response) for response in responses], processing_time
    else:
        return responses, processing_time


def load_model(ollama_client: AsyncClient, model: str):
    """
    Load the specified model by sending an empty query to the server.

    :param ollama_client: AsyncClient: Ollama client to use for the request
    :param model: String: Model code to load
    """

    loop = asyncio.get_event_loop()
    loop.run_until_complete(query_ollama(model, "", "", ollama_client))

def unload_model(ollama_client: AsyncClient, model: str):
    """
    Unload the specified model by sending an empty query to the server with a keep_alive value of 0.

    :param ollama_client: AsyncClient: Ollama client to use for the request
    :param model: String: Model code to unload
    """

    loop = asyncio.get_event_loop()
    loop.run_until_complete(query_ollama(model, "", "", ollama_client, keep_alive=0))
