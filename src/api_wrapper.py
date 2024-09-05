import json
import timeit
import ollama
import asyncio
import logging
import tiktoken
from openai import OpenAI
from ollama import AsyncClient, Options
from typing import Dict, Optional, List, Tuple, Literal, Union

from src import util

OLLAMA_MODELS = {
    'llama3': 'llama3:instruct',
    'llama8b': 'llama3.1:8b-instruct-q4_0',
    'llama8b-fp16': 'llama3.1:8b-instruct-fp16',
    'llama70b': 'llama3.1:70b-instruct-q3_K_S',
    'gemma9b': 'gemma2:9b-instruct-q4_0',
    'gemma27b': 'gemma2:27b-instruct-q4_0',
    'mistral7b': 'mistral:instruct',
    'mistral-nemo12b': 'mistral-nemo:12b-instruct-2407-q4_0',
    'mixtral8x7b': 'mixtral:instruct',
    'phi3b': 'phi3:3.8b-mini-128k-instruct-q4_0',
    'phi3b-fp16': 'phi3:3.8b-mini-128k-instruct-fp16',
    'phi14b': 'phi3:14b-medium-128k-instruct-q4_0',
    'phi14b-fp16': 'phi3:14b-medium-128k-instruct-fp16',
    'qwen7b': 'qwen2:7b-instruct',
    'qwen7b-fp16': 'qwen2:7b-instruct-fp16',
    'qwen72b': 'qwen2:72b-instruct',
}

OPENAI_MODELS = {
    'gpt-3.5-turbo': {
        'name': 'gpt-3.5-turbo',
        'encoding': 'cl100k_base',
        'input_price': (0.5 / 1000000),
        'output_price': (1.5 / 1000000),
        'input_price_batch': (0.25 / 1000000),
        'output_price_batch': (0.75 / 1000000)
    },
    'gpt-4': {
        'name': 'gpt-4',
        'encoding': 'cl100k_base',
        'input_price': (30 / 1000000),
        'output_price': (60 / 1000000),
        'input_price_batch': (15 / 1000000),
        'output_price_batch': (30 / 1000000)
    },
    'gpt-4o': {
        'name': 'gpt-4o',
        'encoding': 'o200k_base',
        'input_price': (5 / 1000000),
        'output_price': (15 / 1000000),
        'input_price_batch': (2.5 / 1000000),
        'output_price_batch': (7.5 / 1000000)
    },
    'gpt-4o-mini': {
        'name': 'gpt-4o-mini',
        'encoding': 'o200k_base',
        'input_price': (0.15 / 1000000),
        'output_price': (0.6 / 1000000),
        'input_price_batch': (0.075 / 1000000),
        'output_price_batch': (0.3 / 1000000)
    }
}

TEMPERATURE = 0.4
MAX_TOKENS = 500
TOP_P = 0.9
TOP_K = 40
CONTEXT_WINDOW = 8192


def process_ollama_response(response, llm_name: str, json_format: bool = True) -> Union[Dict, str]:
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
        print(f'Error decoding JSON: {e}. Continuing with empty dictionary...')
        logging.error(f'Error decoding JSON for model {llm_name}: {e}. Raw model output: {response["response"]}', exc_info=True)
        return {}


async def send_ollama_request(
        model_code: str,
        prompt: str,
        system_prompt: str,
        ollama_client: AsyncClient,
        options: Optional[Dict] = None,
        output_format: Literal['json', ''] = '',
        keep_alive: int = None
) -> Union[Dict, str]:

    """
    Send a request to the Ollama server with the given parameters.

    :param model_code: String: Model code to use for the request
    :param prompt: String: User prompt to send to the server
    :param system_prompt: String: System prompt to send to the server
    :param ollama_client: AsyncClient: Ollama client to use for the request
    :param options: Dict: Additional options to send to the server
    :param output_format: String: Output format to use for the request
    :param keep_alive: Int: Keep alive value in seconds to use for the request
    :return: Dict or String: Response from the server
    """

    # calculate the token count of the prompt
    encoding = tiktoken.get_encoding('cl100k_base')
    prompt_token_count = len(encoding.encode(prompt))
    system_prompt_token_count = len(encoding.encode(system_prompt))

    logging.debug(
        f'Querying model {model_code} with:\nUser prompt: {prompt}\nSystem prompt: {system_prompt[:100]}...\nOptions: {options}\n'
        f'Output format: {output_format if output_format else "default"}\n'
        f'User prompt token count: {prompt_token_count}\nSystem prompt token count: {system_prompt_token_count}')

    try:
        response = process_ollama_response(
            await ollama_client.generate(model=model_code, prompt=prompt, system=system_prompt, format=output_format, options=options, keep_alive=keep_alive),
            llm_name=model_code,
            json_format=output_format == 'json'
        )
        return response
    except Exception as e:
        logging.error(f'Error querying model {model_code}: {e}', exc_info=True)
        return {}


async def query_ollama(
        model_code: str,
        user_prompt: str,
        system_prompt: str,
        ollama_client: AsyncClient,
        options: Optional[Dict] = None,
        output_format: Literal['json', ''] = '',
        keep_alive: int = None
) -> Union[Dict, str]:

    """
    Query the Ollama server with the given parameters.

    :param model_code: String: Model code to use for the request
    :param user_prompt: String: User prompt to send to the server
    :param system_prompt: String: System prompt to send to the server
    :param ollama_client: AsyncClient: Ollama client to use for the request
    :param options: Dict: Additional options to send to the server
    :param output_format: String: Output format to use for the request
    :param keep_alive: Int: Keep alive value in seconds to use for the request
    :return: Dict or String: Response from the server
    """

    request_options = Options(**options)

    return await send_ollama_request(model_code, user_prompt, system_prompt, ollama_client, request_options, output_format, keep_alive)


async def query_ollama_parallel(
        model_code: str,
        user_prompts: List[str],
        system_prompt: str,
        ollama_client: AsyncClient,
        options: Optional[Dict] = None,
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
    :param options: Dict: Additional options to send to the server
    :param output_format: String: Output format to use for the request
    :param keep_alive: Int: Keep alive value in seconds to use for the request
    :param concurrent_requests: Int: Number of concurrent requests to send to the server
    :return: List[Dict or String]: List of responses from the server
    """

    request_options = Options(**options)

    async def worker(task_queue: asyncio.Queue, result_list: List[Union[Dict, str]]):
        while True:
            task_prompt = await task_queue.get()
            if task_prompt is None:
                break
            result = await send_ollama_request(model_code, task_prompt, system_prompt, ollama_client, request_options, output_format, keep_alive)
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


class ApiWrapper:
    def __init__(self, client: Union[AsyncClient, OpenAI]):
        """
        The ApiWrapper class provides either a wrapper for local LLM inference using Ollama or a wrapper for OpenAI's API.

        Initialize the class with the client object from the respective service.

        :param client: Union[AsyncClient, OpenAI]: The client object from either Ollama or OpenAI
        """

        if isinstance(client, AsyncClient):
            self.client = client
            self.client_type = 'ollama'
            self.load_model = self._load_model
            self.unload_model = self._unload_model
            self.prompt = self._prompt_ollama
            self.prompt_parallel = self._prompt_ollama_parallel
        elif isinstance(client, OpenAI):
            self.client = client
            self.client_type = 'openai'
        else:
            raise ValueError('Invalid client type')

    def _fetch_downloaded_models(self) -> List[str]:
        """
        Fetch the list of downloaded models.
        This method is only callable when using Ollama for inference.
        """

        if self.client_type != 'ollama' or not isinstance(self.client, AsyncClient):
            raise RuntimeError('downloaded_models is only available for Ollama clients.')

        return [model['name'] for model in ollama.list()['models']]

    @property
    def downloaded_models(self) -> List[str]:
        """
        List of LLMs downloaded using Ollama.
        """

        return self._fetch_downloaded_models()

    def _load_model(self, model: str):
        """
        Load the specified model by sending an empty query to the server.
        This method is only callable if the client is an Ollama client.

        :param model: String: Model code to load
        """

        if self.client_type != 'ollama' or not isinstance(self.client, AsyncClient):
            raise RuntimeError('load_model is only available for Ollama clients.')

        logging.debug(f'Loading model {model}...')
        loop = asyncio.get_event_loop()
        loop.run_until_complete(query_ollama(model, "", "", self.client))

    def _unload_model(self, model: str):
        """
        Unload the specified model by sending an empty query to the server with a keep_alive value of 0.
        This method is only callable if the client is an Ollama client.

        :param model: String: Model code to unload
        """

        if self.client_type != 'ollama' or not isinstance(self.client, AsyncClient):
            raise RuntimeError('unload_model is only available for Ollama clients.')

        logging.debug(f'Unloading model {model}...')
        loop = asyncio.get_event_loop()
        loop.run_until_complete(query_ollama(model, "", "", self.client, keep_alive=0))

    def _prompt_ollama(
            self,
            run_id: str,
            pkg: str,
            task: str,
            model: str,
            system_msg: str,
            user_msg: str,
            examples: Optional[List[Tuple[str, str]]] = None,
            n: Optional[int] = 1,
            temperature: float = TEMPERATURE,
            max_tokens: int = MAX_TOKENS,
            top_p: float = TOP_P,
            top_k: int = TOP_K,
            context_window: int = CONTEXT_WINDOW,
            json_format: bool = True,
    ) -> Tuple[Union[Dict, str], float]:

        """
        Prompt the Ollama server with the given parameters.

        :param run_id: String: ID of the run
        :param pkg: String: Name of the package
        :param task: String: Name of the task
        :param model: String: Model code to use for the request
        :param system_msg: String: System message to send to the server
        :param user_msg: String: User message to send to the server
        :param examples: **UNUSED** List: List of examples to send to the server (used for OpenAI)
        :param n: Int: **UNUSED** Number of completions to generate (used for OpenAI)
        :param temperature: Float: Temperature to use for the request
        :param max_tokens: Int: Maximum number of tokens to generate
        :param top_p: Float: Top-p value to use for the request
        :param top_k: Int: Top-k value to use for the request
        :param context_window: Int: Context window length in tokens to use for the request
        :param json_format: Bool: Whether to return the response as JSON
        :return: Tuple: Response from the server and the processing time of the request
        """

        if self.client_type != 'ollama' or not isinstance(self.client, AsyncClient):
            raise RuntimeError(f'Trying to use Ollama prompt function with a non-Ollama client: {self.client_type}')

        if examples is not None:
            raise ValueError('Examples are not supported for Ollama queries.')

        if n is not None:
            raise ValueError('Multiple completions not supported for Ollama queries.')

        user_prompt = str(user_msg)
        loop = asyncio.get_event_loop()

        start_time = timeit.default_timer()

        options = {
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': top_p,
            'top_k': top_k,
            'num_ctx': context_window,
        }

        response = loop.run_until_complete(query_ollama(
            model,
            user_prompt,
            system_msg,
            self.client,
            options=options,
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

    def _prompt_ollama_parallel(
            self,
            run_id: str,
            pkg: str,
            task: str,
            model: str,
            system_msg: str,
            user_msgs: List[str],
            examples: Optional[List[Tuple[str, str]]] = None,
            n: Optional[int] = 1,
            temperature: float = TEMPERATURE,
            max_tokens: int = MAX_TOKENS,
            top_p: float = TOP_P,
            top_k: int = TOP_K,
            context_window: int = CONTEXT_WINDOW,
            json_format: bool = True,
    ) -> Tuple[List[Union[Dict, str]], float]:

        """
        Prompt the ollama server with the given parameters using a thread pool to handle multiple requests concurrently.

        :param run_id: String: ID of the run
        :param pkg: String: Name of the package
        :param task: String: Name of the task
        :param model: String: Model code to use for the request
        :param system_msg: String: System message to send to the server
        :param user_msgs: List[String]: List of user messages to send to the server
        :param examples: **UNUSED** List: List of examples to send to the server (used for OpenAI)
        :param n: Int: **UNUSED** Number of completions to generate (used for OpenAI)
        :param temperature: Float: Temperature to use for the request
        :param max_tokens: Int: Maximum number of tokens to generate
        :param top_p: Float: Top-p value to use for the request
        :param top_k: Int: Top-k value to use for the request
        :param context_window: Int: Context window length in tokens to use for the request
        :param json_format: Bool: Whether to return the response as JSON
        :return: Tuple: List of responses from the server and the processing time of the request
        """

        if self.client_type != 'ollama' or not isinstance(self.client, AsyncClient):
            raise RuntimeError(f'Trying to use Ollama prompt function with a non-Ollama client: {self.client_type}')

        if examples is not None:
            raise ValueError('Examples are not supported for Ollama queries.')

        if n is not None:
            raise ValueError('Multiple completions not supported for Ollama queries.')

        loop = asyncio.get_event_loop()

        start_time = timeit.default_timer()

        options = {
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': top_p,
            'top_k': top_k,
            'num_ctx': context_window,
        }

        responses = loop.run_until_complete(query_ollama_parallel(
            model,
            user_msgs,
            system_msg,
            self.client,
            options=options,
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

