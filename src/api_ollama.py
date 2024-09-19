import asyncio
import json
import os
import timeit
from typing import List, Union, Dict, Optional, Literal, Tuple

import ollama
import tiktoken
from ollama import AsyncClient, Options

models = {
    'llama3': {
        'name': 'llama3:instruct',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'llama8b': {
        'name': 'llama3.1:8b-instruct-q4_0',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'llama8b-fp16': {
        'name': 'llama3.1:8b-instruct-fp16',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'llama70b': {
        'name': 'llama3.1:70b-instruct-q3_K_S',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'gemma9b': {
        'name': 'gemma2:9b-instruct-q4_0',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'gemma27b': {
        'name': 'gemma2:27b-instruct-q4_0',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'mistral7b': {
        'name': 'mistral:instruct',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'mistral-nemo12b': {
        'name': 'mistral-nemo:12b-instruct-2407-q4_0',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'mixtral8x7b': {
        'name': 'mixtral:instruct',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'phi3b': {
        'name': 'phi3:3.8b-mini-128k-instruct-q4_0',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'phi3b-fp16': {
        'name': 'phi3:3.8b-mini-128k-instruct-fp16',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'phi14b': {
        'name': 'phi3:14b-medium-128k-instruct-q4_0',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'phi14b-fp16': {
        'name': 'phi3:14b-medium-128k-instruct-fp16',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'qwen7b': {
        'name': 'qwen2:7b-instruct',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'qwen7b-fp16': {
        'name': 'qwen2:7b-instruct-fp16',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'qwen72b': {
        'name': 'qwen2:72b-instruct',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    }
}


def _process_ollama_response(response, llm_name: str, json_format: bool = True) -> Union[Dict, str]:
    """
    Process the response from the ollama server.

    :param response: Response from the server
    :param llm_name: String: Name of the model
    :param json_format: Bool: Whether to return the response as JSON
    :return: Parsed JSON response or empty dictionary if parsing fails
    """

    try:
        print(f'Ouput from model {llm_name}: {response["response"].strip()}')
        if json_format:
            return json.loads(response['response'])
        else:
            return response['response']
    except (json.JSONDecodeError, KeyError) as e:
        print(f'\nError decoding JSON: {e}. Continuing with empty dictionary...')
        print(f'Error decoding JSON for model {llm_name}: {e}. Raw model output: {response["response"]}')
        return {}


async def _send_ollama_request(
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
    encoding = tiktoken.get_encoding('o200k_base')
    prompt_token_count = len(encoding.encode(prompt))
    system_prompt_token_count = len(encoding.encode(system_prompt))

    print(
        f'Querying model {model_code} with:\nUser prompt: {prompt}\nSystem prompt: {system_prompt[:100]}...\nOptions: {options}\n'
        f'Output format: {output_format if output_format else "default"}\n'
        f'User prompt token count: {prompt_token_count}\nSystem prompt token count: {system_prompt_token_count}'
    )
    print(f'Querying model {model_code} with {prompt_token_count + system_prompt_token_count} tokens...')

    try:
        response = _process_ollama_response(
            await ollama_client.generate(model=model_code, prompt=prompt, system=system_prompt, format=output_format, options=options, keep_alive=keep_alive),
            llm_name=model_code,
            json_format=output_format == 'json'
        )
        return response
    except Exception as e:
        print(f'Error querying model {model_code}: {e}')
        return {}


async def _query_ollama(
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

    if options is None:
        options = {}
    request_options = Options(**options)

    return await _send_ollama_request(model_code, user_prompt, system_prompt, ollama_client, request_options, output_format, keep_alive)


async def _query_ollama_parallel(
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

    if options is None:
        options = {}
    request_options = Options(**options)

    async def worker(task_queue: asyncio.Queue, result_list: List[Union[Dict, str]]):
        while True:
            task_prompt = await task_queue.get()
            if task_prompt is None:
                break
            result = await _send_ollama_request(model_code, task_prompt, system_prompt, ollama_client, request_options, output_format, keep_alive)
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


class ApiOllama:
    def __init__(self, run_id: str, default_model: str = None):
        self.run_id = run_id

        if default_model is not None:
            self.default_model = models[default_model]
        else:
            self.default_model = None

        self.supports_batch = False
        self.supports_parallel = True

        self.downloaded_models = [model['name'] for model in ollama.list()['models']]

        self.client = AsyncClient()

    def setup(self):
        # check whether the default model is downloaded and download it if necessary
        if self.default_model['name'] not in self.downloaded_models:
            self._download_model(self.default_model['name'])

        self._load_model(self.default_model['name'])
        print(f'Set up Ollama API client with model {self.default_model["name"]}.')

    def close(self):
        self._unload_model(self.default_model['name'])
        print(f'Closed Ollama API client with model {self.default_model["name"]}.')

    def prompt(
            self,
            pkg: str,
            task: str,
            system_msg: str,
            user_msg: str,
            examples: list[tuple[str, str]] = None,
            model: dict = None,
            response_format: Literal['text', 'json', 'json_schema'] = 'text',
            json_schema: dict = None,
            temperature: float = 1.0,
            max_tokens: int = 2048,
            n: int = 1,
            top_p: int = 1,
            top_k: int = None,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            seed: int = None,
            context_window: int = None,
            timeout: float = None
    ) -> (str, float, float):

        if model is None:
            if self.default_model is None:
                raise ValueError("model must be provided when default_model is not set")
            model = self.default_model

        if model not in self.downloaded_models:
            self._download_model(model['name'])

        encoding = tiktoken.get_encoding(model['encoding'])

        # calculate the length of the input messages
        system_len = len(encoding.encode(system_msg))
        user_len = len(encoding.encode(user_msg))
        example_len = sum(len(encoding.encode(example[0])) + len(encoding.encode(example[1])) for example in examples)

        if example_len > 0:
            print(f'Warning: Ollama does not support examples. Ignoring {example_len} tokens.')
            example_len = 0

        loop = asyncio.get_event_loop()

        start_time = timeit.default_timer()

        options = {
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': top_p,
            'top_k': top_k,
            'num_ctx': context_window,
        }

        output = loop.run_until_complete(_query_ollama(
            model['name'],
            user_msg,
            system_msg,
            self.client,
            options=options,
            output_format='json' if response_format == 'json' else ''
        ))
        output_len = len(encoding.encode(output))

        end_time = timeit.default_timer()
        processing_time = end_time - start_time

        # calculate the cost of the API call based on the total number of tokens used
        cost = (system_len + user_len + example_len) * model['input_price'] + output_len * model['output_price']

        if self.run_id is not None and pkg is not None and task is not None:
            # log the cost, processing time and response
            with open(f"output/{self.run_id}/{model['name']}_responses/costs_{task}.csv", "a") as f:
                f.write(f"{pkg},{cost}\n")
            with open(f"output/{self.run_id}/{model['name']}_responses/times_{task}.csv", "a") as f:
                f.write(f"{pkg},{processing_time}\n")
            with open(f"output/{self.run_id}/{model['name']}_responses/{task}/{pkg}.txt", "w") as f:
                f.write(output)

        return output, cost, processing_time

    def prompt_parallel(
            self,
            pkg: str,
            task: str,
            system_msg: str,
            user_msgs: list[str],
            examples: list[tuple[str, str]] = [],
            model: dict = None,
            response_format: Literal['text', 'json', 'json_schema'] = 'text',
            json_schema: dict = None,
            temperature: float = 1.0,
            max_tokens: int = 2048,
            n: int = 1,
            top_p: int = 1,
            top_k: int = None,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            seed: int = None,
            context_window: int = None,
            timeout: float = None
    ):

        if model is None:
            if self.default_model is None:
                raise ValueError("model must be provided when default_model is not set")
            model = self.default_model

        encoding = tiktoken.get_encoding(model['encoding'])

        # calculate the length of the input messages
        system_len = len(encoding.encode(system_msg))
        user_len = sum(len(encoding.encode(user_msg)) for user_msg in user_msgs)
        example_len = sum(len(encoding.encode(example[0])) + len(encoding.encode(example[1])) for example in examples)

        if example_len > 0:
            print(f'Warning: Ollama does not support examples. Ignoring {example_len} tokens.')
            example_len = 0

        loop = asyncio.get_event_loop()

        start_time = timeit.default_timer()

        options = {
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': top_p,
            'top_k': top_k,
            'num_ctx': context_window,
        }

        outputs = loop.run_until_complete(_query_ollama_parallel(
            model['name'],
            user_msgs,
            system_msg,
            self.client,
            options=options,
            output_format='json' if response_format == 'json' else ''
        ))

        end_time = timeit.default_timer()
        processing_time = end_time - start_time

        # calculate the cost of the API call based on the total number of tokens used
        output_len = sum(len(encoding.encode(output)) for output in outputs)
        cost = (system_len + user_len + example_len) * model['input_price'] + output_len * model['output_price']

        output_format = 'json' if response_format == 'json' else 'txt'

        if self.run_id is not None and pkg is not None and task is not None:
            # create the output folder if it does not exist
            folder_path = f"output/{self.run_id}/{model['name']}_responses"
            os.makedirs(folder_path, exist_ok=True)
            os.makedirs(f"{folder_path}/{task}", exist_ok=True)

            # log the cost, processing time and response
            with open(f"output/{folder_path}/costs_{task}.csv", "a") as f:
                f.write(f"{pkg},{cost}\n")
            with open(f"output/{folder_path}/times_{task}.csv", "a") as f:
                f.write(f"{pkg},{processing_time}\n")
            for i, output in enumerate(outputs):
                with open(f"output/{folder_path}/{task}/{pkg}_{i}.{output_format}", "a") as f:
                    f.write(output + '\n')

        return outputs, cost, processing_time

    def prepare_batch_entry(
            self,
            pkg: str,
            task: str,
            system_msg: str,
            user_msg: str,
            examples: list[tuple[str, str]] = None,
            entry_id: int = 0,
            model: dict = None,
            response_format: str = 'text',
            temperature: float = 1.0,
            max_tokens: int = 2048,
            n: int = 1,
            top_p: int = 1,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            seed: int = None,
            context_window: int = None,
            timeout: int = -1
    ):
        raise NotImplementedError("Batch processing is not supported by Ollama")

    def run_batch(self, task: str):
        raise NotImplementedError("Batch processing is not supported by Ollama")

    def retrieve_batch_result_entry(self, task: str, entry_id: str, batch_results_file: str = "batch_results.jsonl"):
        raise NotImplementedError("Batch processing is not supported by Ollama")

    def check_batch_status(self, task: str, batch_metadata_file: str = "batch_metadata.json"):
        raise NotImplementedError("Batch processing is not supported by Ollama")

    def get_batch_results(self, task: str, batch_metadata_file: str = "batch_metadata.json"):
        raise NotImplementedError("Batch processing is not supported by Ollama")

    def _download_model(self, model: str):
        """
        Download the specified model by sending an empty query to the server.

        :param model: String: Model code to download
        """

        print(f'Downloading model {model}...')
        try:
            ollama.pull(model)
            print(f'Downloaded model {model}')
        except Exception as e:
            print(f'Error downloading model {model}: {e}')
            raise e

    def _load_model(self, model: str):
        """
        Load the specified model by sending an empty query to the server.

        :param model: String: Model code to load
        """

        print(f'Loading model {model}...')
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_query_ollama(model, "", "", self.client))

    def _unload_model(self, model: str):
        """
        Unload the specified model by sending an empty query to the server with a keep_alive value of 0.

        :param model: String: Model code to unload
        """

        print(f'Unloading model {model}...')
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_query_ollama(model, "", "", self.client, keep_alive=0))
