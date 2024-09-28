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

TEMPERATURE = 0.5
MAX_TOKENS = 500
TOP_P = 0.9
TOP_K = 40
CONTEXT_WINDOW = 8192


def _process_ollama_response(response, llm_name: str, json_format: bool = True) -> Union[Dict, str]:
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
        print(f'\nError decoding JSON: {e}. Continuing with empty dictionary...')
        logging.error(f'Error decoding JSON for model {llm_name}: {e}. Raw model output: {response["response"]}', exc_info=True)
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
    encoding = tiktoken.get_encoding('gpt2')
    prompt_token_count = len(encoding.encode(prompt))
    system_prompt_token_count = len(encoding.encode(system_prompt))
    total_token_count = prompt_token_count + system_prompt_token_count

    logging.debug(
        f'Querying model {model_code} with:\nUser prompt: {prompt}\nSystem prompt: {system_prompt[:100]}...\nOptions: {options}\n'
        f'Output format: {output_format if output_format else "default"}\n'
        f'User prompt token count: {prompt_token_count}\nSystem prompt token count: {system_prompt_token_count}'
    )
    logging.info(f'Querying model {model_code} with {total_token_count} tokens...')

    options['num_ctx'] = min(total_token_count + 1000, 6144)
    logging.debug(f'Using context window of {options["num_ctx"]} tokens.')

    try:
        response = _process_ollama_response(
            await ollama_client.generate(model=model_code, prompt=prompt, system=system_prompt, format=output_format, options=options, keep_alive=keep_alive),
            llm_name=model_code,
            json_format=output_format == 'json'
        )
        return response
    except Exception as e:
        logging.error(f'Error querying model {model_code}: {e}', exc_info=True)
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


def _prepare_messages(system_msg: str, user_msg: str, examples: list[tuple[str, str]] = None):
    if examples is None:
        examples = []

    # map the examples to the correct json format
    examples = [({"role": "user", "content": e[0]}, {"role": "assistant", "content": e[1]}) for e in examples]

    # generate the messages list for the API call
    messages = [{"role": "system", "content": system_msg}]
    for example in examples:
        messages.extend(example)
    messages.append({"role": "user", "content": user_msg})

    return messages


class BaseApiWrapper:
    """
    Base class for both Ollama and OpenAI clients.
    Provides a unified interface for model prompting.
    """

    def __init__(self, client):
        self.client = client
        self.client_type = None

    def prompt(
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
    ):
        """
        Prompt method to be implemented by subclasses.
        """

        raise NotImplementedError("This method should be implemented by subclasses.")


class OllamaApiWrapper(BaseApiWrapper):
    """
    Wrapper class for Ollama client.
    """

    def __init__(self, client: AsyncClient):
        super().__init__(client)
        self.client_type = 'ollama'

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
    
    def get_model_info(self, model: str) -> dict:
        """
        Retrieve model information from Ollama.
        
        :param model: String: Model name
        :return: Dict: Model information
        """
        if self.client_type != 'ollama' or not isinstance(self.client, AsyncClient):
            raise RuntimeError('get_model_info is only available for Ollama clients.')

        loop = asyncio.get_event_loop()
        model_info = loop.run_until_complete(self.client.show(model))
        return model_info

    def load_model(self, model: str):
        """
        Load the specified model by sending an empty query to the server.
        This method is only callable if the client is an Ollama client.

        :param model: String: Model code to load
        """

        if self.client_type != 'ollama' or not isinstance(self.client, AsyncClient):
            raise RuntimeError('load_model is only available for Ollama clients.')

        logging.debug(f'Loading model {model}...')
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_query_ollama(model, "", "", self.client))

    def unload_model(self, model: str):
        """
        Unload the specified model by sending an empty query to the server with a keep_alive value of 0.
        This method is only callable if the client is an Ollama client.

        :param model: String: Model code to unload
        """

        if self.client_type != 'ollama' or not isinstance(self.client, AsyncClient):
            raise RuntimeError('unload_model is only available for Ollama clients.')

        logging.debug(f'Unloading model {model}...')
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_query_ollama(model, "", "", self.client, keep_alive=0))

    def prompt(
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

        if n is not None and n > 1:
            raise ValueError('Multiple completions not supported for Ollama queries.')

        user_prompt = str(user_msg)
        loop = asyncio.get_event_loop()

        start_time = timeit.default_timer()

        model_info = self.get_model_info(model)
        model_size = model_info['model_info']['general.parameter_count']
        logging.debug(f'Model {model} has {model_size} parameters.')

        if model_size >= 20_000_000_000 and task in ['annotator', 'reviewer', 'fixer']:
            context_window = min(context_window, 6144)
            logging.debug(f'Model {model} has more than 20B parameters, setting context window to {context_window}')

        options = {
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': top_p,
            'top_k': top_k,
            'num_ctx': context_window,
        }

        response = loop.run_until_complete(_query_ollama(
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

    def prompt_parallel(
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

        if n is not None and n > 1:
            raise ValueError('Multiple completions not supported for Ollama queries.')

        loop = asyncio.get_event_loop()

        start_time = timeit.default_timer()

        model_info = self.get_model_info(model)
        model_size = model_info['model_info']['general.parameter_count']
        logging.debug(f'Model {model} has {model_size} parameters.')

        options = {
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': top_p,
            'top_k': top_k,
            'num_ctx': context_window,
        }

        responses = loop.run_until_complete(_query_ollama_parallel(
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


def retrieve_batch_result_entry(run_id: str, task: str, entry_id: str, batch_results_file: str = "batch_results.jsonl"):
    """
    Retrieve a specific entry from the batch results file.

    This function iterates through all lines of the batch-results.jsonl file and returns the one with the matching custom_id as a dictionary.

    :param run_id: The ID of the run.
    :param task: The name of the task.
    :param entry_id: The custom ID of the entry to retrieve.
    :param batch_results_file: The name of the batch results file (default is "batch_results.jsonl").
    :return: A tuple containing the content of the first choice and the cost of the API call, or (None, 0) if an error occurs.
    :raises json.JSONDecodeError: If there is an error decoding a line in the batch results file.
    """

    with open(f"output/{run_id}/batch/{task}/{batch_results_file}", "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"Error decoding line: {line}")
                raise json.JSONDecodeError
            if entry['custom_id'] == entry_id:
                # check for errors in the response
                if entry['error'] is not None:
                    print(f"Error for entry {entry_id}: {entry['error']}")
                    return None, 0
                elif entry['response']['status_code'] != 200:
                    print(f"Bad status code for entry {entry_id}: {entry['response']['status_code']}")
                    return None, 0
                elif entry['response']['body']['choices'] is None or len(entry['response']['body']['choices']) == 0:
                    print(f"No choices for entry {entry_id}")
                    return None, 0
                elif entry['response']['body']['choices'][0]['finish_reason'] != "stop":
                    print(f"Finish reason not 'stop' for entry {entry_id}: {entry['response']['body']['choices'][0]['finish_reason']}")
                    return None, 0
                # return the content of the first choice
                else:
                    prompt_cost = entry['response']['body']['usage']['prompt_tokens'] * OPENAI_MODELS['gpt-4o-mini']['input_price'] / 2
                    completion_cost = entry['response']['body']['usage']['completion_tokens'] * OPENAI_MODELS['gpt-4o-mini']['output_price'] / 2
                    output = entry['response']['body']['choices'][0]['message']['content']

                    return output, prompt_cost + completion_cost

    print(f"Entry {entry_id} not found")
    return None, 0


class OpenAiApiWrapper(BaseApiWrapper):
    """
    Wrapper class for OpenAI client.
    """

    def __init__(self, client: OpenAI):
        super().__init__(client)
        self.client_type = 'openai'

    def prompt(
            self,
            run_id: str,
            pkg: str,
            task: str,
            model: dict,
            system_msg: str,
            user_msg: str,
            examples: Optional[List[Tuple[str, str]]] = None,
            n: int = 1,
            temperature: Optional[float] = None,
            max_tokens: int = 2048,
            top_p: Optional[float] = None,
            top_k: Optional[int] = None,
            context_window: Optional[int] = None,
            json_format: bool = True,
    ) -> Tuple[str, float]:
        """
        Prompt the OpenAI API with the given parameters and return the response and cost.

        :param run_id: The ID of the run.
        :param pkg: The name of the package.
        :param task: The name of the task.
        :param model: A dictionary containing model details such as name and encoding.
        :param system_msg: The system message to send to the API.
        :param user_msg: The user message to send to the API.
        :param examples: A list of example tuples (user message, assistant message) to include in the prompt.
        :param n: The number of completions to generate.
        :param temperature: The temperature to use for the request.
        :param max_tokens: The maximum number of tokens to generate.
        :param top_p: The top-p value to use for the request.
        :param top_k: The top-k value to use for the request.
        :param context_window: The context window length in tokens to use for the request.
        :param json_format: Whether to return the response as JSON.
        :return: A tuple containing the response from the API and the cost of the API call.
        """

        if self.client_type != 'openai' or not isinstance(self.client, OpenAI):
            raise RuntimeError(f'Trying to use OpenAI prompt function with a non-OpenAI client: {self.client_type}')

        encoding = tiktoken.get_encoding(model['encoding'])

        # calculate the length of the input messages
        system_len = len(encoding.encode(system_msg))
        user_len = len(encoding.encode(user_msg))
        example_len = sum(len(encoding.encode(example[0])) + len(encoding.encode(example[1])) for example in examples)

        messages = _prepare_messages(system_msg, user_msg, examples)

        # configure and query GPT
        completion = self.client.chat.completions.create(
            model=model['name'],
            n=n,
            max_tokens=max_tokens,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"} if json_format else None
        )

        # extract the output text from the response message
        output = completion.choices[0].message.content
        output_len = len(encoding.encode(output))

        # calculate the cost of the API call based on the total number of tokens used
        cost = (system_len + user_len + example_len) * model['input_price'] + output_len * model['output_price']

        if run_id is not None and pkg is not None and task is not None:
            # log the cost and response
            with open(f"output/{run_id}/gpt-responses/costs-{task}.csv", "a") as f:
                f.write(f"{pkg},{cost}\n")
            with open(f"output/{run_id}/gpt-responses/{task}/{pkg}.txt", "w") as f:
                f.write(output)

        return output, cost

    def run_batch(self, run_id: str, task: str):
        """
        Run a batch process using the OpenAI API.

        This function creates a batch input file and submits it to the OpenAI API for processing.
        It then retrieves the batch metadata and writes it to a file.

        :param run_id: str: The ID of the run.
        :param task: str: The name of the task.
        :raises RuntimeError: If the client is not an OpenAI client.
        """

        if self.client_type != 'openai' or not isinstance(self.client, OpenAI):
            raise RuntimeError(f'Trying to use OpenAI batch function with a non-OpenAI client: {self.client_type}')

        batch_input_file = self.client.files.create(
            file=open(f"output/{run_id}/batch/{task}/batch_input.jsonl", "rb"),
            purpose="batch"
        )

        batch_metadata = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "fixing headlines"
            }
        )

        batch_metadata_json = json.dumps(json.loads(batch_metadata.model_dump_json()), indent=4)

        # write the batch metadata to a file
        with open(f"output/{run_id}/batch/{task}/batch_metadata.json", "w") as f:
            f.write(batch_metadata_json)

    def check_batch_status(self, run_id: str, task: str, batch_metadata_file: str = "batch_metadata.json"):
        """
        Check the status of a batch process using the OpenAI API.

        This function reads the batch metadata from a file, retrieves the batch status from the OpenAI API,
        writes the batch status to the same file, and prints the status.

        :param run_id: str: The ID of the run.
        :param task: str: The name of the task.
        :param batch_metadata_file: str: The name of the file containing batch metadata.
        :return: The batch status object.
        """

        with open(f"output/{run_id}/batch/{task}/{batch_metadata_file}", "r") as f:
            batch_metadata = json.load(f)

        batch_status = self.client.batches.retrieve(batch_metadata['id'])

        batch_status_json = json.dumps(json.loads(batch_status.model_dump_json()), indent=4)

        # write the batch status to a file
        with open(f"output/{run_id}/batch/{task}/{batch_metadata_file}", "w") as f:
            f.write(f"{batch_status_json}")

        if batch_status.status == "completed":
            print("Batch completed, getting results...")
        else:
            print("Batch not completed yet, status: %s" % batch_status.status)

        return batch_status

    def get_batch_results(self, run_id: str, task: str, batch_metadata_file: str = "batch_metadata.json"):
        """
        Retrieve the results of a batch process using the OpenAI API.

        This function reads the batch metadata from a file, retrieves the batch results from the OpenAI API,
        writes the batch results to a file, and returns the results as a string.

        :param run_id: str: The ID of the run.
        :param task: str: The name of the task.
        :param batch_metadata_file: str: The name of the file containing batch metadata.
        :return: str: The batch results as a string.
        """

        with open(f"output/{run_id}/batch/{task}/{batch_metadata_file}", "r") as f:
            batch_metadata = json.load(f)

        batch_results = self.client.files.content(batch_metadata['output_file_id']).text

        with open(f"output/{run_id}/batch/{task}/batch_results.jsonl", "w") as f:
            f.write(batch_results)

        return batch_results


class ApiWrapper:
    """
    A unified interface that creates the appropriate wrapper (Ollama or OpenAI) based on the client type.
    """

    def __init__(self, client: Union[AsyncClient, OpenAI]):
        """
        Constructor that creates the appropriate wrapper.
        """
        if isinstance(client, AsyncClient):
            self.wrapper = OllamaApiWrapper(client)
        elif isinstance(client, OpenAI):
            self.wrapper = OpenAiApiWrapper(client)
        else:
            raise ValueError("Invalid client type. Must be either an AsyncClient (Ollama) or OpenAI client.")

    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped object.
        This ensures we can access all properties and methods of the actual wrapper class.
        """
        return getattr(self.wrapper, name)
