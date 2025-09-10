import asyncio
import timeit
from typing import Literal, List, Tuple, Optional

import ollama
from ollama import AsyncClient
from pydantic import BaseModel
from tqdm import tqdm

from src import util
from src.llm_connectors.api_base import ApiBase, BatchStatus
from src.llm_connectors.api_openai import available_models

api = 'ollama'

available_models = {
    'deepseek-r1:7b': {
        'name': 'deepseek-r1:7b',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'deepseek-r1:70b': {
        'name': 'deepseek-r1:70b',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'deepseek-r1:671b': {
        'name': 'deepseek-r1:671b',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'gemma2:27b': {
        'name': 'gemma2:27b',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'gemma3:27b': {
        'name': 'gemma3:27b',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'gemma3n:e4b': {
        'name': 'gemma3n:e4b',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'llama3.3:70b': {
        'name': 'llama3.3:70b',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'llama4:16x17b': {
        'name': 'llama3.3:70b',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'llama3.2': {
        'name': 'llama3.2:latest',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'mistral-large:123b': {
        'name': 'mistral-large:123b',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'mixtral:8x22b': {
        'name': 'mixtral:8x22b',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'phi4:14b': {
        'name': 'phi4:14b',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'qwen2.5:72b': {
        'name': 'qwen2.5:72b',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'qwen3:32b': {
        'name': 'qwen3:32b',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    },
    'qwen3:4b': {
        'name': 'qwen3:4b',
        'api': 'ollama',
        'encoding': 'o200k_base',
        'input_price': 0,
        'output_price': 0,
        'input_price_batch': 0,
        'output_price_batch': 0
    }
}


class ApiOllama(ApiBase):
    """
    Ollama API connector for interacting with LLMs hosted on Ollama.
    """

    def __init__(
            self,
            run_id: str,
            hostname: str = None,
            default_model: str = None,
            use_opp_115: bool = False
    ):
        super().__init__(
            run_id=run_id,
            models=available_models,
            api='ollama',
            api_key_name='OLLAMA_API_KEY',
            default_model=default_model,
            hostname=hostname,
            supports_batch=False,
            supports_parallel=False,
            use_opp_115=use_opp_115
        )

        # Configure headers with Bearer token if API key is available
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'

        if hostname is not None:
            self.client = ollama.Client(host=hostname, headers=headers)
            self.async_client = AsyncClient(host=hostname, headers=headers)
        else:
            self.client = ollama.Client(headers=headers)
            self.async_client = AsyncClient(headers=headers)

        self.downloaded_models = [model['model'] for model in self.client.list()['models']]

    def close(self):
        self._unload_model()

    def prompt(
            self,
            pkg: str,
            task: str,
            user_msg: str,
            system_msg: str = None,
            examples: list[tuple[str, str]] = None,
            model: str = None,
            response_format: Literal['text', 'json', 'json_schema'] = 'text',
            json_schema: dict = None,
            temperature: float = 0.5,
            max_tokens: int = 4096,
            n: int = 1,
            top_p: int = 1,
            top_k: int = None,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0,
            seed: int = None,
            context_window: int = None,
            timeout: float = None
    ) -> tuple[str, float, float]:

        if examples is None:
            examples = []

        start_time = timeit.default_timer()
        self.setup_task(task, model)

        messages, system_msg, response_schema = util.prepare_prompt_messages(
            api=api,
            task=task,
            user_msg=user_msg,
            system_msg=system_msg,
            examples=examples,
            use_opp_115=self.use_opp_115
        )

        response = self.client.chat(
            model=self.active_model,
            messages=messages,
            format=response_schema
        )

        output = response.message.content
        input_len = response.prompt_eval_count
        output_len = response.eval_count

        return self._log_response(start_time, output, input_len, output_len, pkg, task, response_format)

    def prompt_parallel(
            self,
            pkg: str,
            task: str,
            user_msgs: list[str],
            system_msg: str = None,
            examples: list[tuple[str, str]] = None,
            model: str = None,
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
    ) -> tuple[list[str], float, float]:

        if examples is None:
            examples = []

        start_time = timeit.default_timer()
        self.setup_task(task, model)

        # load the messages from the prompts folder or use the provided messages
        # print("Loading messages from prompts folder...")
        examples, system_msg, response_schema = util.prepare_prompt_messages(
            api=api,
            task=task,
            system_msg=system_msg,
            examples=examples,
            use_opp_115=self.use_opp_115
        )
        print(f'Processing {len(user_msgs)} prompts in parallel...')

        # self.async_client.chat(model['name'], messages=examples, format=response_schema)

        # create a results list the size of the user_msgs list
        results = [None] * len(user_msgs)
        total_cost = 0
        total_time = 0

        # get the number of workers to use
        max_workers = min(len(user_msgs), 16)

        message_lists = [examples] * len(user_msgs)
        for i, user_msg in enumerate(user_msgs):
            message_lists[i].append({'role': 'user', 'content': user_msg})

        print(f'Processing {len(message_lists)} message lists in parallel...')

        responses = asyncio.get_event_loop().run_until_complete(
            self._prompt_parallel_async(
                message_lists,
                response_schema,
                self.active_model
            )
        )

        processing_time = timeit.default_timer() - start_time

        return responses, total_cost, processing_time

    def check_batch_status(
            self,
            task: str,
            batch_metadata_file: str = "batch_metadata.json"
    ) -> BatchStatus | None:
        raise NotImplementedError("Batch processing is not supported")

    def prepare_batch_entry(
            self,
            pkg: str,
            task: str,
            user_msg: str,
            system_msg: str = None,
            examples: list[tuple[str, str]] = None,
            entry_id: int = 0,
            model: str = None,
            response_format: Literal['text', 'json', 'json_schema'] | BaseModel = 'text',
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
            timeout: int = -1,
            batch_input_file: str = "batch_input.jsonl"
    ):
        raise NotImplementedError("Batch processing is not supported")

    def run_batch(
            self,
            task: str,
            batch_input_file: str = "batch_input.jsonl",
            batch_metadata_file: str = "batch_metadata.json"
    ):
        raise NotImplementedError("Batch processing is not supported")

    def retrieve_batch_result_entry(
            self,
            task: str,
            entry_id: str,
            batch_results_file: str = "batch_results.jsonl",
            valid_stop_reasons: list[str] = None
    ) -> tuple[str | None, float, float]:
        raise NotImplementedError("Batch processing is not supported")

    def get_batch_results(
            self,
            task: str,
            batch_metadata_file: str = "batch_metadata.json",
            batch_results_file: str = "batch_results.jsonl",
            batch_errors_file: str = "batch_errors.jsonl"
    ) -> str | None:
        raise NotImplementedError("Batch processing is not supported")

    def _load_model(self):

        # download the model if necessary
        self.downloaded_models = [model['model'] for model in self.client.list()['models']]
        if self.active_model not in self.downloaded_models:
            print(f'Model downloading model {self.active_model}...')
            self._pull_model(self.active_model)

        #print(f'Initialising model {self.active_model}...')

        # initialize the model by prompting it with a message
        self.client.chat(self.active_model, messages=[{'role': 'user', 'content': 'What model are you?'}])

        #print(f'Loaded model {self.active_model}.')

    def _unload_model(self):
        if self.active_model is not None:
            print(f'Unloading model {self.active_model}...')
            try:
                self.client.chat(self.active_model, [{'role': 'user', 'content': ' '}], keep_alive=0)
                print(f'Unloaded model {self.active_model}.')
            except ollama.ResponseError as e:
                print(f'Failed to unload model {self.active_model}.')
                print(e)
            self.active_model = None

    def _pull_model(self, model_name: str):
        current_digest, bars = '', {}
        for progress in self.client.pull(model_name, stream=True):
            digest = progress.get('digest', '')
            if digest != current_digest and current_digest in bars:
                bars[current_digest].close()

            if not digest:
                print(f'> {progress.get("status")}...')
                continue

            if digest not in bars and (total := progress.get('total')):
                bars[digest] = tqdm(total=total, desc=f'> pulling {digest[7:19]}', unit='B', unit_scale=True)

            if completed := progress.get('completed'):
                bars[digest].update(completed - bars[digest].n)

            current_digest = digest

    async def _worker(
            self,
            id: int,
            task_queue: asyncio.Queue,
            result_list: list[(str, float, float)],
            response_schema: dict,
            model: dict
    ):
        print(f'Worker active...')
        while True:
            msgs = await task_queue.get()
            print(f'Worker {id} processing...')
            if msgs is None:
                print(f"Task queue is empty. Exiting worker {id}.")
                break
            result = await self._prompt_async(msgs, response_schema, model)
            result_list.append(result)
            task_queue.task_done()

    async def _prompt_parallel_async(
            self,
            messages_list: list[list[dict[str, str]]],
            response_schema: dict,
            model: dict
    ) -> list[(str, float, float)]:

        task_queue = asyncio.Queue()
        for prompt in messages_list:
            task_queue.put_nowait(prompt)

        max_concurrent_requests = 16

        result_list = []
        workers = [asyncio.create_task(self._worker(
            i,
            task_queue,
            result_list,
            response_schema,
            model
        )) for i in range(max_concurrent_requests)]

        await task_queue.join()

        for _ in range(max_concurrent_requests):
            task_queue.put_nowait(None)

        await asyncio.gather(*workers)

        return result_list

    async def _prompt_async(
            self,
            msgs: list[dict[str, str]],
            response_schema: dict,
            model: dict
    ) -> (str, float, float):
        try:
            response = await self.async_client.chat(model['name'], messages=msgs, format=response_schema)
            return response['message']['content'], 0, 0
        except Exception as e:
            print(f'Failed to chat with model {model["name"]}.')
            print(e)
            return None, 0, 0
