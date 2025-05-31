import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, List, Tuple, Optional

import ollama
from ollama import AsyncClient
from tqdm import tqdm

from src import util
from src.llm_connectors.api_base import ApiBase, BatchStatus

api = 'ollama'

models = {
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
    'llama3.3:70b': {
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
}


class ApiOllama(ApiBase):
    def __init__(self, run_id: str, hostname: str = None, default_model: str = None):
        super().__init__(run_id, models, default_model, None, hostname, True, False)

        if hostname is not None:
            self.client = ollama.Client(hostname)
            self.async_client = AsyncClient(hostname)
        else:
            self.client = ollama.Client()
            self.async_client = AsyncClient()

        self.downloaded_models = [model['model'] for model in self.client.list()['models']]

    def setup(self, task: str, model: dict, use_custom_model: bool = False):
        print(f'Setting up Ollama API client for task {task} and model {model["name"]}...')
        self.task = task

        if use_custom_model:
            model_name = f'{model["name"]}-{task}'
            self.use_custom_model = True
            print(f'Using custom model {model_name}.')
        else:
            model_name = model['name']

        if self.active_model['name'] is not model_name:
            print(f'{model_name} is not the loaded model.')

            # unload the current model to free up resources before loading the new model
            # self._unload_model()

            # check whether the default model is downloaded and download it if necessary
            self.downloaded_models = [model['model'] for model in self.client.list()['models']]
            if model_name not in self.downloaded_models:
                if use_custom_model:
                    print(f'Custom model {model_name} does not exist. Creating...')
                    self._create_model(task, model, model_name)
                else:
                    print(f'Model {model_name} is not downloaded. Downloading...')
                    self._pull_model(model['name'])

            print(f'Initialising model {model_name}...')

            # initialize the model by prompting it with a message
            try:
                response = self.client.chat(model_name, messages=[{ 'role': 'user', 'content': 'What model are you?' }])
                print(f'> {response["message"]["content"]}')
                self.active_model = self.models[model_name]
                print(f'Loaded model {model_name}.')
            except ollama.ResponseError as e:
                print(f'Failed to load model {model_name}.')
                print(e)
                self.active_model = None

    def close(self):
        self._unload_model()
        print(f'Closed Ollama API client.')

    def prompt(
            self,
            pkg: str,
            task: str,
            user_msg: str,
            system_msg: str = None,
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
    ) -> Tuple[str, float, float]:
        if model is None:
            if self.default_model is None:
                raise ValueError("model must be provided when default_model is not set")
            model = self.default_model

        # load the messages from the prompts folder or use the provided messages
        # print("Loading messages from prompts folder...")
        messages, system_msg, response_schema, _, _, _ = util.prepare_prompt_messages(
            api=api,
            task=task,
            user_msg=user_msg,
            system_msg=system_msg,
            examples=examples,
            schema_only=self.use_custom_model
        )

        try:
            if self.use_custom_model:
                response = self.client.chat(
                    model=model['name'],
                    messages=[{'role': 'user', 'content': user_msg}],
                    format=response_schema
                )
            else:
                response = self.client.chat(
                    model=model['name'],
                    messages=messages,
                    format=response_schema
                )
            return response['message']['content'], 0, 0
        except ollama.ResponseError as e:
            print(f'Failed to chat with model {model["name"]}.')
            print(e)
            return None, 0, 0


    def prompt_parallel(
            self,
            pkg: str,
            task: str,
            user_msgs: list[str],
            system_msg: str = None,
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
    ) -> Tuple[List[str], float, float]:
        if model is None:
            if self.default_model is None:
                raise ValueError("model must be provided when default_model is not set")
            model = self.default_model

        # load the messages from the prompts folder or use the provided messages
        # print("Loading messages from prompts folder...")
        examples, system_msg, response_schema, _, _, _ = util.prepare_prompt_messages(
            api=api,
            task=task,
            system_msg=system_msg,
            examples=examples,
            schema_only=self.use_custom_model
        )
        print(f'Processing {len(user_msgs)} prompts in parallel...')

        # self.async_client.chat(model['name'], messages=examples, format=response_schema)

        # create a results list the size of the user_msgs list
        results = [None] * len(user_msgs)
        total_cost = 0
        total_time = 0

        # get the number of workers to use
        max_workers = min(len(user_msgs), 16)

        if self.use_custom_model:
            print(f'Using custom model {model["name"]}.')
            message_lists = [[{ 'role': 'user', 'content': user_msg }] for user_msg in user_msgs]
        else:
            print(f'Using model {model["name"]}.')
            message_lists = [examples] * len(user_msgs)
            for i, user_msg in enumerate(user_msgs):
                message_lists[i].append({ 'role': 'user', 'content': user_msg })

        print(f'Processing {len(message_lists)} message lists in parallel...')

        responses = asyncio.get_event_loop().run_until_complete(
            self._prompt_parallel_async(
                message_lists,
                response_schema,
                model
            )
        )

        return responses, total_cost, total_time

    def prepare_batch_entry(
            self,
            pkg: str,
            task: str,
            user_msg: str,
            system_msg: str = None,
            examples: list[tuple[str, str]] = [],
            entry_id: int = 0,
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
            timeout: int = -1
    ):
        raise NotImplementedError("Batch processing is not supported by Ollama")

    def run_batch(self, task: str):
        raise NotImplementedError("Batch processing is not supported by Ollama")

    def retrieve_batch_result_entry(self, task: str, entry_id: str, batch_results_file: str = "batch_results.jsonl"):
        raise NotImplementedError("Batch processing is not supported by Ollama")

    def check_batch_status(self, task: str, batch_metadata_file: str = "batch_metadata.json") -> Optional[BatchStatus]:
        raise NotImplementedError("Batch processing is not supported by Ollama")

    def get_batch_results(self, task: str, batch_metadata_file: str = "batch_metadata.json"):
        raise NotImplementedError("Batch processing is not supported by Ollama")

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

    def _create_model(self, task: str, model: dict, created_model_name: str):
        examples, system_msg, response_schema, _, _, _ = util.prepare_prompt_messages(
            "ollama",
            task,
            bundle_system_msg=False,
            schema_only=False
        )

        print(f'Creating custom model {created_model_name} from model {model["name"]}...')

        self.client.create(
            model=f"{created_model_name}",
            from_=model['name'],
            system=system_msg,
            messages=examples
        )

        print(f'Done.')

    def _unload_model(self):
        if self.active_model is not None:
            print(f'Unloading model {self.active_model['name']}...')
            try:
                self.client.chat(self.active_model['name'], [{ 'role': 'user', 'content': ' ' }], keep_alive=0)
                print(f'Unloaded model {self.active_model['name']}.')
            except ollama.ResponseError as e:
                print(f'Failed to unload model {self.active_model['name']}.')
                print(e)
            self.active_model = None

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