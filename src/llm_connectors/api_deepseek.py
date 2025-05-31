import os
import sys
import timeit
from typing import Literal, List, Tuple, Optional

import tiktoken
from openai import OpenAI

from src import util
from src.llm_connectors.api_base import ApiBase, BatchStatus

api = 'deepseek'

models = {
    'deepseek-r1': {
        'name': 'deepseek-reasoner',
        'api': api,
        'supports_json': False,
        'encoding': 'cl100k_base',
        'input_price': (0.135 / 1000000),
        'output_price': (0.55 / 1000000),
        'input_price_batch': (0.25 / 1000000),
        'output_price_batch': (0.75 / 1000000)
    },
    'deepseek-v3': {
        'name': 'deepseek-chat',
        'api': api,
        'supports_json': True,
        'encoding': 'cl100k_base',
        'input_price': (0.135 / 1000000),
        'output_price': (0.55 / 1000000),
        'input_price_batch': (15 / 1000000),
        'output_price_batch': (30 / 1000000)
    }
}


def _parse_response_format(response_format, json_schema: dict, task: str, supports_json: bool) -> dict:
    if response_format == 'text' or supports_json is False:
        parsed_format = {'type': 'text'}
    elif response_format == 'json':
        parsed_format = {'type': 'json_object'}
    elif response_format == 'json_schema':
        parsed_format = {'type': 'json_object'}
    else:
        raise ValueError(f"response_format '{response_format}' not supported")

    return parsed_format


class ApiDeepSeek(ApiBase):
    def __init__(self, run_id: str, default_model: str = None):
        super().__init__(run_id, models, default_model, None, True, False)
        self.active_model = None

        api_key = os.getenv('DEEPSEEK_API_KEY')
        if api_key is None:
            print("Please set the DEEPSEEK_API_KEY environment variable.")
            sys.exit(1)

        self.client = OpenAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )

    def setup(self, task: str, model: dict, use_custom_model: bool = False):
        super().setup(task, model, use_custom_model)

    def close(self):
        print("Closed OpenAI API client.")
        self.client.close()

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

        if self.active_model is None:
            self.setup(task, model, False)

        # load the messages from the prompts folder or use the provided messages
        messages, system_msg, response_schema, system_len, user_len, example_len  = util.prepare_prompt_messages(api, task, user_msg, system_msg, examples)

        # parse the given response format into the correct format for the API call
        parsed_response_format = _parse_response_format(response_format, response_schema, task, self.active_model['supports_json'])

        # start the timer to measure the processing time
        start_time = timeit.default_timer()

        # configure and query GPT
        completion = self.client.chat.completions.create(
            messages=messages,
            model=self.active_model['name'],
            response_format=parsed_response_format,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            n=n,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            timeout=timeout
        )

        try:
            reasoning = completion.choices[0].message.reasoning_content
        except AttributeError:
            reasoning = None
        output = completion.choices[0].message.content

        print(f"Reasoning:\n{reasoning}\n")
        print("x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x")
        print(f"Output:\n{output}")

        # extract the output text from the response message
        output_len = len(tiktoken.get_encoding(model['encoding']).encode(output))

        # stop the timer and calculate the processing time
        end_time = timeit.default_timer()
        processing_time = end_time - start_time

        # calculate the cost of the API call based on the total number of tokens used
        cost = (system_len + user_len + example_len) * self.active_model['input_price'] + output_len * self.active_model['output_price']

        # log the prompt and result
        output_format = 'txt' if response_format == 'text' else 'json'
        if self.run_id is not None and pkg is not None and task is not None:
            print(f"Logging prompt and result for {pkg}...")
            util.log_prompt_result(self.run_id, task, pkg, self.active_model['name'], output_format, cost, processing_time, [output])

        return output, cost, processing_time

    def prompt_parallel(
            self,
            pkg: str,
            task: str,
            user_msgs: list[str],
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
    ) -> Tuple[List[str], float, float]:
        raise NotImplementedError("Parallel requests are not supported by the DeepSeek API")

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
        raise NotImplementedError("Batch processing is not supported by the DeepSeek API")

    def run_batch(self, task: str):
        raise NotImplementedError("Batch processing is not supported by the DeepSeek API")

    def retrieve_batch_result_entry(self, task: str, entry_id: str, batch_results_file: str = "batch_results.jsonl"):
        raise NotImplementedError("Batch processing is not supported by the DeepSeek API")

    def check_batch_status(self, task: str, batch_metadata_file: str = "batch_metadata.json") -> Optional[BatchStatus]:
        raise NotImplementedError("Batch processing is not supported by the DeepSeek API")

    def get_batch_results(self, task: str, batch_metadata_file: str = "batch_metadata.json"):
        raise NotImplementedError("Batch processing is not supported by the DeepSeek API")