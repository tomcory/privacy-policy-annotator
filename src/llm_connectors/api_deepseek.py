import timeit
from typing import Literal, List, Tuple, Optional

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
    def __init__(self, run_id: str, hostname: str = "https://api.deepseek.com", default_model: str = None):
        super().__init__(
            run_id=run_id,
            models=models,
            api=api,
            api_key_name='DEEPSEEK_API_KEY',
            default_model=default_model,
            hostname=hostname,
            supports_batch=True
        )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.hostname
        )

    def close(self):
        print("Closed OpenAI API client.")
        self.client.close()

    def setup_task(self, task: str, model: str):
        super().setup_task(task, model)

    def prompt(
            self,
            pkg: str,
            task: str,
            user_msg: str,
            system_msg: str = None,
            examples: list[tuple[str, str]] = [],
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
    ) -> Tuple[str, float, float]:

        start_time = timeit.default_timer()
        self.setup_task(task, model)

        messages, system_msg, response_schema = util.prepare_prompt_messages(
            api=api,
            task=task,
            user_msg=user_msg,
            system_msg=system_msg,
            examples=examples
        )

        # parse the given response format into the correct format for the API call
        parsed_response_format = _parse_response_format(response_format, response_schema, task,
                                                        self.active_model['supports_json'])

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

        # TODO: Handle reasoning content if needed

        input_len = completion.usage.prompt_tokens
        output_len = completion.usage.completion_tokens

        return self._log_response(start_time, output, input_len, output_len, pkg, task, response_format)

    def prompt_parallel(
            self,
            pkg: str,
            task: str,
            user_msgs: list[str],
            system_msg: str = None,
            examples: list[tuple[str, str]] = [],
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

    def _load_model(self):
        # This method is intentionally left empty as DeepSeek models are not loaded in the same way as other APIs.
        pass

    def _unload_model(self):
        # This method is intentionally left empty as DeepSeek models are not unloaded in the same way as other APIs.
        pass
