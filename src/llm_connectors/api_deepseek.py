from typing import Literal

from pydantic import BaseModel

from src.llm_connectors.api_base import BatchStatus
from src.llm_connectors.api_openai import ApiOpenAI

api = 'deepseek'

available_models = {
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


class ApiDeepSeek(ApiOpenAI):
    def __init__(
            self,
            run_id: str,
            hostname: str = "https://api.deepseek.com",
            default_model: str = None,
            models: dict = None,
            api: str = 'openai',
            api_key_name: str = 'OPENAI_API_KEY',
            supports_batch: bool = True
    ):
        if models is None:
            models = available_models

        super().__init__(
            run_id=run_id,
            models=models,
            api=api,
            api_key_name='DEEPSEEK_API_KEY',
            default_model=default_model,
            hostname=hostname,
            supports_batch=False
        )

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
        # This method is intentionally left empty as DeepSeek models are not loaded in the same way as other APIs.
        pass

    def _unload_model(self):
        # This method is intentionally left empty as DeepSeek models are not unloaded in the same way as other APIs.
        pass

    def _parse_response_format(self, response_format, json_schema: dict) -> dict:
        if response_format == 'text' or self.active_model.get('supports_json', False) is False:
            parsed_format = {'type': 'text'}
        elif response_format == 'json':
            parsed_format = {'type': 'json_object'}
        elif response_format == 'json_schema':
            parsed_format = {'type': 'json_object'}
        else:
            raise ValueError(f"response_format '{response_format}' not supported")

        return parsed_format
