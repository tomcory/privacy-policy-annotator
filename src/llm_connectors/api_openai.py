import json
import sys
import timeit
from datetime import datetime
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from src import util
from src.llm_connectors.api_base import ApiBase, BatchStatus

available_models = {
    'gpt-3.5-turbo': {
        'name': 'gpt-3.5-turbo',
        'api': 'openai',
        'encoding': 'cl100k_base',
        'input_price': (0.5 / 1000000),
        'input_price_cached': (0.5 / 1000000),
        'output_price': (1.5 / 1000000),
        'input_price_batch': (0.25 / 1000000),
        'output_price_batch': (0.75 / 1000000)
    },
    'gpt-4': {
        'name': 'gpt-4',
        'api': 'openai',
        'encoding': 'cl100k_base',
        'input_price': (30 / 1000000),
        'input_price_cached': (0.5 / 1000000),
        'output_price': (60 / 1000000),
        'input_price_batch': (15 / 1000000),
        'output_price_batch': (30 / 1000000)
    },
    'gpt-4o': {
        'name': 'gpt-4o-2024-11-20',
        'api': 'openai',
        'encoding': 'o200k_base',
        'input_price': (2.5 / 1000000),
        'input_price_cached': (0.5 / 1000000),
        'output_price': (10 / 1000000),
        'input_price_batch': (1.25 / 1000000),
        'output_price_batch': (5 / 1000000)
    },
    'gpt-4o-mini-2024-07-18': {
        'name': 'gpt-4o-mini-2024-07-18',
        'api': 'openai',
        'encoding': 'o200k_base',
        'input_price': (0.15 / 1000000),
        'input_price_cached': (0.15 / 1000000),
        'output_price': (0.6 / 1000000),
        'input_price_batch': (0.075 / 1000000),
        'output_price_batch': (0.3 / 1000000)
    },
    'gpt-4.1-2025-04-14': {
        'name': 'gpt-4.1-2025-04-14',
        'api': 'openai',
        'encoding': 'cl100k_base',
        'input_price': (2 / 1000000),
        'input_price_cached': (0.5 / 1000000),
        'output_price': (8 / 1000000),
        'input_price_batch': (1 / 1000000),
        'output_price_batch': (4 / 1000000)
    },
    'gpt-4.1-mini-2025-04-14': {
        'name': 'gpt-4.1-mini-2025-04-14',
        'api': 'openai',
        'encoding': 'cl100k_base',
        'input_price': (0.4 / 1000000),
        'input_price_cached': (0.1 / 1000000),
        'output_price': (1.6 / 1000000),
        'input_price_batch': (0.2 / 1000000),
        'output_price_batch': (0.8 / 1000000)
    },
    'gpt-4.1-nano-2025-04-14': {
        'name': 'gpt-4.1-nano-2025-04-14',
        'api': 'openai',
        'encoding': 'cl100k_base',
        'input_price': (0.1 / 1000000),
        'input_price_cached': (0.025 / 1000000),
        'output_price': (0.4 / 1000000),
        'input_price_batch': (0.05 / 1000000),
        'output_price_batch': (0.2 / 1000000)
    }
}


class ApiOpenAI(ApiBase):
    """
    API connector for OpenAI's GPT models.
    """

    def __init__(
            self,
            run_id: str,
            hostname: str = None,
            default_model: str = None,
            models: dict = None,
            api: str = 'openai',
            api_key_name: str = 'OPENAI_API_KEY',
            supports_batch: bool = True,
            use_opp_115: bool = False
    ):
        if models is None:
            models = available_models

        super().__init__(
            run_id=run_id,
            models=models,
            api=api,
            api_key_name=api_key_name,
            default_model=default_model,
            hostname=hostname,
            supports_batch=supports_batch,
            use_opp_115=use_opp_115
        )

        if self.hostname:
            self.client = OpenAI(
                api_key=self.api_key
            )
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.hostname
            )

    def close(self):
        self.client.close()

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
    ) -> tuple[str, float, float]:

        if examples is None:
            examples = []

        # start the timer and set up the client for the task
        start_time = timeit.default_timer()
        self.setup_task(task, model)

        # load the messages from the prompts folder or use the provided messages
        messages, system_msg, response_schema = util.prepare_prompt_messages(
            self.api, self.active_task, user_msg, system_msg, examples, use_opp_115=self.use_opp_115
        )

        print("--------------------------------")
        print(f"messages: {messages}")
        print("--------------------------------")

        # parse the given response format into the correct format for the API call
        parsed_response_format = self._parse_response_format(response_format, response_schema)

        # configure and prompt the API
        completion = self.client.chat.completions.create(
            messages=messages,
            model=self.active_model,
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

        print(completion)
        print("--------------------------------")

        # Get cached tokens from the nested structure, defaulting to 0 if not available
        cached_tokens = getattr(completion.usage.prompt_tokens_details, 'cached_tokens', 0) if hasattr(completion.usage, 'prompt_tokens_details') else 0
        input_tokens = completion.usage.prompt_tokens - cached_tokens
        output_tokens = completion.usage.completion_tokens

        cost = cached_tokens * self.models[self.active_model]['input_price_cached'] + input_tokens * self.models[self.active_model]['input_price'] + output_tokens * self.models[self.active_model]['output_price']
        self.total_cost += cost

        print(f"Cached tokens: {cached_tokens}")
        print(f"Input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print(f"Cost: {cost}")
        print("--------------------------------")

        try:
            reasoning = completion.choices[0].message.reasoning_content
        except AttributeError:
            reasoning = None
        output = completion.choices[0].message.content

        # TODO: Handle reasoning content if needed

        output = completion.choices[0].message.content

        input_len = completion.usage.prompt_tokens
        output_len = completion.usage.completion_tokens

        return self._log_response(start_time, output, input_len, output_len, pkg, task, response_format)

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
    ) -> tuple[list[str], float, float]:
        raise NotImplementedError("Parallel requests are not supported")

    def check_batch_status(
            self,
            task: str,
            batch_metadata_file: str = "batch_metadata.json"
    ) -> BatchStatus | None:
        # read the batch metadata file
        batch_metadata = util.read_json_file(batch_metadata_file)
        if batch_metadata is None:
            raise FileNotFoundError(f"Batch metadata file {batch_metadata_file} not found for task {task}.")

        batch = self.client.batches.retrieve(batch_metadata['id'])

        # update the batch metadata file
        util.write_to_file(batch_metadata_file, json.dumps(json.loads(batch.model_dump_json()), indent=4))

        # Convert OpenAI batch status to unified format
        status_mapping = {
            "validating": "in_progress",
            "in_progress": "in_progress",
            "finalizing": "in_progress",
            "completed": "completed",
            "failed": "failed",
            "expired": "expired",
            "cancelling": "in_progress",
            "cancelled": "canceled"
        }

        unified_status = BatchStatus(
            batch_id=batch.id,
            status=status_mapping.get(batch.status, "in_progress"),
            created_at=datetime.fromtimestamp(batch.created_at) if batch.created_at else None,
            ended_at=datetime.fromtimestamp(batch.completed_at) if batch.completed_at else None,
            expires_at=datetime.fromtimestamp(batch.expires_at) if batch.expires_at else None,
            total_requests=batch.request_counts.total if batch.request_counts else None,
            completed_requests=batch.request_counts.completed if batch.request_counts else None,
            failed_requests=batch.request_counts.failed if batch.request_counts else None,
            results_available=(batch.status == "completed" and batch.output_file_id is not None)
        )

        return unified_status

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
        if examples is None:
            examples = []

        self.setup_task(task, model)

        messages, system_msg, response_schema = util.prepare_prompt_messages(self.api, self.active_task, user_msg, system_msg, examples, use_opp_115=self.use_opp_115)
        parsed_response_format = self._parse_response_format(response_format, response_schema)

        entry = {
            "custom_id": f"{self.run_id}_{task}_{pkg}_{entry_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "messages": messages,
                "model": model,
                "n": n,
                "max_tokens": max_tokens,
                "response_format": parsed_response_format,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }
        }

        util.append_to_file(batch_input_file, json.dumps(entry))
        return entry

    def run_batch(
            self,
            task: str,
            batch_input_file: str = "batch_input.jsonl",
            batch_metadata_file: str = "batch_metadata.json"
    ):

        # count the characters in the input file and exit if it is above 209715200
        with (open(batch_input_file, "r") as f):
            input_chars = len(f.read())
            if input_chars == 0:
                raise ValueError(f"Batch input file for task {task} is empty")
            elif input_chars > 209715200:
                raise ValueError(f"Batch input file for task {task} is too large ({input_chars} characters)")
            else:
                print(f"Batch input file for task {task} is {input_chars} characters")

        batch_input_file = self.client.files.create(
            file=open(batch_input_file, "rb"),
            purpose="batch"
        )

        batch_metadata = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": task
            }
        )

        util.write_to_file(batch_metadata_file, json.dumps(json.loads(batch_metadata.model_dump_json()), indent=4))

    def retrieve_batch_result_entry(
            self,
            task: str,
            entry_id: str,
            batch_results_file: str = "batch_results.jsonl",
            valid_stop_reasons: list[str] = None
    ) -> tuple[str | None, float, float]:

        if valid_stop_reasons is None:
            valid_stop_reasons = ['stop', 'length']

        # iterate through all lines of the batch-results.jsonl file and return the one with the matching custom_id
        with (open(batch_results_file, "r") as f):
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error decoding line: {line}", file=sys.stderr)
                    raise e

                if entry['custom_id'] == entry_id:
                    # check for errors in the response
                    if entry['error'] is not None:
                        raise ValueError(f"Error in batch entry {entry_id}: {entry['error']}")
                    elif entry['response']['status_code'] != 200:
                        raise ValueError(f"Bad status code for entry {entry_id}: {entry['response']['status_code']}")
                    elif entry['response']['body']['choices'] is None or len(entry['response']['body']['choices']) == 0:
                        raise ValueError(f"No choices in response for entry {entry_id}")
                    elif entry['response']['body']['choices'][0]['finish_reason'] not in valid_stop_reasons:
                        raise ValueError(f"Finish reason not in valid stop reasons for entry {entry_id}: {entry['response']['body']['choices'][0]['finish_reason']}")

                    # return the content of the first choice if all checks pass
                    else:
                        input_length = entry['response']['body']['usage']['prompt_tokens']
                        output_length = entry['response']['body']['usage']['completion_tokens']
                        cost = input_length * self.models[self.active_model]['input_price_batch'] \
                               + output_length * self.models[self.active_model]['output_price_batch']
                        output = entry['response']['body']['choices'][0]['message']['content']
                        return output, cost, 0

        # if we got through the entire file without finding the entry we want, raise an error
        raise ValueError(f"Entry {entry_id} not found in batch results file {batch_results_file}")

    def get_batch_results(
            self,
            task: str,
            batch_metadata_file: str = "batch_metadata.json",
            batch_results_file: str = "batch_results.jsonl",
            batch_errors_file: str = "batch_errors.jsonl"
    ) -> str | None:

        # read the batch metadata file
        batch_metadata = util.read_json_file(batch_metadata_file)
        if batch_metadata is None:
            raise FileNotFoundError(f"Batch metadata file {batch_metadata_file} not found for task {task}.")

        # check for errors and write them to the errors file if necessary
        error_file_id = batch_metadata['error_file_id']
        if error_file_id is not None:
            batch_errors = self.client.files.content(batch_metadata['error_file_id']).text
            util.write_to_file(batch_errors_file, batch_errors)

        # check if the batch has completed and get the results if available
        output_file_id = batch_metadata['output_file_id']
        if output_file_id is not None:
            batch_results = self.client.files.content(batch_metadata['output_file_id']).text
            util.write_to_file(batch_results_file, batch_results)
            return batch_results
        else:
            return None

    def _load_model(self):
        # This method is intentionally left empty as OpenAI models are not loaded in the same way as other APIs.
        pass

    def _unload_model(self):
        # This method is intentionally left empty as OpenAI models are not unloaded in the same way as other APIs.
        pass

    def _parse_response_format(self, response_format, json_schema: dict) -> dict:
        if response_format == 'text':
            parsed_format = {'type': 'text'}
        elif response_format == 'json':
            parsed_format = {'type': 'json_object'}
        elif response_format == 'json_schema':
            if json_schema is None:
                raise ValueError("json_schema must be provided when response_format is 'json_schema'")
            parsed_format = {
                'type': 'json_schema',
                'json_schema': {
                    'name': f"response_schema_{self.active_task}",
                    'strict': True,
                    'schema': json_schema
                }
            }
        else:
            raise ValueError(f"response_format '{response_format}' not supported")

        return parsed_format