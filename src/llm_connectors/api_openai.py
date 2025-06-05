import json
import os
import sys
import timeit
from datetime import datetime
from typing import Literal, Union, Tuple, List, Optional

from openai import OpenAI
from pydantic import BaseModel

from src import util
from src.llm_connectors.api_base import ApiBase, BatchStatus

api = 'openai'

models = {
    'gpt-3.5-turbo': {
        'name': 'gpt-3.5-turbo',
        'api': api,
        'encoding': 'cl100k_base',
        'input_price': (0.5 / 1000000),
        'output_price': (1.5 / 1000000),
        'input_price_batch': (0.25 / 1000000),
        'output_price_batch': (0.75 / 1000000)
    },
    'gpt-4': {
        'name': 'gpt-4',
        'api': api,
        'encoding': 'cl100k_base',
        'input_price': (30 / 1000000),
        'output_price': (60 / 1000000),
        'input_price_batch': (15 / 1000000),
        'output_price_batch': (30 / 1000000)
    },
    'gpt-4o': {
        'name': 'gpt-4o-2024-11-20',
        'api': api,
        'encoding': 'o200k_base',
        'input_price': (2.5 / 1000000),
        'output_price': (10 / 1000000),
        'input_price_batch': (1.25 / 1000000),
        'output_price_batch': (5 / 1000000)
    },
    'gpt-4o-mini': {
        'name': 'gpt-4o-mini-2024-07-18',
        'api': api,
        'encoding': 'o200k_base',
        'input_price': (0.15 / 1000000),
        'output_price': (0.6 / 1000000),
        'input_price_batch': (0.075 / 1000000),
        'output_price_batch': (0.3 / 1000000)
    }
}


def _parse_response_format(response_format, json_schema: dict, task: str) -> dict:
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
                'name': f"response_schema_{task}",
                'strict': True,
                'schema': json_schema
            }
        }
    else:
        raise ValueError(f"response_format '{response_format}' not supported")

    return parsed_format


class ApiOpenAI(ApiBase):
    """
    API connector for OpenAI's GPT models.
    """

    def __init__(self, run_id: str, hostname: str = None, default_model: str = None):
        super().__init__(
            run_id=run_id,
            models=models,
            api=api,
            api_key_name='OPENAI_API_KEY',
            default_model=default_model,
            hostname=hostname,
            supports_batch=True
        )

        self.client = OpenAI(
            api_key=self.api_key
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

        # start the timer to measure the processing time
        start_time = timeit.default_timer()

        self.setup_task(task, model)

        # load the messages from the prompts folder or use the provided messages
        messages, system_msg, response_schema = util.prepare_prompt_messages(
            api, task, user_msg, system_msg, examples
        )

        # parse the given response format into the correct format for the API call
        parsed_response_format = _parse_response_format(response_format, response_schema, task)

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

        output = completion.choices[0].message.content

        # stop the timer and calculate the processing time
        processing_time = timeit.default_timer() - start_time

        input_len = completion.usage.prompt_tokens
        output_len = completion.usage.completion_tokens

        # calculate the cost of the API call based on the total number of tokens used
        cost = input_len * self.active_model['input_price'] + output_len * self.active_model['output_price']

        # log the prompt and result
        if self.run_id is not None and pkg is not None and task is not None:
            util.log_prompt_result(
                run_id=self.run_id,
                task=task,
                pkg=pkg,
                model_name=self.active_model['name'],
                output_format='txt' if response_format == 'text' else 'json',
                cost=cost,
                processing_time=processing_time,
                outputs=[output]
            )

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
        raise NotImplementedError("Parallel requests are not supported by the OpenAI API")

    def prepare_batch_entry(
            self,
            pkg: str,
            task: str,
            user_msg: str,
            system_msg: str = None,
            examples: list[tuple[str, str]] = [],
            entry_id: int = 0,
            model: dict = None,
            response_format: Union[Literal['text', 'json', 'json_schema'], BaseModel] = 'text',
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
        if model is None:
            if self.default_model is None:
                raise ValueError("model must be provided when default_model is not set")
            model = self.default_model

        messages, system_msg, response_schema = util.prepare_prompt_messages(api, task, user_msg, system_msg, examples)
        parsed_response_format = _parse_response_format(response_format, response_schema, task)

        return {
            "custom_id": f"{self.run_id}_{task}_{pkg}_{entry_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "messages": messages,
                "model": model['name'],
                "n": n,
                "max_tokens": max_tokens,
                "response_format": parsed_response_format,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }
        }

    def run_batch(self, task: str):

        # count the characters in the input file and exit if it is above 209715200
        with (open(f"../output/{self.run_id}/batch/{task}/batch_input.jsonl", "r") as f):
            input_chars = len(f.read())
            if input_chars == 0:
                print(f"Input file for task {task} is empty, exiting...")
                sys.exit(1)
            elif input_chars > 209715200:
                print(f"Input file for task {task} is too large ({input_chars} characters), exiting...")
                sys.exit(1)
            else:
                print(f"Input file for task {task} is {input_chars} characters")

        print(f"Running batch for task {task}...")
        batch_input_file = self.client.files.create(
            file=open(f"../output/{self.run_id}/batch/{task}/batch_input.jsonl", "rb"),
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

        batch_metadata_json = json.dumps(json.loads(batch_metadata.model_dump_json()), indent=4)

        # write the batch metadata to a file
        with open(f"../output/{self.run_id}/batch/{task}/batch_metadata.json", "w") as f:
            f.write(batch_metadata_json)

    def retrieve_batch_result_entry(self, task: str, entry_id: str, batch_results_file: str = "batch_results.jsonl"):
        # iterate through all lines of the batch-results.jsonl file and return the one with the matching custom_id
        # print(f"Retrieving entry {entry_id} from batch results...")
        with open(f"../output/{self.run_id}/batch/{task}/{batch_results_file}", "r") as f:
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
                        return None, 0, 0
                    elif entry['response']['status_code'] != 200:
                        print(f"Bad status code for entry {entry_id}: {entry['response']['status_code']}")
                        return None, 0, 0
                    elif entry['response']['body']['choices'] is None or len(entry['response']['body']['choices']) == 0:
                        print(f"No choices for entry {entry_id}")
                        return None, 0, 0
                    elif entry['response']['body']['choices'][0]['finish_reason'] not in ['stop', 'length']:
                        print(
                            f"Finish reason not 'stop' or 'length' for entry {entry_id}: {entry['response']['body']['choices'][0]['finish_reason']}")
                        return None, 0, 0
                    # return the content of the first choice
                    else:
                        prompt_cost = entry['response']['body']['usage']['prompt_tokens'] * models['gpt-4o-mini'][
                            'input_price'] / 2
                        completion_cost = entry['response']['body']['usage']['completion_tokens'] * \
                                          models['gpt-4o-mini']['output_price'] / 2
                        output = entry['response']['body']['choices'][0]['message']['content']

                        return output, prompt_cost + completion_cost, 0

        print(f"Entry {entry_id} not found")
        return None, 0, 0

    def check_batch_status(self, task: str, batch_metadata_file: str = "batch_metadata.json") -> Optional[BatchStatus]:
        if not os.path.exists(f"../output/{self.run_id}/batch/{task}/{batch_metadata_file}"):
            return None

        with open(f"../output/{self.run_id}/batch/{task}/{batch_metadata_file}", "r") as f:
            batch_metadata = json.load(f)

        batch = self.client.batches.retrieve(batch_metadata['id'])

        batch_status_json = json.dumps(json.loads(batch.model_dump_json()), indent=4)

        # write the batch status to a file
        with open(f"../output/{self.run_id}/batch/{task}/{batch_metadata_file}", "w") as f:
            f.write(f"{batch_status_json}")

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

    def get_batch_results(self, task: str, batch_metadata_file: str = "batch_metadata.json"):

        with open(f"../output/{self.run_id}/batch/{task}/{batch_metadata_file}", "r") as f:
            batch_metadata = json.load(f)

        error_file_id = batch_metadata['error_file_id']
        if error_file_id is not None:
            batch_errors = self.client.files.content(batch_metadata['error_file_id']).text
            with open(f"../output/{self.run_id}/batch/{task}/batch_errors.jsonl", "w") as f:
                f.write(batch_errors)

        output_file_id = batch_metadata['output_file_id']
        if output_file_id is not None:
            batch_results = self.client.files.content(batch_metadata['output_file_id']).text
            with open(f"../output/{self.run_id}/batch/{task}/batch_results.jsonl", "w") as f:
                f.write(batch_results)

            return batch_results
        else:
            return None

    def _load_model(self):
        # This method is intentionally left empty as OpenAI models are not loaded in the same way as other APIs.
        pass

    def _unload_model(self):
        # This method is intentionally left empty as OpenAI models are not unloaded in the same way as other APIs.
        pass
