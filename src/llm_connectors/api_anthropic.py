import json
import os
import sys
import timeit
from datetime import datetime
from typing import Literal, Union, Tuple, List, Optional

from anthropic import Anthropic

from src import util
from src.llm_connectors.api_base import ApiBase, BatchStatus

api = 'anthropic'

models = {
    'claude-opus-4': {
        'name': 'claude-opus-4-0',
        'display_name': 'Claude Opus 4',
        'api': api,
        'encoding': 'cl100k_base',
        'input_price': (15 / 1000000),
        'output_price': (75 / 1000000),
        'input_price_batch': (7.5 / 1000000),  # 50% discount for batch
        'output_price_batch': (37.5 / 1000000)  # 50% discount for batch
    },
    'claude-sonnet-4': {
        'name': 'claude-sonnet-4-0',
        'display_name': 'Claude Sonnet 4',
        'api': api,
        'encoding': 'cl100k_base',
        'input_price': (3 / 1000000),
        'output_price': (15 / 1000000),
        'input_price_batch': (1.5 / 1000000),  # 50% discount for batch
        'output_price_batch': (7.5 / 1000000)  # 50% discount for batch
    },
    'claude-3-7-sonnet': {
        'name': 'claude-3-7-sonnet-latest',
        'display_name': 'Claude Sonnet 3.7',
        'api': api,
        'encoding': 'cl100k_base',
        'input_price': (3 / 1000000),
        'output_price': (15 / 1000000),
        'input_price_batch': (1.5 / 1000000),  # 50% discount for batch
        'output_price_batch': (7.5 / 1000000)  # 50% discount for batch
    },
    'claude-3-5-sonnet': {
        'name': 'claude-3-5-sonnet-latest',
        'display_name': 'Claude Sonnet 3.5 (New)',
        'api': api,
        'encoding': 'cl100k_base',
        'input_price': (15 / 1000000),
        'output_price': (75 / 1000000),
        'input_price_batch': (7.5 / 1000000),  # 50% discount for batch
        'output_price_batch': (37.5 / 1000000)  # 50% discount for batch
    },
    'claude-3-5-haiku': {
        'name': 'claude-3-5-haiku-latest',
        'display_name': 'Claude Haiku 3.5',
        'api': api,
        'encoding': 'cl100k_base',
        'input_price': (0.8 / 1000000),
        'output_price': (4 / 1000000),
        'input_price_batch': (0.4 / 1000000),  # 50% discount for batch
        'output_price_batch': (2 / 1000000)  # 50% discount for batch
    }
}


def _format_messages(system_msg: str, user_msg: str, examples: List[Tuple[str, str]] = None) -> List[dict]:
    """
    Format messages for the Anthropic API.
    
    Args:
        system_msg: The system message
        user_msg: The user message
        examples: List of example (user, assistant) pairs
        
    Returns:
        List of message dictionaries in Anthropic format
    """
    messages = []

    if examples and len(examples) > 0:
        for user_example, assistant_example in examples:
            messages.append({"role": "user", "content": user_example})
            messages.append({"role": "assistant", "content": assistant_example})

    messages.append({"role": "user", "content": user_msg})

    return messages


class ApiAnthropic(ApiBase):

    def __init__(self, run_id: str, hostname: str = None, default_model: str = None):
        super().__init__(
            run_id=run_id,
            models=models,
            api=api,
            api_key_name='ANTHROPIC_API_KEY',
            default_model=default_model,
            hostname=hostname,
            supports_batch=True
        )

        if hostname:
            self.client = Anthropic(
                api_key=self.api_key,
                base_url=hostname
            )
        else:
            self.client = Anthropic(api_key=self.api_key)

    def close(self):
        print("Closed Anthropic API client.")
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
            examples=examples,
            bundle_system_msg=False
        )

        completion = self.client.messages.create(
            model=self.active_model['name'],
            messages=messages,
            system=system_msg,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False
        )

        output = completion.content[0].text
        input_len = completion.usage.input_tokens
        output_len = completion.usage.output_tokens

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
        raise NotImplementedError("Parallel requests are not supported by the Anthropic API connector")

    def prepare_batch_entry(
            self,
            pkg: str,
            task: str,
            user_msg: str,
            system_msg: str = None,
            examples: list[tuple[str, str]] = [],
            entry_id: int = 0,
            model: str = None,
            response_format: Union[Literal['text', 'json', 'json_schema']] = 'text',
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
        """
        Prepare a batch entry for Anthropic Message Batches API.
        """
        self.setup_task(task, model)

        # Load the messages from the prompts folder or use the provided messages
        _, system_msg, response_schema, _, _, _ = util.prepare_prompt_messages(
            api, task, user_msg, system_msg, examples
        )

        # Format messages for Anthropic API
        messages = _format_messages(system_msg, user_msg, examples)

        # Set up response format
        response_format_dict = None
        if response_format == 'json' or response_format == 'json_schema':
            response_format_dict = {"type": "json"}
            if json_schema is not None and response_format == 'json_schema':
                response_format_dict["schema"] = json_schema

        # Create the batch request entry
        batch_entry = {
            "custom_id": f"{self.run_id}_{task}_{pkg}_{entry_id}",
            "params": {
                "model": model['name'],
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
        }

        # Add system message if provided
        if system_msg:
            batch_entry["params"]["system"] = system_msg

        # Add response format if specified
        if response_format_dict:
            batch_entry["params"]["response_format"] = response_format_dict

        return batch_entry

    def run_batch(self, task: str):
        """
        Run a batch of message requests using Anthropic's Message Batches API.
        """
        batch_input_file = f"../output/{self.run_id}/batch/{task}/batch_input.jsonl"

        # Check if input file exists and is not empty
        if not os.path.exists(batch_input_file):
            print(f"Batch input file for task {task} does not exist, exiting...")
            sys.exit(1)

        with open(batch_input_file, "r") as f:
            input_content = f.read()
            if len(input_content.strip()) == 0:
                print(f"Input file for task {task} is empty, exiting...")
                sys.exit(1)

            # Parse and validate requests
            requests = []
            for line in input_content.strip().split('\n'):
                if line.strip():
                    requests.append(json.loads(line))

            print(f"Input file for task {task} contains {len(requests)} requests")

        print(f"Running batch for task {task}...")

        # Create the message batch using the correct API endpoint
        message_batch = self.client.messages.batches.create(
            requests=requests
        )

        # Save batch metadata
        batch_metadata = {
            "id": message_batch.id,
            "type": message_batch.type,
            "processing_status": message_batch.processing_status,
            "request_counts": message_batch.request_counts.model_dump() if message_batch.request_counts else None,
            "ended_at": message_batch.ended_at,
            "created_at": message_batch.created_at,
            "expires_at": message_batch.expires_at,
            "cancel_initiated_at": message_batch.cancel_initiated_at,
            "results_url": message_batch.results_url
        }

        batch_metadata_json = json.dumps(batch_metadata, indent=4, default=str)

        # Write the batch metadata to a file
        with open(f"../output/{self.run_id}/batch/{task}/batch_metadata.json", "w") as f:
            f.write(batch_metadata_json)

        print(f"Batch {message_batch.id} created successfully for task {task}")

    def retrieve_batch_result_entry(self, task: str, entry_id: str, batch_results_file: str = "batch_results.jsonl"):
        """
        Retrieve a specific batch result entry from the results file.
        """
        results_file_path = f"../output/{self.run_id}/batch/{task}/{batch_results_file}"

        if not os.path.exists(results_file_path):
            print(f"Batch results file {results_file_path} does not exist")
            return None, 0, 0

        with open(results_file_path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Error decoding line: {line}")
                    continue

                if entry.get('custom_id') == entry_id:
                    # Check for errors in the response
                    if 'error' in entry and entry['error'] is not None:
                        print(f"Error for entry {entry_id}: {entry['error']}")
                        return None, 0, 0

                    # Extract the result - check if request succeeded
                    if entry.get('result', {}).get('type') == 'succeeded':
                        result = entry['result']
                        message = result.get('message')
                        if not message or not message.get('content'):
                            print(f"No message content for entry {entry_id}")
                            return None, 0, 0

                        output = message['content'][0]['text']

                        # Calculate cost using batch pricing
                        usage = result.get('usage', {})
                        input_tokens = result.usage.get('input_tokens', 0)
                        output_tokens = usage.get('output_tokens', 0)

                        # Use the model from the result or default
                        model_name = result.get('model', self.active_model[
                            'name'] if self.active_model else 'claude-3-5-sonnet-latest')
                        model_config = None
                        for model_key, config in models.items():
                            if config['name'] == model_name:
                                model_config = config
                                break

                        if model_config:
                            cost = (input_tokens * model_config['input_price_batch'] +
                                    output_tokens * model_config['output_price_batch'])
                        else:
                            cost = 0

                        return output, cost, 0
                    elif entry.get('result', {}).get('type') == 'errored':
                        error = entry['result'].get('error', {})
                        print(f"Request errored for entry {entry_id}: {error}")
                        return None, 0, 0
                    elif entry.get('result', {}).get('type') == 'canceled':
                        print(f"Request was canceled for entry {entry_id}")
                        return None, 0, 0
                    elif entry.get('result', {}).get('type') == 'expired':
                        print(f"Request expired for entry {entry_id}")
                        return None, 0, 0

        print(f"Entry {entry_id} not found in batch results")
        return None, 0, 0

    def check_batch_status(self, task: str, batch_metadata_file: str = "batch_metadata.json") -> Optional[BatchStatus]:
        """
        Check the status of a message batch.
        """
        metadata_file_path = f"../output/{self.run_id}/batch/{task}/{batch_metadata_file}"

        if not os.path.exists(metadata_file_path):
            return None

        with open(metadata_file_path, "r") as f:
            batch_metadata = json.load(f)

        batch_id = batch_metadata.get('id')
        if not batch_id:
            print("No batch ID found in metadata")
            return None

        # Retrieve current batch status using the correct API endpoint
        try:
            message_batch = self.client.messages.batches.retrieve(batch_id)

            # Update metadata with current status
            updated_metadata = {
                "id": message_batch.id,
                "type": message_batch.type,
                "processing_status": message_batch.processing_status,
                "request_counts": message_batch.request_counts.model_dump() if message_batch.request_counts else None,
                "ended_at": message_batch.ended_at,
                "created_at": message_batch.created_at,
                "expires_at": message_batch.expires_at,
                "cancel_initiated_at": message_batch.cancel_initiated_at,
                "results_url": message_batch.results_url
            }

            # Write updated metadata back to file
            with open(metadata_file_path, "w") as f:
                json.dump(updated_metadata, f, indent=4, default=str)

            # Convert Anthropic batch status to unified format
            status_mapping = {
                "in_progress": "in_progress",
                "ended": "completed",
                "failed": "failed",
                "canceled": "canceled",
                "expired": "expired"
            }

            # Parse datetime strings if they exist
            created_at = None
            ended_at = None
            expires_at = None

            if message_batch.created_at:
                try:
                    created_at = datetime.fromisoformat(message_batch.created_at.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    created_at = message_batch.created_at

            if message_batch.ended_at:
                try:
                    ended_at = datetime.fromisoformat(message_batch.ended_at.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    ended_at = message_batch.ended_at

            if message_batch.expires_at:
                try:
                    expires_at = datetime.fromisoformat(message_batch.expires_at.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    expires_at = message_batch.expires_at

            unified_status = BatchStatus(
                batch_id=message_batch.id,
                status=status_mapping.get(message_batch.processing_status, "in_progress"),
                created_at=created_at,
                ended_at=ended_at,
                expires_at=expires_at,
                total_requests=message_batch.request_counts.processing + message_batch.request_counts.succeeded + message_batch.request_counts.errored + message_batch.request_counts.canceled + message_batch.request_counts.expired if message_batch.request_counts else None,
                completed_requests=message_batch.request_counts.succeeded if message_batch.request_counts else None,
                failed_requests=message_batch.request_counts.errored if message_batch.request_counts else None,
                canceled_requests=message_batch.request_counts.canceled if message_batch.request_counts else None,
                expired_requests=message_batch.request_counts.expired if message_batch.request_counts else None,
                results_available=(message_batch.processing_status == "ended" and message_batch.results_url is not None)
            )

            return unified_status

        except Exception as e:
            print(f"Error retrieving batch status: {e}")
            return None

    def get_batch_results(self, task: str, batch_metadata_file: str = "batch_metadata.json"):
        """
        Get the results of a completed message batch.
        """
        metadata_file_path = f"../output/{self.run_id}/batch/{task}/{batch_metadata_file}"

        if not os.path.exists(metadata_file_path):
            print(f"Batch metadata file {metadata_file_path} does not exist")
            return None

        with open(metadata_file_path, "r") as f:
            batch_metadata = json.load(f)

        batch_id = batch_metadata.get('id')
        if not batch_id:
            print("No batch ID found in metadata")
            return None

        try:
            # Retrieve the batch to get current status
            message_batch = self.client.messages.batches.retrieve(batch_id)

            if message_batch.processing_status != "ended":
                print(f"Batch {batch_id} is not yet completed. Status: {message_batch.processing_status}")
                return None

            if not message_batch.results_url:
                print(f"No results URL available for batch {batch_id}")
                return None

            # Get the results using the correct API endpoint
            results = self.client.messages.batches.results(batch_id)

            # Write results to file
            results_file_path = f"../output/{self.run_id}/batch/{task}/batch_results.jsonl"
            with open(results_file_path, "w") as f:
                for result in results:
                    f.write(json.dumps(result.model_dump()) + "\n")

            print(f"Batch results saved to {results_file_path}")
            return results

        except Exception as e:
            print(f"Error retrieving batch results: {e}")
            return None

    def _load_model(self):
        # This method is intentionally left empty as Anthropic models are not loaded in the same way as other APIs.
        pass

    def _unload_model(self):
        # This method is intentionally left empty as Anthropic models are not unloaded in the same way as other APIs.
        pass
