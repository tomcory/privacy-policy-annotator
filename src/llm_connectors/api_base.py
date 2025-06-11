import os
import timeit
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from src import util


@dataclass
class BatchStatus:
    """
    Unified batch status representation across all API connectors.
    """
    batch_id: str
    status: Literal["in_progress", "completed", "failed", "canceled", "expired"]
    created_at: datetime | None = None
    ended_at: datetime | None = None
    expires_at: datetime | None = None
    total_requests: int | None = None
    completed_requests: int | None = None
    failed_requests: int | None = None
    canceled_requests: int | None = None
    expired_requests: int | None = None
    results_available: bool = False


class ApiBase(ABC):
    """
    Abstract base class for LLM API connectors.
    """

    def __init__(
            self,
            run_id: str,
            models: dict,
            api: str,
            api_key_name: str = None,
            default_model: str = None,
            active_model: str = None,
            active_task: str = None,
            hostname: str = None,
            supports_batch: bool = False,
            supports_parallel: bool = False,
            supports_custom: bool = False
    ):
        """
        Initialize the API connector.
        
        Args:
            run_id: Unique identifier for the current run
            models: Dictionary of available models with model name as key and model configuration as value
            default_model: Name of the default model to use
            active_model: Name of the active model to use
            supports_batch: Whether the API supports batch processing
            supports_parallel: Whether the API supports parallel processing
        """
        self.run_id = run_id
        self.models = models
        self.api = api
        self.default_model = self.models[default_model] if default_model is not None else None
        self.active_model = self.models[active_model] if active_model is not None else None
        self.active_task = active_task
        self.hostname = hostname
        self.supports_batch = supports_batch
        self.supports_parallel = supports_parallel
        self.supports_custom = supports_custom

        if api_key_name is not None:
            self.api_key = os.getenv(api_key_name)
            if self.api_key is None:
                raise ValueError(f"Please set the {api_key_name} environment variable.")

    @abstractmethod
    def close(self):
        """
        Close the API client connection.
        """
        pass

    def setup_task(self, task: str, model: str):
        """
        Set up the API connector for a specific task and model.
        
        Args:
            task: Name of the task
            model: Name of the model to use for the task
        Raises:
            ValueError: If the model is not available or if no model is specified
            RuntimeError: If there is an error unloading or loading the active model
        """
        if model is not None and self.models[model] is None:
            raise ValueError(f"Model '{model}' is not available in the models dictionary.")

        if model is None and self.active_model is None and self.default_model is None:
            raise ValueError("Either a default model or a task-specific model must be specified.")

        model_changed = False

        if self.active_model is None:
            if model is None:
                self.active_model = self.default_model
            else:
                self.active_model = model
                model_changed = True
        elif model != self.active_model:
            try:
                self._unload_model()
                self.active_model = model
                model_changed = True
            except Exception as e:
                raise RuntimeError(f"Failed to unload model: {self.active_model}.\nError: {e}")

        if model_changed:
            try:
                self._load_model()
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {self.active_model}.\nError: {e}")

        self.active_task = task

    @abstractmethod
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
        """
        Send a prompt to the LLM and get a response.
        
        Args:
            pkg: Package or component name
            task: Task name
            user_msg: User message content
            system_msg: System message content
            examples: List of example input-output pairs
            model: Name of the model to use (if None, uses active or default model)
            response_format: Format of the response
            json_schema: JSON schema for structured output
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
            n: Number of responses to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            frequency_penalty: Frequency penalty parameter
            presence_penalty: Presence penalty parameter
            seed: Random seed for reproducibility
            context_window: Size of the context window
            timeout: Timeout for the request in seconds
            
        Returns:
            Tuple of (response_text, cost, processing_time)
        """
        pass

    @abstractmethod
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
        """
        Send multiple prompts to the LLM in parallel and get responses.
        
        Args:
            pkg: Package or component name
            task: Task name
            user_msgs: List of user message contents
            system_msg: System message content
            examples: List of example input-output pairs
            model: Name of the model to use (if None, uses active or default model)
            response_format: Format of the response
            json_schema: JSON schema for structured output
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
            n: Number of responses to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            frequency_penalty: Frequency penalty parameter
            presence_penalty: Presence penalty parameter
            seed: Random seed for reproducibility
            context_window: Size of the context window
            timeout: Timeout for the request in seconds
            
        Returns:
            Tuple of (list_of_responses, total_cost, total_processing_time)
        """
        pass

    @abstractmethod
    def check_batch_status(
            self,
            task: str,
            batch_metadata_file: str = "batch_metadata.json"
    ) -> BatchStatus | None:
        pass

    @abstractmethod
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
        """
        Prepare a batch entry for batch processing.
        
        Args:
            pkg: Package or component name
            task: Task name
            user_msg: User message content
            system_msg: System message content
            examples: List of example input-output pairs
            entry_id: Unique identifier for the batch entry
            model: Name of the model to use for the batch entry (if None, uses active or default model)
            response_format: Format of the response
            json_schema: JSON schema for structured output
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
            n: Number of responses to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            frequency_penalty: Frequency penalty parameter
            presence_penalty: Presence penalty parameter
            seed: Random seed for reproducibility
            context_window: Size of the context window
            timeout: Timeout for the request in seconds
            batch_input_file: Path to the batch input file
            
        Returns:
            Batch entry configuration
        """
        pass

    @abstractmethod
    def run_batch(
            self,
            task: str,
            batch_input_file: str = "batch_input.jsonl",
            batch_metadata_file: str = "batch_metadata.json"
    ):
        """
        Run a batch of prompts.
        
        Args:
            task: Task name
            batch_input_file: Path to the input file
            batch_metadata_file: Path to the metadata file
        """
        pass

    @abstractmethod
    def retrieve_batch_result_entry(
            self,
            task: str,
            entry_id: str,
            batch_results_file: str = "batch_results.jsonl",
            valid_stop_reasons: list[str] = None
    ) -> tuple[str | None, float, float]:
        """
            Check the status of a specific entry in a batch job.

            This method retrieves the result entry of a batch job from a specified
            batch results file. If the entry is not yet completed or has an invalid
            stop reason, it will return None for the result. The method also calculates
            and returns the elapsed time and total time for the task.

            Args:
                task: The name of the task whose batch entry result is to be retrieved.
                entry_id: The identifier of the batch entry to retrieve.
                batch_results_file: Path to the JSONL file containing batch results.
                valid_stop_reasons: A list of acceptable stop reasons for the task.

            Returns:
                A tuple containing:
                - The result of the batch entry (str or None if not valid/completed).
                - The elapsed time for the task execution as a float.
                - The total time for the task execution as a float.
        """
        pass

    @abstractmethod
    def get_batch_results(
            self,
            task: str,
            batch_metadata_file: str = "batch_metadata.json",
            batch_results_file: str = "batch_results.jsonl",
            batch_errors_file: str = "batch_errors.jsonl"
    ) -> str | None:
        """
        Get the results of a batch job.

        Args:
            task: Task name
            batch_metadata_file: Path to the batch metadata file
            batch_results_file: Path to the batch results file
            batch_errors_file: Path to the batch errors file

        Returns:
            Batch results
        """
        pass

    @abstractmethod
    def _unload_model(self):
        """
        Unload the currently active model. This method can be overridden by subclasses if needed.
        """
        pass

    @abstractmethod
    def _load_model(self):
        """
        Load the currently active model. This method can be overridden by subclasses if needed.
        """
        pass

    def _log_response(
            self,
            start_time: float,
            output: str,
            input_len: int,
            output_len: int,
            pkg: str,
            task: str,
            response_format: Literal['text', 'json', 'json_schema']
    ) -> tuple[str, float, float]:
        processing_time = timeit.default_timer() - start_time

        # calculate the cost of the API call based on the total number of tokens used
        cost = input_len * self.models[self.active_model]['input_price'] + output_len * self.models[self.active_model]['output_price']

        # log the prompt and result
        if self.run_id is not None and pkg is not None and task is not None:
            util.log_prompt_result(
                run_id=self.run_id,
                task=task,
                pkg=pkg,
                model_name=self.active_model,
                output_format='txt' if response_format == 'text' else 'json',
                cost=cost,
                processing_time=processing_time,
                outputs=[output]
            )

        return output, cost, processing_time
