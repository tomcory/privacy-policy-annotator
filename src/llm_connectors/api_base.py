from abc import ABC, abstractmethod
from typing import Literal, Union, Tuple, List, Dict, Any, Optional
from pydantic import BaseModel
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BatchStatus:
    """
    Unified batch status representation across all API connectors.
    """
    batch_id: str
    status: Literal["in_progress", "completed", "failed", "canceled", "expired"]
    created_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    total_requests: Optional[int] = None
    completed_requests: Optional[int] = None
    failed_requests: Optional[int] = None
    canceled_requests: Optional[int] = None
    expired_requests: Optional[int] = None
    results_available: bool = False


class ApiBase(ABC):
    """
    Abstract base class for LLM API connectors.
    """
    
    @abstractmethod
    def __init__(
        self,
        run_id: str,
        models: dict,
        default_model: str = None,
        active_model: str = None,
        active_task: str = None,
        hostname: str = None,
        supports_batch: bool = False,
        supports_parallel: bool = False
        ):
        """
        Initialize the API connector.
        
        Args:
            run_id: Unique identifier for the current run
            models: Dictionary of availablemodels with model name as key and model configuration as value
            default_model: Name of the default model to use
            active_model: Name of the active model to use
            supports_batch: Whether the API supports batch processing
            supports_parallel: Whether the API supports parallel processing
        """
        self.run_id = run_id
        self.active_task = active_task
        self.hostname = hostname
        self.supports_batch = supports_batch
        self.supports_parallel = supports_parallel

        if default_model is not None:
            self.default_model = models[default_model]

        if active_model is not None:
            self.active_model = models[active_model]
    
    @abstractmethod
    def setup(self, task: str, model: dict, use_custom_model: bool = False):
        """
        Set up the API connector for a specific task and model.
        
        Args:
            task: Name of the task
            model: Model configuration dictionary
            use_custom_model: Whether to use a custom model for the task
        """
        self.active_task = task
        self.active_model = model
    
    @abstractmethod
    def close(self):
        """
        Close the API client connection.
        """
        pass
    
    @abstractmethod
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
        """
        Send a prompt to the LLM and get a response.
        
        Args:
            pkg: Package or component name
            task: Task name
            user_msg: User message content
            system_msg: System message content
            examples: List of example input-output pairs
            model: Model configuration dictionary
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
        """
        Send multiple prompts to the LLM in parallel and get responses.
        
        Args:
            pkg: Package or component name
            task: Task name
            user_msgs: List of user message contents
            system_msg: System message content
            examples: List of example input-output pairs
            model: Model configuration dictionary
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
        """
        Prepare a batch entry for batch processing.
        
        Args:
            pkg: Package or component name
            task: Task name
            user_msg: User message content
            system_msg: System message content
            examples: List of example input-output pairs
            entry_id: Unique identifier for the batch entry
            model: Model configuration dictionary
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
            Batch entry configuration
        """
        pass
    
    @abstractmethod
    def run_batch(self, task: str):
        """
        Run a batch of prompts.
        
        Args:
            task: Task name
        """
        pass
    
    @abstractmethod
    def retrieve_batch_result_entry(self, task: str, entry_id: str, batch_results_file: str = "batch_results.jsonl"):
        """
        Retrieve a specific batch result entry.
        
        Args:
            task: Task name
            entry_id: Unique identifier for the batch entry
            batch_results_file: Path to the batch results file
            
        Returns:
            Batch result entry
        """
        pass
    
    @abstractmethod
    def check_batch_status(self, task: str, batch_metadata_file: str = "batch_metadata.json") -> Optional[BatchStatus]:
        """
        Check the status of a batch job.
        
        Args:
            task: Task name
            batch_metadata_file: Path to the batch metadata file
            
        Returns:
            Unified BatchStatus object or None if batch doesn't exist
        """
        pass
    
    @abstractmethod
    def get_batch_results(self, task: str, batch_metadata_file: str = "batch_metadata.json"):
        """
        Get the results of a batch job.
        
        Args:
            task: Task name
            batch_metadata_file: Path to the batch metadata file
            
        Returns:
            Batch results
        """
        pass
