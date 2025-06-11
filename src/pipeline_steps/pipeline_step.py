import json
from abc import ABC, abstractmethod

from src import util
from src.llm_connectors.api_base import ApiBase
from src.state_manager import BaseStateManager


class PipelineStep(ABC):
    """Represents a step in the pipeline."""
    def __init__(
            self,
            run_id: str,
            task: str,
            details: str,
            skip: bool,
            is_llm_step: bool,
            is_batch_step: bool,
            in_folder: str,
            out_folder: str,
            state_manager: BaseStateManager,
            batch_input_file: str,
            batch_metadata_file: str,
            batch_results_file: str,
            batch_errors_file: str,
            parallel_prompt: bool = False,
            model: str = None,
            client: ApiBase = None
    ):
        self.run_id = run_id
        self.task = task
        self.details = details
        self.skip = skip
        self.is_llm_step = is_llm_step
        self.is_batch_step = is_batch_step
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.state_manager = state_manager
        self.batch_input_file = batch_input_file
        self.batch_metadata_file = batch_metadata_file
        self.batch_results_file = batch_results_file
        self.batch_errors_file = batch_errors_file
        self.parallel_prompt = parallel_prompt
        self.model = model
        self.client = client

        print(f"Initializing pipeline step '{task}' for run '{run_id}'")
        print(f"  details: {details}")
        print(f"  skip: {skip}")
        print(f"  is_llm_step: {is_llm_step}")
        print(f"  is_batch_step: {is_batch_step}")
        print(f"  in folder '{self.in_folder}'")
        print(f"  out folder '{self.out_folder}'")
        print(f"  batch input file '{self.batch_input_file}'")
        print(f"  batch metadata file '{self.batch_metadata_file}'")
        print(f"  batch results file '{self.batch_results_file}'")
        print(f"  batch errors file '{self.batch_errors_file}'")
        print(f"  state manager '{self.state_manager}'")

    @abstractmethod
    async def execute(self, pkg: str):
        pass

    @abstractmethod
    async def prepare_batch(self, pkg: str):
        pass

    def create_batch_input_file(self, batch_entries: list[str]):
        jsonl = ''
        for entry in batch_entries:
            jsonl += json.dumps(entry) + '\n'
        util.write_to_file(self.batch_input_file, jsonl)

    def run_batch(self):
        self.client.run_batch(
            self.task,
            self.batch_input_file,
            self.batch_metadata_file
        )

    def check_batch_status(self):
        return self.client.check_batch_status(self.task, self.batch_metadata_file)

    def get_batch_results(self):
        self.client.get_batch_results(
            self.task,
            self.batch_metadata_file,
            self.batch_results_file,
            self.batch_errors_file
        )