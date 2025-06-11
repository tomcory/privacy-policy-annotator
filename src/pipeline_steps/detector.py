import tiktoken
from bs4 import BeautifulSoup

from src import util
from src.llm_connectors.api_base import ApiBase
from src.pipeline_steps.pipeline_step import PipelineStep
from src.state_manager import BaseStateManager


class Detector(PipelineStep):
    def __init__(
            self,
            run_id: str,
            skip: bool,
            is_batch_step: bool,
            in_folder: str,
            out_folder: str,
            state_manager: BaseStateManager,
            batch_input_file: str = "batch_input.json",
            batch_metadata_file: str = "batch_metadata.json",
            batch_results_file: str = "batch_results.jsonl",
            batch_errors_file: str = "batch_errors.jsonl",
            parallel_prompt: bool = False,
            model: str = None,
            client: ApiBase = None
    ):
        super().__init__(
            run_id=run_id,
            task='detect',
            details='',
            skip=skip,
            is_llm_step=True,
            is_batch_step=is_batch_step,
            in_folder=in_folder,
            out_folder=out_folder,
            state_manager=state_manager,
            batch_input_file=batch_input_file,
            batch_metadata_file=batch_metadata_file,
            batch_results_file=batch_results_file,
            batch_errors_file=batch_errors_file,
            parallel_prompt=parallel_prompt,
            model=model,
            client=client
        )

    async def execute(self, pkg: str):
        try:
            html = util.read_from_file(f"{self.in_folder}/{pkg}.html")
            if html is None:
                return None

            if self.is_batch_step:
                output, cost, time = self.client.retrieve_batch_result_entry(self.task, f"{self.run_id}_{self.task}_{pkg}_0")
            else:
                text = BeautifulSoup(html, 'html.parser').get_text()
                if html is None or html == '':
                    return None
                output, cost, time = self.client.prompt(
                    pkg=pkg,
                    task=self.task,
                    model=self.model,
                    user_msg=self._generate_excerpt(text),
                    max_tokens=1
                )

            # sort the output accordingly
            if output == 'true':
                folder = ''
            elif output == 'unknown':
                folder = '/unknown'
            else:
                folder = '/rejected'

            file_name = f"{self.out_folder}{folder}/{pkg}.html"
            util.write_to_file(file_name, html)

            await self.state_manager.update_state(file_progress=0.5)
        except Exception as e:
            await self.state_manager.raise_error(error_message=str(e))
            util.write_to_file(f"../../output/{self.run_id}/log/failed_detect.txt", pkg) #TODO: replace with logger
            return None


    async def prepare_batch(self, pkg: str):
        try:
            html = util.read_from_file(f"{self.in_folder}/{pkg}.html")

            if html is None or html == '':
                return None

            batch_entry = self.client.prepare_batch_entry(
                pkg=pkg,
                task=self.task,
                model=self.model,
                user_msg=str(self._generate_excerpt(html)),
                max_tokens=1
            )

            await self.state_manager.update_state(file_progress=0.5)
            return [batch_entry]

        except Exception as e:
            await self.state_manager.raise_error(error_message=str(e))
            return None

    def _generate_excerpt(self, text: str):
        # Split the text into words
        words = text.split()

        # Calculate the total number of words
        total_words = len(words)

        # If the text has fewer than 50 words, return the entire text
        if total_words <= 50:
            return text

        # Calculate the starting index for the center 100 words
        center_index = total_words // 2
        start_index = max(center_index - 25, 0)
        end_index = min(start_index + 50, total_words)

        # Join the selected words back into a string
        excerpt = ' '.join(words[start_index:end_index])

        return excerpt

