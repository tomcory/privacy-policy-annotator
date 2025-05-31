import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Union, Callable, Awaitable

from src import util
from src.llm_connectors import api_openai, api_ollama, api_deepseek, api_anthropic
from src.llm_connectors.api_base import ApiBase, BatchStatus
from src.pipeline_steps import cleaner, detector, parser, annotator
from src.state_manager import PipelineStatus, BaseStateManager


@dataclass
class PipelineStep:
    """Represents a step in the pipeline."""
    name: str
    details: str
    skip: bool
    is_llm_step: bool
    is_batch_step: bool
    use_batch_result: bool
    func: Callable
    in_folder: str
    out_folder: str
    parallel_prompt: bool = False
    model: dict = None
    client: ApiBase = None


class PipelineExecutor:
    def __init__(
            self,
            run_id: str,
            default_model: str,
            models: dict,
            skip_clean: bool,
            skip_detect: bool,
            skip_parse: bool,
            skip_annotate: bool,
            skip_review: bool,
            batch_detect: bool,
            batch_annotate: bool ,
            batch_review: bool,
            parallel_prompt: bool,
            hostname: str,
            state_manager: BaseStateManager
    ):
        self.run_id = run_id
        self.state_manager = state_manager

        if not self.run_id:
            self.state_manager.raise_error("No run_id supplied", abort=True)

        self.pipeline_steps = [
            PipelineStep(
                "clean",
                details="Cleaning HTML files",
                skip=skip_clean,
                is_llm_step=False,
                is_batch_step=False,
                use_batch_result=False,
                func=cleaner.execute,
                in_folder="html",
                out_folder="cleaned"
            ),
            PipelineStep(
                "detect_batch",
                details="Running detector in batch mode",
                skip=not batch_detect,
                is_llm_step=True,
                is_batch_step=True,
                use_batch_result=False,
                func=detector.prepare_batch,
                in_folder="cleaned",
                out_folder="accepted"
            ),
            PipelineStep(
                "detect",
                details="Running detector in single mode",
                skip=skip_detect,
                is_llm_step=True,
                is_batch_step=False,
                use_batch_result=batch_detect,
                func=detector.execute,
                in_folder="cleaned",
                out_folder="accepted"
            ),
            PipelineStep(
                "parse",
                details="Parsing HTML files",
                skip=skip_parse,
                is_llm_step=False,
                is_batch_step=False,
                use_batch_result=False,
                func=parser.execute,
                in_folder="accepted",
                out_folder="json"
            ),
            PipelineStep(
                "annotate_batch",
                details="Running annotator in batch mode",
                skip=not batch_annotate,
                is_llm_step=True,
                is_batch_step=True,
                use_batch_result=False,
                func=annotator.prepare_batch,
                in_folder="json",
                out_folder="annotated"
            ),
            PipelineStep(
                "annotate",
                details="Running annotator in single mode",
                skip=skip_annotate,
                is_llm_step=True,
                is_batch_step=False,
                use_batch_result=batch_annotate,
                func=annotator.execute,
                in_folder="json",
                out_folder="annotated"
            ),
            PipelineStep(
                "review_batch",
                details="Running reviewer in batch mode",
                skip=not batch_review,
                is_llm_step=True,
                is_batch_step=True,
                use_batch_result=False,
                func=annotator.prepare_batch,
                in_folder="annotated",
                out_folder="reviewed"
            ),
            PipelineStep(
                "review",
                details="Running reviewer in single mode",
                skip=skip_review,
                is_llm_step=True,
                is_batch_step=False,
                use_batch_result=batch_review,
                func=annotator.execute,
                in_folder="annotated",
                out_folder="reviewed"
            )
        ]

        self.clients = {
            'openai': None,
            'ollama': None,
            'deepseek': None,
            'anthropic': None
        }

        openai_models = list(api_openai.models.keys())
        ollama_models = list(api_ollama.models.keys())
        deepseek_models = list(api_deepseek.models.keys())
        anthropic_models = list(api_anthropic.models.keys())

        for i, step in enumerate(self.pipeline_steps):
            step_model = models.get(step.name)
            if not step.is_llm_step or step.skip:
                pass
            elif step_model is None and default_model is None:
                state_manager.raise_error(
                    f"Error: No model provided for step {step}. Please provide a default model with the -model "
                    f"argument or the model field in the configuration file or specify a model for each pipeline "
                    f"step with the -model-<step> arguments or the models field in the configuration file.", abort=True)
            else:
                if step_model is None:
                    # use the default model for the step if no specific model is provided
                    print(f"Using default model '{default_model}' for step {step}")
                    if default_model in openai_models:
                        print(f"Using OpenAI model '{default_model}' for step {step}")
                        step.model = api_openai.models[default_model]
                    elif default_model in deepseek_models:
                        print(f"Using DeepSeek model '{default_model}' for step {step}")
                        step.model = api_deepseek.models[default_model]
                    elif default_model in ollama_models:
                        print(f"Using Ollama model '{default_model}' for step {step}")
                        step.model = api_ollama.models[default_model]
                    elif default_model in anthropic_models:
                        print(f"Using Anthropic model '{default_model}' for step {step}")
                        step.model = api_anthropic.models[default_model]
                    else:
                        state_manager.raise_error(
                            f"Error: Invalid default model provided: {default_model}. Please provide a valid model from the "
                            f"following list: {openai_models + ollama_models + deepseek_models + anthropic_models}", abort=True)
                elif step_model in openai_models:
                    print(f"Using OpenAI model '{step_model}' for step {step.name}")
                    step.model = api_openai.models[step_model]
                elif step_model in ollama_models:
                    print(f"Using Ollama model '{step_model}' for step {step.name}")
                    step.model = api_ollama.models[step_model]
                elif step_model in deepseek_models:
                    print(f"Using DeepSeek model '{step_model}' for step {step.name}")
                    step.model = api_deepseek.models[step_model]
                elif step_model in anthropic_models:
                    print(f"Using Anthropic model '{step_model}' for step {step.name}")
                    step.model = api_anthropic.models[step_model]
                else:
                    state_manager.raise_error(
                        f"Error: Invalid model provided for step {step}: {step_model}. Please provide a valid model from the "
                        f"following list: {openai_models + ollama_models + deepseek_models + anthropic_models}", abort=True)

                # initialize the API client for the step
                step.client = self._init_api_client(
                        run_id=run_id,
                        api=step.model['api'],
                        model=step.model,
                        hostname=hostname
                    )

                # Check if the step's client supports batch processing
                if step.is_batch_step and step.client.supports_batch is False:
                    state_manager.raise_warning(f"Warning: The API client for step {step.name.replace('_batch', '')} ({step.model['api']}) does not support batch processing. Batch processing will be skipped for this step.")
                    step.skip = True
                    self.pipeline_steps[i + 1].use_batch_result = False

                # Check if the step's client needs to be set to parallel prompt mode
                if parallel_prompt and step.client.supports_parallel:
                    step.parallel_prompt = True

                # count all non-skipped pipeline steps
                self.pipeline_length = len([s for s in self.pipeline_steps if not s.skip])


    def _init_api_client(self, run_id: str, api: str, model: dict, hostname: str) -> ApiBase:
        if api == 'openai':
            from src.llm_connectors.api_openai import ApiOpenAI
            return ApiOpenAI(run_id)
        elif api == 'ollama':
            from src.llm_connectors.api_ollama import ApiOllama
            return ApiOllama(run_id, hostname)
        elif api == 'deepseek':
            from src.llm_connectors.api_deepseek import ApiDeepSeek
            return ApiDeepSeek(run_id)
        elif api == 'anthropic':
            from src.llm_connectors.api_anthropic import ApiAnthropic
            return ApiAnthropic(run_id, hostname=hostname)
        else:
            self.state_manager.raise_error(
                f"Error: Unknown API '{api}' specified in the configuration. Please provide a valid API.",
                abort=True
            )

    async def _process_file(
            self,
            pkg: str,
            in_folder: str,
            out_folder: str,
            step: str,
            client: ApiBase,
            model: dict,
            func: Callable[[str, str, str, str, str, BaseStateManager, ApiBase, dict, bool, bool], Awaitable[list[dict]]],
            use_batch_result: bool,
            parallel_prompt: bool,
    ):
        try:
            await func(self.run_id, pkg, step, in_folder, out_folder, self.state_manager, client, model, use_batch_result, parallel_prompt)
        except Exception as e:
            await self.state_manager.raise_error(error_message=str(e), abort=True)
            print(f"Error processing package {pkg} in step {step}.")
            print(f"Error: {sys.exc_info()}")
            with open(f"../../output/{self.run_id}/log/error.txt", 'a') as error_file:
                error_file.write(f"{pkg}: {step}\n")
            with open(f"../../output/error.txt", 'a') as global_error_file:
                global_error_file.write(f"{self.run_id} {pkg} {step}\n")

    async def _prepare_batch(
            self,
            step: str,
            in_folder: str,
            client: ApiBase,
            model: dict,
            func: Callable[[str, str, str, str, ApiBase, dict], list[dict]]
    ):
        print(f"Preparing batch for step {step}...")
        batch = []

        # get the current timestamp
        files = [f for f in os.listdir(in_folder) if
                 os.path.isfile(os.path.join(in_folder, f)) and not f.startswith('.')]
        packages = [os.path.splitext(f)[0] for f in files]
        package_count = len(packages)

        await self.state_manager.update_state(total_files=package_count)

        # process each file in the input folder
        for pkg_index, pkg in enumerate(packages):
            await self.state_manager.update_state(
                step_progress=pkg_index / package_count / 3,
                current_file=pkg
            )
            # generate the batch entries for the current package and append them to the batch
            entries = func(self.run_id, pkg, in_folder, step, client, model)
            if entries is not None:
                batch += entries

        # write the batch entries to a JSONL file
        jsonl = ''
        for entry in batch:
            jsonl += json.dumps(entry) + '\n'
        util.write_to_file(f"../../output/{self.run_id}/batch/{step}/batch_input.jsonl", jsonl)

    async def _run_batch(
            self,
            task: str,
            in_folder: str,
            client: ApiBase,
            model: dict,
            func: Callable[[str, str, str, str, ApiBase, dict], list[dict]]
    ):
        while True:
            # check the status of the batch
            batch_status = client.check_batch_status(task)

            if batch_status is None:
                # only start a new batch if there is no active batch for the given task
                await self._prepare_batch(task, in_folder, client, model, func)
                client.run_batch(task)
                await self.state_manager.update_state(
                    step_progress=0.5,
                    message=f"Running batch..."
                )
            elif batch_status.status == "completed":
                # retrieve the results of the batch if it is completed
                await self.state_manager.update_state(
                    step_progress=0.75,
                    message="Batch completed, retrieving results..."
                )
                client.get_batch_results(task)
                await self.state_manager.update_state(
                    step_progress=1,
                    message="Batch results retrieved successfully"
                )
                return True
            elif batch_status.status == "failed":
                # exit the function if the batch failed
                await self.state_manager.raise_error(
                    f"Error: Batch processing failed for task {task}. Please check the logs for more details.",
                    abort=True
                )
                return False
            else:
                # wait for one minute before checking the status again
                if batch_status.created_at:
                    elapsed_time = int((time.time() - batch_status.created_at.timestamp()) / 60)
                else:
                    elapsed_time = 0
                await self.state_manager.update_state(
                    message=f"Waiting for batch to complete... ({elapsed_time} minutes elapsed)"
                )
                time.sleep(60)

    async def execute(self):
        """Execute the pipeline asynchronously, updating state after each step."""

        # prepare the output folders for the given run id
        util.prepare_output(self.run_id, overwrite=False)

        # iterate over the processing steps and execute the corresponding function for each package
        for step_index, step in enumerate([s for s in self.pipeline_steps if not s.skip]):
            in_folder = f"../../output/{self.run_id}/policies/{step.in_folder}"
            out_folder = f"../../output/{self.run_id}/policies/{step.out_folder}"

            if step.client is not None:
                step.client.setup(step.name, step.model)

            await self.state_manager.update_state(
                status=PipelineStatus.RUNNING,
                current_step=step.name,
                progress=step_index / self.pipeline_length,
                step_progress=0,
                current_file="",
                total_files=0,
                step_details=step.details
            )

            step_name = step.name.replace('_batch', '')

            if step.is_batch_step:
                # run the batch processing function of this step for all packages
                await self._run_batch(step_name, in_folder, step.client, step.model, step.func)
            else:
                # List all files in the input folder
                files = [f for f in os.listdir(in_folder) if
                         os.path.isfile(os.path.join(in_folder, f)) and not f.startswith('.')]

                # Extract package names
                packages = [os.path.splitext(f)[0] for f in files]

                package_count = len(packages)

                # process each file in the input folder
                for pkg_index, pkg in enumerate(packages):
                    await self.state_manager.update_state(
                        step_progress=pkg_index / package_count,
                        current_file=pkg,
                        total_files=package_count
                    )
                    await self._process_file(
                        pkg=pkg,
                        in_folder=in_folder,
                        out_folder=out_folder,
                        step=step_name,
                        client=step.client,
                        model=step.model,
                        func=step.func,
                        use_batch_result=step.use_batch_result,
                        parallel_prompt=step.parallel_prompt
                    )

            if step.client is not None:
                step.client.close()

            await self.state_manager.update_state(
                step_progress=1,
                current_file=""
            )

        await self.state_manager.update_state(
            status=PipelineStatus.COMPLETED,
            current_step="",
            progress=1,
            step_progress=1,
            file_progress=1,
            current_file="",
            step_details="",
            message="Pipeline completed successfully"
        )