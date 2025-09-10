import os
import os
import sys
import time
from typing import Optional

from src import util
from src.llm_connectors import api_openai, api_ollama, api_deepseek, api_anthropic
from src.llm_connectors.api_anthropic import ApiAnthropic
from src.llm_connectors.api_base import ApiBase
from src.llm_connectors.api_deepseek import ApiDeepSeek
from src.llm_connectors.api_ollama import ApiOllama
from src.llm_connectors.api_openai import ApiOpenAI
from src.pipeline_steps.annotator import Annotator
from src.pipeline_steps.reviewer import Reviewer
from src.pipeline_steps.metadata_crawler import MetadataCrawler
from src.pipeline_steps.policy_crawler import PolicyCrawler
from src.pipeline_steps.cleaner import Cleaner
from src.pipeline_steps.llm_classifier import LLMClassifier
from src.pipeline_steps.detector import Detector
from src.pipeline_steps.parser import Parser
from src.pipeline_steps.pipeline_step import PipelineStep
from src.pipeline_steps.targeted_annotator import TargetedAnnotator
from src.state_manager import PipelineStatus, BaseStateManager


class PipelineExecutor:
    def __init__(
            self,
            run_id: str,
            pkg: str,
            default_model: str,
            models: dict,
            crawl_retries: int = 2,
            skip_crawl: bool = False,
            skip_clean: bool = False,
            skip_detect: bool = False,
            skip_parse: bool = False,
            skip_annotate: bool = False,
            skip_review: bool = False,
            skip_classify: bool = False,
            skip_targeted_annotate: bool = False,
            batch_detect: bool = False,
            batch_annotate: bool = False,
            batch_review: bool = False,
            batch_classify: bool = False,
            batch_targeted_annotate: bool = False,
            parallel_prompt: bool = False,
            hostname: str = None,
            state_manager: BaseStateManager = None,
            use_two_step_annotation: bool = False,
            confidence_threshold: float = 0.3,
            use_rag: bool = True,
            max_examples_per_requirement: int = 2,
            min_example_confidence: float = 0.8,
            annotated_folder: str = None,
            use_opp_115: bool = False,
            root_dir: str = "./output/"
    ):
        if not run_id and not pkg:
            raise ValueError("No run_id or pkg provided")
        
        if not run_id and pkg:
            run_id = pkg

        if not default_model and not models:
            raise ValueError("Neither default model nor step-specific models provided")

        self.run_id = run_id
        self.pkg = pkg
        self.default_model = default_model
        self.models = models
        self.crawl_retries = crawl_retries

        # parse pipeline step flags
        self.skip_crawl = skip_crawl
        self.skip_clean = skip_clean
        self.skip_detect = skip_detect
        self.skip_parse = skip_parse
        self.skip_annotate = skip_annotate
        self.skip_review = skip_review
        self.skip_classify = skip_classify
        self.skip_targeted_annotate = skip_targeted_annotate
        self.batch_detect = batch_detect
        self.batch_annotate = batch_annotate
        self.batch_review = batch_review
        self.batch_classify = batch_classify
        self.batch_targeted_annotate = batch_targeted_annotate

        self.parallel_prompt = parallel_prompt
        self.hostname = hostname
        self.state_manager = state_manager
        self.use_two_step_annotation = use_two_step_annotation
        self.confidence_threshold = confidence_threshold
        self.use_rag = use_rag
        self.max_examples_per_requirement = max_examples_per_requirement
        self.min_example_confidence = min_example_confidence
        self.annotated_folder = annotated_folder
        self.use_opp_115 = use_opp_115

        # create the output folders
        self.root_folder = os.path.join(self.root_dir, self.run_id)
        self.policies_folder = os.path.join(self.root_folder, "policies")
        self.batch_folder = os.path.join(self.root_folder, "batch")
        self.log_folder = os.path.join(self.root_folder, "log")

        # create the error log file
        self.error_log_file = os.path.join(self.root_folder, "error.txt")

        # create the batch filenames
        self.batch_input_filename = "batch_input.json"
        self.batch_metadata_filename = "batch_metadata.json"
        self.batch_results_filename = "batch_results.json"
        self.batch_errors_filename = "batch_errors.json"

        self.setup_performed = False
        self.pipeline_steps = []


    async def setup(self):
        """Setup the pipeline steps"""
        if self.setup_performed:
            return
        else:
            self.setup_performed = True

        # start the state manager
        await self.state_manager.start()
        
        # prepare the output folders for the given run id
        util.prepare_output(self.run_id, overwrite=False)

        # if a single package is provided, write it to the pkgs folder for the crawler to find it
        if self.pkg:
            util.write_to_file(os.path.join(self.policies_folder, "pkgs", f"{self.pkg}.txt"), self.pkg)

            # Pipeline setup
            await self._setup_pipeline_steps()
            await self._setup_pipeline_clients()

            # count all non-skipped pipeline steps
            self.pipeline_length = len([s for s in self.pipeline_steps if not s.skip])

    async def execute(self):
        
        """Execute the pipeline asynchronously, updating state after each step."""

        # setup the pipeline steps (will only execute once, so we can call it as many times as we want)
        await self.setup()

        # iterate over the processing steps and execute the corresponding function for each package
        for step_index, step in enumerate([s for s in self.pipeline_steps if not s.skip]):

            await self.state_manager.update_state(
                status=PipelineStatus.RUNNING,
                current_step=step.task,
                progress=step_index / self.pipeline_length,
                step_progress=0,
                current_file="",
                total_files=0,
                step_details=step.details
            )

            # List all files in the input folder
            files = [f for f in os.listdir(step.in_folder) if
                     os.path.isfile(os.path.join(step.in_folder, f)) and not f.startswith('.')]

            # Extract and count package names
            packages = [os.path.splitext(f)[0] for f in files]
            package_count = len(packages)

            # if necessary, run the batch processing first
            if step.is_batch_step:
                await self._run_batch(step, packages)

            # process each file in the input folder
            for pkg_index, pkg in enumerate(packages):
                await self.state_manager.update_state(
                    step_progress=pkg_index / package_count,
                    current_file=pkg,
                    total_files=package_count
                )

                await self._process_file(step, pkg)

            await self.state_manager.update_state(
                step_progress=1,
                current_file=""
            )

        total_cost = sum(step.client.total_cost for step in self.pipeline_steps if step.client)
        print("--------------------------------")
        print(f"Total cost: {total_cost}")
        print("--------------------------------")

        # close clients after the pipeline completes
        for step in self.pipeline_steps:
            if step.client:
                step.client.close()

        # update the state
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

    async def _setup_pipeline_steps(self):
        """Setup the pipeline steps"""
        self.pipeline_steps = []
        self.pipeline_steps = [
            MetadataCrawler(
                run_id=self.run_id,
                state_manager=self.state_manager,
                in_folder=os.path.join(self.policies_folder, "pkgs"),
                out_folder=os.path.join(self.policies_folder, "metadata"),
                skip=self.skip_crawl
            ),
            PolicyCrawler(
                run_id=self.run_id,
                state_manager=self.state_manager,
                in_folder=os.path.join(self.policies_folder, "metadata"),
                out_folder=os.path.join(self.policies_folder, "html"),
                skip=self.skip_crawl
            ),
            Cleaner(
                run_id=self.run_id,
                state_manager=self.state_manager,
                in_folder=os.path.join(self.policies_folder, "html"),
                out_folder=os.path.join(self.policies_folder, "cleaned"),
                skip=self.skip_clean
            ),
            Detector(
                run_id=self.run_id,
                state_manager=self.state_manager,
                in_folder=os.path.join(self.policies_folder, "cleaned"),
                out_folder=os.path.join(self.policies_folder, "detected"),
                skip=self.skip_detect,
                is_batch_step=self.batch_detect,
                parallel_prompt=self.parallel_prompt
            ),
            Parser(
                run_id=self.run_id,
                state_manager=self.state_manager,
                in_folder=os.path.join(self.policies_folder, "detected"),
                out_folder=os.path.join(self.policies_folder, "json"),
                skip=self.skip_parse
            )
        ]

        if self.use_two_step_annotation:
            self.pipeline_steps.append(
                LLMClassifier(
                    run_id=self.run_id,
                    state_manager=self.state_manager,
                    in_folder=os.path.join(self.policies_folder, "json"),
                    out_folder=os.path.join(self.policies_folder, "classified"),
                    skip=self.skip_classify,
                    is_batch_step=self.batch_classify,
                    parallel_prompt=self.parallel_prompt,
                    use_opp_115=self.use_opp_115
                )
            )
            self.pipeline_steps.append(
                TargetedAnnotator(
                        run_id=self.run_id,
                        state_manager=self.state_manager,
                        in_folder=os.path.join(self.policies_folder, "classified"),
                        out_folder=os.path.join(self.policies_folder, "annotated"),
                        skip=self.skip_annotate,
                        is_batch_step=self.batch_targeted_annotate,
                        parallel_prompt=self.parallel_prompt,
                        confidence_threshold=self.confidence_threshold,
                        use_rag=True,
                        max_examples_per_requirement=self.max_examples_per_requirement,
                        min_example_confidence=self.min_example_confidence,
                        use_opp_115=self.use_opp_115
                )
            )
        
        else:
            self.pipeline_steps.append(
                Annotator(
                        run_id=self.run_id,
                        state_manager=self.state_manager,
                        in_folder=os.path.join(self.policies_folder, "json"),
                        out_folder=os.path.join(self.policies_folder, "annotated"),
                        skip=self.skip_annotate,
                        is_batch_step=self.batch_annotate,
                        parallel_prompt=self.parallel_prompt,
                        use_opp_115=self.use_opp_115
                    )   
                )

        self.pipeline_steps.append(
            Reviewer(
                run_id=self.run_id,
                state_manager=self.state_manager,
                in_folder=os.path.join(self.policies_folder, "annotated"),
                out_folder=os.path.join(self.policies_folder, "reviewed"),
                skip=self.skip_review,
                is_batch_step=self.batch_review,
                parallel_prompt=self.parallel_prompt,
                use_rag=True,
                max_examples_per_requirement=self.max_examples_per_requirement,
                min_example_confidence=self.min_example_confidence,
                use_opp_115=self.use_opp_115
            )
        )
        
    async def _setup_pipeline_clients(self):
        """Setup the pipeline clients"""
        
        # Model selection and client setup
        self.clients: dict[str, Optional[ApiBase]] = {
            'anthropic': None,
            'deepseek': None,
            'ollama': None,
            'openai': None
        }

        anthropic_models = list(api_anthropic.available_models.keys())
        deepseek_models = list(api_deepseek.available_models.keys())
        ollama_models = list(api_ollama.available_models.keys())
        openai_models = list(api_openai.available_models.keys())

        for i, step in enumerate(self.pipeline_steps):
            if step.is_llm_step and not step.skip:

                step_model = self.models.get(step.task)
                selected_model = None

                # check whether a step-specific model or default model was provided
                if step_model is None:
                    if self.default_model is None:
                        await self.state_manager.raise_error(
                            f"Error: No model provided for step {step}. Please provide a default model with the -model "
                            f"argument or the model field in the configuration file or specify a model for each pipeline "
                            f"step with the -model-<step> arguments or the models field in the configuration file.",
                            abort=True)
                    else:
                        step_model = self.default_model

                # find the specified model in the pool of available models
                if step_model in openai_models:
                    print(f"Using OpenAI model '{step_model}' for step {step.task}")
                    selected_model = api_openai.available_models[self.default_model]
                elif step_model in deepseek_models:
                    print(f"Using DeepSeek model '{step_model}' for step {step.task}")
                    selected_model = api_deepseek.available_models[self.default_model]
                elif step_model in ollama_models:
                    print(f"Using Ollama model '{step_model}' for step {step.task}")
                    selected_model = api_ollama.available_models[self.default_model]
                elif step_model in anthropic_models:
                    print(f"Using Anthropic model '{step_model}' for step {step.task}")
                    selected_model = api_anthropic.available_models[self.default_model]
                else:
                    await self.state_manager.raise_error(
                        f"Error: Invalid default model provided: {step_model}. Please provide a valid model from the "
                        f"following list: {openai_models + ollama_models + deepseek_models + anthropic_models}",
                        abort=True)
                    return

                # model found, store it in the step's instance
                step.model = selected_model['name']

                # initialize the API client for the step
                if selected_model['api'] == 'anthropic':
                    if self.clients['anthropic'] is None:
                        self.clients['anthropic'] = ApiAnthropic(self.run_id, self.hostname, use_opp_115=self.use_opp_115)
                    step.client = self.clients['anthropic']
                elif selected_model['api'] == 'deepseek':
                    if self.clients['deepseek'] is None:
                        self.clients['deepseek'] = ApiDeepSeek(self.run_id, self.hostname, use_opp_115=self.use_opp_115)
                    step.client = self.clients['deepseek']
                elif selected_model['api'] == 'ollama':
                    if self.clients['ollama'] is None:
                        self.clients['ollama'] = ApiOllama(self.run_id, self.hostname, use_opp_115=self.use_opp_115)
                    step.client = self.clients['ollama']
                elif selected_model['api'] == 'openai':
                    if self.clients['openai'] is None:
                        self.clients['openai'] = ApiOpenAI(self.run_id, self.hostname, use_opp_115=self.use_opp_115)
                    step.client = self.clients['openai']
                else:
                    await self.state_manager.raise_error(
                        f"Error: Unknown API '{selected_model['api']}' specified in the configuration. Please provide a valid API.",
                        abort=True
                    )

                # Check if the step's client supports batch processing
                if step.is_batch_step and (step.client.supports_batch is False):
                    await self.state_manager.raise_warning(
                        f"Warning: The API client for step {step.task} ({selected_model['api']}) does not support batch processing. Disabling batch processing for this step.")
                    step.is_batch_step = False

                # Check if the step's client supports parallel prompting
                if self.parallel_prompt and step.client.supports_parallel:
                    step.parallel_prompt = True
                elif self.parallel_prompt and (step.client.supports_parallel is False):
                    await self.state_manager.raise_warning(
                        f"Warning: The API client for step {step.task} ({selected_model['api']}) does not support parallel prompting. Disabling parallel prompting for this step.")
                    step.parallel_prompt = False

    async def _process_file(self, step: PipelineStep, pkg: str):
        try:
            await step.execute(pkg)
        except Exception as e:
            print(f"Error processing package {pkg} in step {step}.")
            print(f"Error: {sys.exc_info()}")
            with open(self.error_log_file, 'a') as error_log_file: #TODO: replace with logger
                error_log_file.write(f"{pkg}: {step}\n")
            await self.state_manager.raise_error(error_message=str(e), abort=True)

    async def _run_batch(self, step: PipelineStep, packages: list[str]):
        while True:
            # check the status of the batch
            try:
                batch_status = step.check_batch_status()
            except FileNotFoundError:
                batch_status = None

            # only start a new batch if there is no active batch for the given task
            if batch_status is None:
                await self._prepare_batch(step, packages)
                step.run_batch()
                await self.state_manager.update_state(current_file="Running batch...")

            # retrieve the results of the batch if it is completed
            elif batch_status.status == "completed":
                await self.state_manager.update_state(current_file="Batch finalizing...")
                step.get_batch_results()
                await self.state_manager.update_state(current_file="Batch complete")
                return True

            # exit the function if the batch failed
            elif batch_status.status == "failed":
                await self.state_manager.raise_error(
                    f"Error: Batch processing failed for task {step.task}. Please check the logs for more details.",
                    abort=True
                )
                return False

            # wait for one minute before checking the status again
            else:
                progress = batch_status.completed_requests / max(1, batch_status.total_requests)
                await self.state_manager.update_state(
                    step_progress=progress,
                    total_files=batch_status.total_requests,
                )
                if batch_status.created_at:
                    elapsed_time = int((time.time() - batch_status.created_at.timestamp()) / 60) #TODO: use elapsed time
                else:
                    elapsed_time = 0
                time.sleep(60) #TODO: replace with configurable polling interval

    async def _prepare_batch(self, step: PipelineStep, packages: list[str]):
        batch_entries = []
        package_count = len(packages)
        await self.state_manager.update_state(total_files=package_count)

        # process each file in the input folder
        for pkg_index, pkg in enumerate(packages):
            await self.state_manager.update_state(
                step_progress=pkg_index / package_count
            )
            # generate the batch entries for the current package and append them to the batch
            pkg_entries = await step.prepare_batch(pkg)
            if pkg_entries is not None:
                batch_entries += pkg_entries

        step.create_batch_input_file(batch_entries)