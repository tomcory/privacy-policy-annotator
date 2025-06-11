import json

from src import util
from src.llm_connectors.api_base import ApiBase
from src.pipeline_steps.pipeline_step import PipelineStep
from src.state_manager import BaseStateManager


class Annotator(PipelineStep):
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
            task='annotate',
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
        policy = util.load_policy_json(f'{self.in_folder}/{pkg}.json')
        if policy is None:
            return None

        output = []
        total_cost = 0
        total_time = 0

        if self.parallel_prompt:
            results, total_cost, total_time = self.client.prompt_parallel(
                pkg=pkg,
                task=self.task,
                user_msgs=[json.dumps(passage) for passage in policy],
                model=self.model,
                response_format='json_schema',
                max_tokens=8192
            )

            # iterate over each result and add the annotations to the policy
            for index, result in enumerate(results):
                try:
                    policy[index]['annotations'] = json.loads(result)['annotations']
                except json.JSONDecodeError as e:
                    policy[index]['annotations'] = []
                    await self.state_manager.raise_error(error_message=str(e))
                output.append(policy[index])
        else:
            # iterate over each passage in the policy and annotate it or retrieve the annotation from the batch result
            for index, passage in enumerate(policy):
                await self.state_manager.update_state(file_progress=index / len(policy))

                if self.is_batch_step:
                    result, cost, time = self.client.retrieve_batch_result_entry(
                        task=self.task,
                        entry_id=f"{self.run_id}_{self.task}_{pkg}_{index}",
                        batch_results_file=self.batch_results_file
                    )
                else:
                    result, cost, time = self.client.prompt(
                        pkg=pkg,
                        task=self.task,
                        model=self.model,
                        response_format='json_schema',
                        user_msg=json.dumps(passage),
                        max_tokens=8192
                    )

                total_cost += cost
                total_time += time

                # add the annotations to the policy
                try:
                    passage['annotations'] = json.loads(result)['annotations']
                except json.JSONDecodeError as e:
                    passage['annotations'] = []
                    await self.state_manager.raise_error(error_message=str(e))
                output.append(passage)

        await self.state_manager.update_state(
            file_progress=1,
            message=f"Cost: {total_cost:.2f} USD, Time: {total_time:.2f} seconds"
        )

        util.write_to_file(f"{self.out_folder}/{pkg}.json", json.dumps(output, indent=4))


    async def prepare_batch(self, pkg: str):
        policy = util.load_policy_json(f'{self.in_folder}/{pkg}.json')
        if policy is None:
            return None

        entries = []
        policy_length = len(policy)

        for index, passage in enumerate(policy):
            await self.state_manager.update_state(file_progress=index / policy_length)

            entry = self.client.prepare_batch_entry(
                pkg=pkg,
                task=self.task,
                model=self.model,
                user_msg=json.dumps(passage),
                response_format='json_schema',
                max_tokens=8192,
                entry_id=index
            )
            entries.append(entry)

        await self.state_manager.update_state(file_progress=1)

        return entries
