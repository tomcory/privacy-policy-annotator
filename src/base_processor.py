import json
import os
import logging
from src import util, api_wrapper
from src.api_wrapper import ApiWrapper

class BaseProcessor:
    def __init__(self, run_id: str, pkg: str, llm_api: ApiWrapper, model: str, task: str, use_batch: bool = False):
        self.task = task
        self.model = model
        self.in_folder = f"output/{run_id}/json"
        self.out_folder = f"output/{run_id}/annotated" if task == "annotator" else f"output/{run_id}/reviewed"

        self.run_id = run_id
        self.pkg = pkg
        self.llm_api = llm_api
        self.system_prompt_file = f"system-prompts/{task}_system_prompt.md"
        self.use_batch = use_batch

    def execute(self):
        # print(f">>>{'Annotat' if self.task == 'annotator' else 'Review'}ing {self.pkg}...")
        logging.info(f"{'Annotat' if self.task == 'annotator' else 'Review'}ing {self.pkg}...")

        policy = util.load_policy_json(self.run_id, self.pkg, 'json')
        if policy is None:
            return None

        output = []
        total_inference_time = 0

        for index, passage in enumerate(policy):
            if self.use_batch:
                result, inference_time = api_wrapper.retrieve_batch_result_entry(self.run_id, self.task, f"{self.run_id}_{self.task}_{self.pkg}_{index}")
            else:
                system_message = util.read_from_file(f'{os.path.join(os.getcwd(), self.system_prompt_file)}')

                result, inference_time = self.llm_api.prompt(
                    run_id=self.run_id,
                    pkg=self.pkg,
                    task=self.task,
                    model=self.model,
                    system_msg=system_message,
                    user_msg=json.dumps(passage),
                    max_tokens=2048,
                    context_window=8192,
                    json_format=True
                )

                if ('annotations' if self.task == 'annotator' else 'revised') not in result:
                    logging.warning(f"Retrying {self.task} for {self.pkg} due to missing key in result.")
                    result, inference_time = self.llm_api.prompt(
                        run_id=self.run_id,
                        pkg=self.pkg,
                        task=self.task,
                        model=self.model,
                        system_msg=system_message,
                        user_msg=json.dumps(passage),
                        max_tokens=2048,
                        context_window=8192,
                        json_format=True
                    )

            total_inference_time += inference_time
            try:
                passage = json.loads(result)
            except json.JSONDecodeError:
                logging.error(f"Error parsing result for {self.pkg} at index {index}: {result}", exc_info=True)
                raise json.JSONDecodeError
            output.append(passage)

#         print(f"{self.task.capitalize()} time: {total_inference_time} s\n")

        util.add_to_file(f"output/{self.run_id}/{self.model}_responses/processing_times_{self.task}.csv", f"{self.pkg},{total_inference_time}\n")

        util.write_to_file(self.out_folder + f"/{self.model}.{self.pkg}.json", json.dumps(output, indent=4))

    def execute_parallel(self):
#         print(f">>>{'Annotat' if self.task == 'annotator' else 'Review'}ing {self.pkg} in batch mode...")
        logging.info(f"{'Annotat' if self.task == 'annotator' else 'Review'}ing {self.pkg} in batch mode...")

        policy = util.load_policy_json(self.run_id, self.pkg, 'json')
        if policy is None:
            return None

        system_message = util.read_from_file(f'{os.path.join(os.getcwd(), self.system_prompt_file)}')
        user_msgs = [json.dumps(passage) for passage in policy]

        result, total_inference_time = self.llm_api.prompt_parallel(
            run_id=self.run_id,
            pkg=self.pkg,
            task=self.task,
            model=self.model,
            system_msg=system_message,
            user_msgs=user_msgs,
            max_tokens=4096,
            context_window=8192,
            json_format=True
        )

        output = []
        for res in result:
            try:
                passage = json.loads(res)
            except json.JSONDecodeError:
#                 print(res)
                logging.error(res, exc_info=True)
                raise json.JSONDecodeError
            output.append(passage)

#         print(f"{self.task.capitalize()} time: {total_inference_time} s\n")
        logging.info(f"{self.task.capitalize()} time: {total_inference_time} s")

        util.add_to_file(f"output/{self.run_id}/{self.model}_responses/processing_times_{self.task}.csv", f"{self.pkg},{total_inference_time}\n")
        util.write_to_file(self.out_folder + f"/{self.model}.{self.pkg}.json", json.dumps(output, indent=4))

    def skip(self):
#         print(f"\n>>> Skipping {self.task} for {self.pkg}...")
        logging.info(f"Skipping {self.task} for {self.pkg}...")
        util.copy_file(f"{self.in_folder}/{self.pkg}.json", f"{self.out_folder}/{self.pkg}.json")