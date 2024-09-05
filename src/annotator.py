import json
import os
import logging

from src import util, api_wrapper
from src.api_wrapper import ApiWrapper


class Annotator:
    def __init__(self, run_id: str, pkg: str, llm_api: ApiWrapper, model: str, use_batch: bool = False):
        self.task = "annotator"
        self.model = model
        self.in_folder = f"output/{run_id}/json"
        self.out_folder = f"output/{run_id}/annotated"

        self.run_id = run_id
        self.pkg = pkg
        self.llm_api = llm_api
        self.use_batch = use_batch

    def execute(self):
        print(f"Annotating {self.pkg}...")
        logging.info(f"Annotating {self.pkg}...")

        policy = util.load_policy_json(self.run_id, self.pkg, 'json')
        if policy is None:
            return None

        output = []
        total_inference_time = 0

        for index, passage in enumerate(policy):
            if self.use_batch:
                result, inference_time = api_wrapper.retrieve_batch_result_entry(self.run_id, self.task, f"{self.run_id}_{self.task}_{self.pkg}_{index}")
            else:
                system_message = util.read_from_file(f'{os.path.join(os.getcwd(), "system-prompts/annotator_system_prompt.md")}')

                # TODO: check if the outputs are actually formatted correctly as JSON (sometimes whitespaces seem to be contained in the output)
                result, inference_time = self.llm_api.prompt(
                    run_id=self.run_id,
                    pkg=self.pkg,
                    task=self.task,
                    model=self.model,
                    system_msg=system_message,
                    user_msg=json.dumps(passage),
                    max_tokens=2048,
                    json_format=True
                )

                # retry once if the result JSON does not contain the key 'annotations'
                if 'annotations' not in result:
                    logging.warning(f"Retrying annotation for {self.pkg} due to missing 'annotations' key in result.")
                    result, inference_time = self.llm_api.prompt(
                        run_id=self.run_id,
                        pkg=self.pkg,
                        task=self.task,
                        model=self.model,
                        system_msg=system_message,
                        user_msg=json.dumps(passage),
                        max_tokens=2048,
                        json_format=True
                    )

            total_inference_time += inference_time
            try:
                passage = json.loads(result)
            except json.JSONDecodeError:
                print(result)
                raise json.JSONDecodeError
            output.append(passage)

        print(f"Annotation time: {total_inference_time} s\n")

        util.add_to_file(f"output/{self.run_id}/{self.model}_responses/processing_times_annotator.csv", f"{self.pkg},{total_inference_time}\n")

        util.write_to_file(f"output/{self.run_id}/annotated/{self.model}.{self.pkg}.json", json.dumps(output, indent=4))

    def execute_batched(self):
        print(f"Annotating {self.pkg} in batch mode...")
        logging.info(f"Annotating {self.pkg} in batch mode...")

        policy = util.load_policy_json(self.run_id, self.pkg, 'json')
        if policy is None:
            return None

        system_message = util.read_from_file(f'{os.path.join(os.getcwd(), "system-prompts/annotator_system_prompt.md")}')
        user_msgs = [json.dumps(passage) for passage in policy]

        result, total_inference_time = self.llm_api.prompt_parallel(
            run_id=self.run_id,
            pkg=self.pkg,
            task=self.task,
            model=self.model,
            system_msg=system_message,
            user_msgs=user_msgs,
            max_tokens=4096,
            context_window=7168,
            json_format=True
        )

        output = []
        for res in result:
            try:
                passage = json.loads(res)
            except json.JSONDecodeError:
                print(res)
                raise json.JSONDecodeError
            output.append(passage)

        print(f"Annotation time: {total_inference_time} s\n")

        util.add_to_file(f"output/{self.run_id}/{self.model}_responses/processing_times_annotator.csv", f"{self.pkg},{total_inference_time}\n")
        util.write_to_file(f"output/{self.run_id}/annotated/{self.model}.{self.pkg}.json", json.dumps(output, indent=4))

    def skip(self):
        print(">>> Skipping annotation for %s..." % self.pkg)
        logging.info("Skipping annotation for %s..." % self.pkg)
        util.copy_file(f"{self.in_folder}/{self.pkg}.json", f"{self.out_folder}/{self.pkg}.json")
