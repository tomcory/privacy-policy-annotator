import json
import os
import logging

from ollama import AsyncClient

from src import util, api_wrapper


class Annotator:
    def __init__(self, run_id: str, pkg: str, ollama_client: AsyncClient, use_batch: bool = False):
        self.task = "annotator"
        self.model = api_wrapper.models[os.environ.get('LLM_MODEL', 'llama8b')]
        self.in_folder = f"output/{run_id}/json"
        self.out_folder = f"output/{run_id}/annotated"

        self.run_id = run_id
        self.pkg = pkg
        self.ollama_client = ollama_client
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
                with open(f'{os.path.join(os.getcwd(), "system-prompts/annotator_system_prompt.md")}', 'r') as f:
                    system_message = f.read()

                # TODO: check if the outputs are actually formatted correctly as JSON (sometimes whitespaces seem to be contained in the output)
                result, inference_time = api_wrapper.prompt(
                    run_id=self.run_id,
                    pkg=self.pkg,
                    task=self.task,
                    model=self.model,
                    ollama_client=self.ollama_client,
                    system_msg=system_message,
                    user_msg=json.dumps(passage),
                    options={"max_tokens": 2048},
                    json_format=True
                )

            total_inference_time += inference_time
            try:
                passage = json.loads(result)
            except json.JSONDecodeError:
                print(result)
                raise json.JSONDecodeError
            output.append(passage)

        print(f"Annotation time: {total_inference_time}")

        with open(f"output/{self.run_id}/{self.model}_responses/processing_times_annotator.csv", "a") as f:
            f.write(f"{self.pkg},{total_inference_time}\n")

        util.write_to_file(f"output/{self.run_id}/annotated/{self.model}.{self.pkg}.json", json.dumps(output, indent=4))

    def skip(self):
        print(">>> Skipping annotation for %s..." % self.pkg)
        logging.info("Skipping annotation for %s..." % self.pkg)
        util.copy_folder(self.in_folder, self.out_folder)
