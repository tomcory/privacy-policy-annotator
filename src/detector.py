import os
import logging
import tiktoken
from bs4 import BeautifulSoup
from ollama import AsyncClient

from src import api_wrapper, util


def generate_excerpt(html: str):
    encoding = tiktoken.get_encoding('cl100k_base')

    html = BeautifulSoup(html, 'html.parser').get_text()
    encoded_html = encoding.encode(html)

    # 200 tokens should be enough to determine whether the text is a privacy policy
    # get 200 tokens from the middle of the document to ensure that we have actual content
    encoded_length = len(encoded_html)
    excerpt_start = max(encoded_length // 2 - 100, 0)
    excerpt_end = min(encoded_length // 2 + 100, encoded_length)
    return encoding.decode(encoded_html[excerpt_start:excerpt_end])


class Detector:
    def __init__(self, run_id: str, pkg: str, ollama_client: AsyncClient, use_batch: bool = False):
        self.task = "detector"
        self.model = api_wrapper.models[os.environ.get('DETECTOR_MODEL', 'llama8b')]
        self.in_folder = f"output/{run_id}/cleaned"
        self.out_folder = f"output/{run_id}/accepted"

        self.run_id = run_id
        self.pkg = pkg
        self.ollama_client = ollama_client
        self.use_batch = use_batch

    def execute(self):
        print(f">>> Detecting whether {self.pkg} is a policy...")
        logging.info(f"Detecting whether {self.pkg} is a policy...")

        file_path = f"{self.in_folder}/{self.pkg}.html"

        try:
            policy = util.read_from_file(file_path)

            if self.use_batch:
                output, inference_time = api_wrapper.retrieve_batch_result_entry(self.run_id, self.task, f"{self.run_id}_{self.task}_{self.pkg}_0")
            else:
                with open(f'{os.path.join(os.getcwd(), "system-prompts/detector_system_prompt.md")}', 'r') as f:
                    system_message = f.read()

                output, inference_time = api_wrapper.prompt(
                    run_id=self.run_id,
                    pkg=self.pkg,
                    task=self.task,
                    model=self.model,
                    ollama_client=self.ollama_client,
                    system_msg=system_message,
                    user_msg=generate_excerpt(policy),
                    options={"max_tokens": 2048},
                    json_format=False
                )

            output = output.strip().lower()

            # write the pkg and inference time to "output/inference_times_detector.csv"
            with open(f"output/{self.run_id}/{self.model}_responses/inference_times_detector.csv", "a") as f:
                f.write(f"{self.pkg},{inference_time}\n")

            with open(f"output/{self.run_id}/{self.model}_responses/detector/{self.pkg}.txt", "w") as f:
                f.write(output)

            print(f"Detector time {inference_time} s.")

            # sort the output accordingly
            if output == 'true':
                print("It's a policy, saving to file in /output/accepted...")
                folder = "accepted"
            elif output == 'unknown':
                print("Unsure whether it's a policy, saving to file in /output/unknown...")
                folder = 'unknown'
            else:
                print("It's not a policy, saving to file in /output/rejected...")
                folder = 'rejected'

            file_name = f"output/{self.run_id}/{folder}/{self.pkg}.html"
            util.write_to_file(file_name, policy)
        except Exception as e:
            print(f"Error cleaning {self.pkg}: {e}")
            logging.error(f"Error cleaning {self.pkg}: {e}", exc_info=True)
            util.write_to_file(f"output/{self.run_id}/log/failed.txt", self.pkg)

    def skip(self):
        print(">>> Skipping headline detection for %s..." % self.pkg)
        logging.info("Skipping headline detection for %s..." % self.pkg)
        util.copy_folder(self.in_folder, self.out_folder)
