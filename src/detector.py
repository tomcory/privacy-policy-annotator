import os
import logging
import tiktoken
from bs4 import BeautifulSoup

from src import api_wrapper, util
from src.api_wrapper import ApiWrapper


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
    def __init__(self, run_id: str, pkg: str, llm_api: ApiWrapper, model: str, use_batch: bool = False):
        self.task = "detector"
        self.model = model
        self.in_folder = f"output/{run_id}/cleaned"
        self.out_folder = f"output/{run_id}/accepted"

        self.run_id = run_id
        self.pkg = pkg
        self.llm_api = llm_api
        self.use_batch = use_batch

    def execute(self) -> bool:
        print(f">>> Detecting whether {self.pkg} is a policy...")
        logging.info(f"Detecting whether {self.pkg} is a policy...")

        file_path = f"{self.in_folder}/{self.pkg}.html"

        try:
            policy = util.read_from_file(file_path)

            if self.use_batch:
                output, inference_time = api_wrapper.retrieve_batch_result_entry(self.run_id, self.task, f"{self.run_id}_{self.task}_{self.pkg}_0")
            else:
                system_message = util.read_from_file(f'{os.path.join(os.getcwd(), "system-prompts/detector_system_prompt.md")}')

                output, inference_time = self.llm_api.prompt(
                    run_id=self.run_id,
                    pkg=self.pkg,
                    task=self.task,
                    model=self.model,
                    system_msg=system_message,
                    user_msg=generate_excerpt(policy),
                    max_tokens=1,
                    context_window=1024,
                    json_format=False
                )

            output = output.strip().lower()

            # write the pkg and inference time to "output/inference_times_detector.csv"
            util.add_to_file(f"output/{self.run_id}/{self.model}_responses/inference_times_detector.csv", f"{self.pkg},{inference_time}\n")

            util.write_to_file(f"output/{self.run_id}/{self.model}_responses/detector/{self.pkg}.txt", output)

            print(f"Detector time {inference_time} s\n")

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

            return output == 'true'
        except Exception as e:
            print(f"Error cleaning {self.pkg}: {e}")
            logging.error(f"Error cleaning {self.pkg}: {e}", exc_info=True)
            util.write_to_file(f"output/{self.run_id}/log/failed.txt", self.pkg)

            return False

    def skip(self):
        print(">>> Skipping headline detection for %s..." % self.pkg)
        logging.info("Skipping headline detection for %s..." % self.pkg)
        util.copy_file(f"{self.in_folder}/{self.pkg}.html", f"{self.out_folder}/{self.pkg}.html")
