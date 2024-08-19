import os
import tiktoken
import logging
from bs4 import BeautifulSoup

from src import api_wrapper, util

system_msg = """Your task is to analyze a given text snippet and determine if the excerpt is likely part of a 
    privacy policy. Respond with only one word: 'true' if the excerpt seems to be from a privacy policy, 'false' if 
    it likely is not, and 'unknown' if there's not enough information to decide. Do not provide any additional 
    explanations or context in your response.""".replace('\n', '')

task = "detector"
model = api_wrapper.models[os.environ.get('DETECTOR_MODEL', 'llama8b')]


def execute(run_id: str, pkg: str, in_folder: str, out_folder: str, use_batch: bool = False):
    print(f">>> Detecting whether {pkg} is a policy...")
    file_path = f"{in_folder}/{pkg}.html"

    try:
        policy = util.read_from_file(file_path)

        if use_batch:
            output, inference_time = api_wrapper.retrieve_batch_result_entry(run_id, task, f"{run_id}_{task}_{pkg}_0")
        else:
            with open(f'{os.path.join(os.getcwd(), "system-prompts/detector_system_prompt.md")}', 'r') as f:
                system_message = f.read()

            output, inference_time = api_wrapper.prompt(
                run_id=run_id,
                pkg=pkg,
                task=task,
                model=model,
                system_msg=system_message,
                user_msg=generate_excerpt(policy),
                max_tokens=1,
                json_format=False
            )

        output = output.strip().lower()

        # write the pkg and inference time to "output/inference_times_detector.csv"
        with open(f"output/{run_id}/{model}_responses/inference_times_detector.csv", "a") as f:
            f.write(f"{pkg},{inference_time}\n")

        with open(f"output/{run_id}/{model}_responses/detector/{pkg}.txt", "w") as f:
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

        file_name = f"output/{run_id}/{folder}/{pkg}.html"
        util.write_to_file(file_name, policy)
    except Exception as e:
        print(f"Error cleaning {pkg}: {e}")
        logging.error(f"Error cleaning {pkg}: {e}", exc_info=True)
        util.write_to_file(f"output/{run_id}/log/failed.txt", pkg)


def prepare_batch(run_id: str, pkg: str, in_folder: str):
    html = util.read_from_file(f"{in_folder}/{pkg}.html")
    if html is None or html == '':
        return None

    batch_entry = api_wrapper.prepare_batch_entry(
        run_id=run_id,
        pkg=pkg,
        task=task,
        model=model,
        system_msg=system_msg,
        user_msg=str(generate_excerpt(html))
    )

    return [batch_entry]


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
