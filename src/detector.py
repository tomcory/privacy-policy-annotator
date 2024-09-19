from typing import Union

import tiktoken
from bs4 import BeautifulSoup

from src import util
from src.api_ollama import ApiOllama
from src.api_openai import ApiOpenAI

system_msg = """Your task is to analyze a given text snippet and determine if the excerpt is likely part of a 
    privacy policy. Respond with only one word: 'true' if the excerpt seems to be from a privacy policy, 'false' if 
    it likely is not, and 'unknown' if there's not enough information to decide. Do not provide any additional 
    explanations or context in your response.""".replace('\n', '')


def execute(
        run_id: str,
        pkg: str,
        in_folder: str,
        out_folder: str,
        task: str,
        client: Union[ApiOpenAI, ApiOllama],
        model: dict = None,
        use_batch_result: bool = False,
        use_parallel: bool = False
):
    print(f">>> Detecting whether {pkg} is a policy...")
    file_path = f"{in_folder}/{pkg}.html"

    try:
        policy = util.read_from_file(file_path)

        if use_batch_result:
            output, cost, time = client.retrieve_batch_result_entry(run_id, task, f"{run_id}_{task}_{pkg}_0")
        else:
            output, cost, time = client.prompt(
                pkg=pkg,
                task=task,
                model=model,
                system_msg=system_msg,
                user_msg=_generate_excerpt(policy, model),
                max_tokens=1
            )

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
        util.write_to_file(f"output/{run_id}/log/failed.txt", pkg)


def prepare_batch(
        pkg: str,
        in_folder: str,
        task: str,
        client: Union[ApiOpenAI, ApiOllama],
        model: dict
):
    html = util.read_from_file(f"{in_folder}/{pkg}.html")
    if html is None or html == '':
        return None

    batch_entry = client.prepare_batch_entry(
        pkg=pkg,
        task=task,
        model=model,
        system_msg=system_msg,
        user_msg=str(_generate_excerpt(html, model))
    )

    return [batch_entry]


def _generate_excerpt(html: str, model: dict):
    encoding = tiktoken.get_encoding(model['encoding'])

    html = BeautifulSoup(html, 'html.parser').get_text()
    encoded_html = encoding.encode(html)

    # 200 tokens should be enough to determine whether the text is a privacy policy
    # get 200 tokens from the middle of the document to ensure that we have actual content
    encoded_length = len(encoded_html)
    excerpt_start = max(encoded_length // 2 - 100, 0)
    excerpt_end = min(encoded_length // 2 + 100, encoded_length)
    return encoding.decode(encoded_html[excerpt_start:excerpt_end])
