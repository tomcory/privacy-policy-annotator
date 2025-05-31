import json
import math
import os
from typing import Union

import tiktoken
from bs4 import BeautifulSoup
from py3langid.langid import LanguageIdentifier, MODEL_FILE

from src import util
from src.llm_connectors import api_openai, api_ollama
from src.llm_connectors.api_ollama import ApiOllama
from src.llm_connectors.api_openai import ApiOpenAI
from src.state_manager import BaseStateManager

accepted_languages = ['en']


async def execute(
        run_id: str,
        pkg: str,
        task: str,
        in_folder: str,
        out_folder: str,
        state_manager: BaseStateManager,
        client: Union[ApiOpenAI, ApiOllama],
        model: dict = None,
        use_batch_result: bool = False,
        use_parallel: bool = False
):
    try:
        html = util.read_from_file(f"{in_folder}/{pkg}.html")
        if html is None:
            return None

        if use_batch_result:
            output, cost, time = client.retrieve_batch_result_entry(task, f"{run_id}_{task}_{pkg}_0")
        elif False:
            text = BeautifulSoup(html, 'html.parser').get_text()
            language, confidence = identify_language(pkg, text)

            # log the language detection result
            util.append_to_file(
                f"../output/{run_id}/log/language.jsonl",
                json.dumps({'pkg': pkg, 'language': language, 'confidence': float(confidence)})
            )
            if confidence < 0.9:
                path = f"../output/{run_id}/policies/langid/unsure"
                if not os.path.exists(path):
                    os.mkdir(path)
                file_name = f"{path}/{pkg}.html"
                util.write_to_file(file_name, html)
            else:
                # copy the file to the other_languages folder
                # check whether the folder exists, if not, create it
                path = f"../output/{run_id}/policies/langid/{language}"
                if not os.path.exists(path):
                    os.mkdir(path)
                file_name = f"{path}/{pkg}.html"
                util.write_to_file(file_name, html)
            # if the html is empty or contains the string "privacy policy" (case-insensitive), return None
            if language in accepted_languages:
                # if 'privacy policy' in text.lower():
                #     # copy the file to the accepted folder
                #     file_name = f"../output/{run_id}/policies/accepted/{pkg}.html"
                #     util.write_to_file(file_name, html)
                #     return None

                if html is None or html == '':
                    return None

                output, cost, time = client.prompt(
                    pkg=pkg,
                    task=task,
                    model=model,
                    user_msg=_generate_excerpt(text, model),
                    max_tokens=1
                )
            else:
                return None
        else:
            text = BeautifulSoup(html, 'html.parser').get_text()
            if html is None or html == '':
                return None
            output, cost, time = client.prompt(
                pkg=pkg,
                task=task,
                model=model,
                user_msg=_generate_excerpt(text, model),
                max_tokens=1
            )

        # sort the output accordingly
        if output == 'true':
            folder = "accepted"
        elif output == 'unknown':
            folder = 'unknown'
        else:
            folder = 'rejected'

        file_name = f"../../output/{run_id}/policies/{folder}/{pkg}.html"
        util.write_to_file(file_name, html)

        await state_manager.update_state(file_progress=0.5)
    except Exception as e:
        await state_manager.raise_error(error_message=str(e))
        util.write_to_file(f"../../output/{run_id}/log/failed_detect.txt", pkg)
        return None


async def prepare_batch(
        run_id: str,
        pkg: str,
        task: str,
        in_folder: str,
        state_manager: BaseStateManager,
        client: Union[ApiOpenAI, ApiOllama],
        model: dict
):
    try:
        html = util.read_from_file(f"{in_folder}/{pkg}.html")
        if html is None:
            return None

        text = BeautifulSoup(html, 'html.parser').get_text()

        if False:
            language, confidence = identify_language(pkg, text)

            # log the language detection result
            util.append_to_file(
                f"../output/{run_id}/log/language.jsonl",
                json.dumps({'pkg': pkg, 'language': language, 'confidence': float(confidence)})
            )
            if confidence < 0.9:
                path = f"../output/{run_id}/policies/langid/unsure"
                if not os.path.exists(path):
                    os.mkdir(path)
                file_name = f"{path}/{pkg}.html"
                util.write_to_file(file_name, html)
            else:
                # copy the file to the other_languages folder
                # check whether the folder exists, if not, create it
                path = f"../output/{run_id}/policies/langid/{language}"
                if not os.path.exists(path):
                    os.mkdir(path)
                file_name = f"{path}/{pkg}.html"
                util.write_to_file(file_name, html)
        # if the html is empty or contains the string "privacy policy" (case-insensitive), return None
        # if language in accepted_languages:
        #     if 'privacy policy' in text.lower():
        #         # copy the file to the accepted folder
        #         file_name = f"../output/{run_id}/policies/accepted/{pkg}.html"
        #         util.write_to_file(file_name, html)
        #         return None
        elif True:
            if html is None or html == '':
                return None

            # Convert model name to model config if needed
            if isinstance(model, str):
                if model in api_openai.models:
                    model = api_openai.models[model]
                elif model in api_ollama.models:
                    model = api_ollama.models[model]
                else:
                    raise ValueError(f"Unknown model: {model}")

            batch_entry = client.prepare_batch_entry(
                pkg=pkg,
                task=task,
                model=model,
                user_msg=str(_generate_excerpt(html, model)),
                max_tokens=1
            )

            await state_manager.update_state(file_progress=0.5)
            return [batch_entry]
        else:
            return None
    except Exception as e:
        await state_manager.raise_error(error_message=str(e))
        return None


def identify_language(pkg: str, text: str):
    identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
    language, confidence = identifier.classify(text)
    return language, confidence


def detect_policy_simple(pkg: str, text: str):
    # check if the first quarter of the text contains the string "privacy policy" (case-insensitive)
    excerpt = text[:math.ceil(len(text) / 4)].lower()
    if 'privacy policy' in excerpt:
        return True


def _generate_excerpt(text: str, model: dict):
    encoding = tiktoken.get_encoding(model['encoding'])
    encoded_text = encoding.encode(text)

    # 100 tokens should be enough to determine whether the text is a privacy policy
    # get 100 tokens from around the 1/3 mark of the document to ensure that we have actual content
    encoded_length = len(encoded_text)
    excerpt_start = max(encoded_length // 3 - 150, 0)
    excerpt_end = min(encoded_length // 3 + 150, encoded_length)
    return encoding.decode(encoded_text[excerpt_start:excerpt_end])
