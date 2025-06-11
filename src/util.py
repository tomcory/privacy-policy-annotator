import json
import os
import shutil
import sys

import chardet

def _detect_encoding(file_path: str) -> str:
    """
    Detects the encoding of a file.
    Args:
        file_path (str): The path to the file.
    Returns:
        str: The detected encoding of the file.
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def write_to_file(file_name: str, text: str):
    """
    Writes text to a file, creating the file if it does not exist.
    Args:
        file_name (str): The name of the file to write to.
        text (str): The text to write to the file.
    """
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(text)


def append_to_file(file_name: str, text: str):
    """
    Appends text to a file, creating the file if it does not exist.
    Args:
        file_name (str): The name of the file to append to.
        text (str): The text to append to the file.
    """
    if os.path.exists(file_name):
        encoding = _detect_encoding(file_name)
    else:
        encoding = 'utf-8'

    with open(file_name, 'a', encoding=encoding) as file:
        file.write(text + '\n')


def read_from_file(file_path: str):
    """
    Reads a file and returns its content as a string, detecting the encoding automatically.
    Args:
        file_path (str): The path to the file to read.
    Returns:
        str | None: The content of the file as a string or None if the file does not exist.
    """
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist.")
        return None

    # Read the file with the detected encoding
    content = ''
    with open(file_path, 'r', encoding=_detect_encoding(file_path)) as file:
        for line in file:
            content += line
        return content

def read_json_file(file_path: str):
    """
    Reads a JSON file and returns its content.
    Args:
        file_path (str): The path to the JSON file.
    Returns:
        dict | None: The content of the JSON file as a dictionary or None if the file does not exist.
    """
    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r', encoding=_detect_encoding(file_path)) as file:
        return json.load(file)


def prepare_output(run_id: str, output_folder: str = '../../output', overwrite: bool = False):
    if overwrite and os.path.exists('output'):
        shutil.rmtree('output')

    path_list = [
        f'{output_folder}',
        f'{output_folder}/{run_id}',
        f'{output_folder}/{run_id}/log',
        f'{output_folder}/{run_id}/policies',
        f'{output_folder}/{run_id}/policies/html',
        f'{output_folder}/{run_id}/policies/cleaned',
        f'{output_folder}/{run_id}/policies/detected',
        f'{output_folder}/{run_id}/policies/detected/rejected',
        f'{output_folder}/{run_id}/policies/detected/unknown',
        f'{output_folder}/{run_id}/policies/json',
        f'{output_folder}/{run_id}/policies/annotated',
        f'{output_folder}/{run_id}/policies/reviewed',
        f'{output_folder}/{run_id}/batch',
        f'{output_folder}/{run_id}/batch/detect',
        f'{output_folder}/{run_id}/batch/annotate',
        f'{output_folder}/{run_id}/batch/review'
    ]

    for path in path_list:
        if not os.path.exists(path):
            os.mkdir(path)


def load_policy_json(file_path: str):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"The file {file_path} does not contain valid JSON.")
        return None


def log_prompt_result(
        run_id: str,
        task: str,
        pkg: str,
        model_name: str,
        output_format: str,
        cost: float,
        processing_time: float,
        outputs: list
):
    # create the output folder if it does not exist
    folder_path = f"../output/{run_id}/{model_name}_responses"
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(f"{folder_path}/{task}", exist_ok=True)

    # log the cost, processing time and response
    with open(f"{folder_path}/costs_{task}.csv", "a") as f:
        f.write(f"{pkg},{cost}\n")
    with open(f"{folder_path}/times_{task}.csv", "a") as f:
        f.write(f"{pkg},{processing_time}\n")

    if len(outputs) > 1:
        for i, output in enumerate(outputs):
            with open(f"{folder_path}/{task}/{pkg}_{i}.{output_format}", "a") as f:
                f.write(output + '\n')
    else:
        with open(f"{folder_path}/{task}/{pkg}.{output_format}", "a") as f:
            f.write(outputs[0] + '\n')


def prepare_prompt_messages(
        api: str,
        task: str,
        user_msg: str = None,
        system_msg: str = None,
        examples: list[tuple[str, str]] = None,
        response_schema: dict = None,
        bundle_system_msg: bool = True
) -> (list[dict], str, dict, int, int, int):
    """
    Prepare the prompt messages for the API call.
    Args:
        api (str): The API to use (e.g., 'anthropic', 'openai').
        task (str): The task for which the prompt is prepared.
        user_msg (str): The user message to include in the prompt.
        system_msg (str): The system message to include in the prompt.
        examples (list[tuple[str, str]]): A list of example pairs (user message, assistant message).
        response_schema (dict): The response schema to use for the prompt.
        bundle_system_msg (bool): Whether to bundle the system message with the examples.
    Returns:
        tuple: A tuple containing:
            - messages (list[dict]): The prepared messages for the API call.
            - system_msg (str): The system message used in the prompt.
            - response_schema (dict): The response schema used in the prompt.
            - system_len (int): The length of the system message.
            - user_len (int): The length of the user message.
            - example_len (int): The total length of all example messages.
    """
    if system_msg is None or examples is None or response_schema is None:
        sys_msg, exs, res_schema = _load_prompts_from_files(task, api)

        if system_msg is None:
            if sys_msg is None:
                raise ValueError(f"No system prompt specified for task '{task}' and API '{api}'.")
            else:
                system_msg = sys_msg
        if examples is None:
            examples = exs
        if response_schema is None:
            response_schema = res_schema

    # map the examples to the correct json format
    es = [(
        {"role": "user", "content": e[0]}, # (e[0].replace('\n', ''))
        {"role": "assistant", "content": e[1]} # (e[1].replace('\n', ''))
    ) for e in examples]

    examples = es

    # generate the messages list for the API call
    messages = []
    if bundle_system_msg:
        messages.append({"role": "system", "content": system_msg})
    for example in examples:
        messages.extend(example)
    if user_msg is not None:
        messages.append({"role": "user", "content": user_msg})

    return messages, system_msg, response_schema


def _load_prompts_from_files(task: str, api: str) -> (str, list[tuple[str, str]], dict):
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        prompts_folder = f"{dir_path}/../prompts/{api}/{task}"

        # get all file names in the prompts folder
        file_names = os.listdir(prompts_folder)

        system_prompt = None
        examples = {}
        response_schema = None
        for file_name in file_names:
            if file_name.startswith("response_schema."):
                # print(f"Loading response schema from {file_name}")
                response_schema = read_from_file(f"{prompts_folder}/{file_name}")
                # minify the json schema
                response_schema = json.loads(response_schema)
            elif file_name.startswith("system."):
                # print(f"Loading system prompt from {file_name}")
                system_prompt = read_from_file(f"{prompts_folder}/{file_name}")
                if file_name.endswith("json"):
                    system_prompt = json.dumps(json.loads(system_prompt))
            elif file_name.startswith("example_"):
                name_split = file_name.split("_")
                index = name_split[1]
                if examples.get(index) is None:
                    examples[index] = (None, None)
                file_type = name_split[-1].split(".")
                file_type = file_type[0]
                if file_type == "user":
                    # print(f"Loading user example from {file_name}")
                    msg = read_from_file(f"{prompts_folder}/{file_name}")
                    if file_name.endswith("json"):
                        msg = json.dumps(json.loads(msg))
                    examples[str(index)] = (msg, examples[str(index)][1])
                elif file_type == "assistant":
                    # print(f"Loading assistant example from {file_name}")
                    msg = read_from_file(f"{prompts_folder}/{file_name}")
                    if file_name.endswith("json"):
                        msg = json.dumps(json.loads(msg))
                    examples[str(index)] = (examples[str(index)][0], msg)
                else:
                    raise ValueError(f"Unknown file type in example: {file_type}. Expected 'user' or 'assistant'.")

        # parse examples to a list of tuples
        examples = [(user, assistant) for user, assistant in examples.values()]

        return system_prompt, examples, response_schema

    except Exception as e:
        print(f"Error loading prompts: {e}", file=sys.stderr)
        return None, [], None