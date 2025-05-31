import json
import os
import shutil
import sys

import chardet


def write_to_file(file_name: str, text: str):
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(text)


def append_to_file(file_name: str, text: str):
    with open(file_name, 'a', encoding='utf-8') as file:
        file.write(text + '\n')


def read_from_file(file_path: str):
    # Detect the encoding of the file
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    # Read the file with the detected encoding
    content = ''
    with open(file_path, 'r', encoding=encoding) as file:
        for line in file:
            content += line
        return content


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
        f'{output_folder}/{run_id}/policies/accepted',
        f'{output_folder}/{run_id}/policies/rejected',
        f'{output_folder}/{run_id}/policies/unknown',
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
        bundle_system_msg: bool = True,
        schema_only: bool = False
) -> (list[dict], str, dict, int, int, int):

    sys_msg, exs, res_schema = _load_prompts_from_files(task, api, schema_only)

    if system_msg is None:
        system_msg = sys_msg
    if examples is None:
        examples = exs
    if response_schema is None:
        response_schema = res_schema

    if schema_only:
        return None, None, response_schema, 0, 0, 0

    if bundle_system_msg and system_msg is None:
        raise ValueError("system_msg must be provided, either as an argument or in the prompts folder")

    # map the examples to the correct json format
    es = [(
        {"role": "user", "content": e[0]}, # (e[0].replace('\n', ''))
        {"role": "assistant", "content": e[1]} # (e[1].replace('\n', ''))
    ) for e in examples]

    examples = es

    # sum the length of the example contents
    example_len = sum(len(e[0]['content']) + len(e[1]['content']) for e in examples)

    # generate the messages list for the API call
    messages = [{"role": "system", "content": system_msg}] # .replace('\n', '')}]
    for example in examples:
        messages.extend(example)

    if user_msg is not None:
        messages.append({"role": "user", "content": user_msg}) # .replace('\n', '')})

    # calculate the length of the input messages
    system_len = len(messages[0]['content'])
    user_len = len(messages[-1]['content'])

    return messages, system_msg, response_schema, system_len, user_len, example_len


def _load_prompts_from_files(task: str, api: str, schema_only = False) -> (str, list[tuple[str, str]], dict):
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
                if schema_only:
                    return None, None, response_schema
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
                    print(f"Error: unknown example type: {file_type}, each example must have a user and an assistant message", file=sys.stderr)

        # parse examples to a list of tuples
        examples = [(user, assistant) for user, assistant in examples.values()]

        return system_prompt, examples, response_schema

    except Exception as e:
        print(f"Error loading prompts: {e}", file=sys.stderr)
        return None, []