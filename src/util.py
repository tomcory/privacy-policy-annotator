import json
import os
import shutil


def write_to_file(file_name: str, text: str):
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(text)


def read_from_file(file_path: str, encoding='utf-8'):
    content = ''
    with open(file_path, 'r', encoding=encoding) as file:
        for line in file:
            content += line
        return content


def prepare_output(run_id: str, overwrite: bool = False):
    if overwrite and os.path.exists('output'):
        shutil.rmtree('output')

    path_list = [
        'output',
        'output/%s' % run_id,
        'output/%s/log' % run_id,
        'output/%s/original' % run_id,
        'output/%s/cleaned' % run_id,
        'output/%s/accepted' % run_id,
        'output/%s/unknown' % run_id,
        'output/%s/rejected' % run_id,
        'output/%s/fixed' % run_id,
        'output/%s/json' % run_id,
        'output/%s/annotated' % run_id,
        'output/%s/reviewed' % run_id,
        'output/%s/batch' % run_id,
        'output/%s/batch/detector' % run_id,
        'output/%s/batch/fixer' % run_id,
        'output/%s/batch/annotator' % run_id,
        'output/%s/batch/reviewer' % run_id,
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
    folder_path = f"output/{run_id}/{model_name}_responses"
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(f"{folder_path}/{task}", exist_ok=True)

    # log the cost, processing time and response
    with open(f"output/{folder_path}/costs_{task}.csv", "a") as f:
        f.write(f"{pkg},{cost}\n")
    with open(f"output/{folder_path}/times_{task}.csv", "a") as f:
        f.write(f"{pkg},{processing_time}\n")

    if len(outputs) > 1:
        for i, output in enumerate(outputs):
            with open(f"output/{folder_path}/{task}/{pkg}_{i}.{output_format}", "a") as f:
                f.write(output + '\n')
    else:
        with open(f"output/{folder_path}/{task}/{pkg}.{output_format}", "a") as f:
            f.write(outputs[0] + '\n')