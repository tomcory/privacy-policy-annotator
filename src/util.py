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
        'output/%s/gpt_responses' % run_id,
        'output/%s/gpt_responses/detector' % run_id,
        'output/%s/gpt_responses/fixer' % run_id,
        'output/%s/gpt_responses/annotator' % run_id,
        'output/%s/gpt_responses/reviewer' % run_id
    ]

    for path in path_list:
        if not os.path.exists(path):
            os.mkdir(path)


def load_policy_json(run_id: str, pkg: str, task: str):
    file_path = 'output/%s/%s/%s.json' % (run_id, task, pkg)
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"The file {file_path} does not contain valid JSON.")
        return None