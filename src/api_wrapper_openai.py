import json

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI


models = {
    'gpt-3.5-turbo': {
        'name': 'gpt-3.5-turbo',
        'encoding': 'cl100k_base',
        'input_price': (0.5 / 1000000),
        'output_price': (1.5 / 1000000),
        'input_price_batch': (0.25 / 1000000),
        'output_price_batch': (0.75 / 1000000)
    },
    'gpt-4': {
        'name': 'gpt-4',
        'encoding': 'cl100k_base',
        'input_price': (30 / 1000000),
        'output_price': (60 / 1000000),
        'input_price_batch': (15 / 1000000),
        'output_price_batch': (30 / 1000000)
    },
    'gpt-4o': {
        'name': 'gpt-4o',
        'encoding': 'o200k_base',
        'input_price': (5 / 1000000),
        'output_price': (15 / 1000000),
        'input_price_batch': (2.5 / 1000000),
        'output_price_batch': (7.5 / 1000000)
    },
    'gpt-4o-mini': {
        'name': 'gpt-4o-mini',
        'encoding': 'o200k_base',
        'input_price': (0.15 / 1000000),
        'output_price': (0.6 / 1000000),
        'input_price_batch': (0.075 / 1000000),
        'output_price_batch': (0.3 / 1000000)
    }
}


def prompt(
        run_id: str,
        pkg: str,
        task: str,
        model: dict,
        system_msg: str,
        user_msg: str,
        examples: list[tuple[str, str]] = None,
        n: int = 1,
        max_tokens: int = 2048
) -> (str, float):

    # load the OPENAI_API_KEY environment variable from the .env file and create an OpenAI client with it
    load_dotenv()
    client = OpenAI()
    encoding = tiktoken.get_encoding(model['encoding'])

    # calculate the length of the input messages
    system_len = len(encoding.encode(system_msg))
    user_len = len(encoding.encode(user_msg))
    example_len = sum(len(encoding.encode(example[0])) + len(encoding.encode(example[1])) for example in examples)

    messages = prepare_messages(system_msg, user_msg, examples)

    # configure and query GPT
    completion = client.chat.completions.create(
        model=model['name'],
        n=n,
        max_tokens=max_tokens,
        messages=messages
    )

    # extract the output text from the response message
    output = completion.choices[0].message.content
    output_len = len(encoding.encode(output))

    # calculate the cost of the API call based on the total number of tokens used
    cost = (system_len + user_len + example_len) * model['input_price'] + output_len * model['output_price']

    if run_id is not None and pkg is not None and task is not None:
        # log the cost and response
        with open(f"output/{run_id}/gpt-responses/costs-{task}.csv", "a") as f:
            f.write(f"{pkg},{cost}\n")
        with open(f"output/{run_id}/gpt-responses/{task}/{pkg}.txt", "w") as f:
            f.write(output)

    return output, cost


def prepare_batch_entry(
        run_id: str,
        pkg: str,
        task: str,
        model: dict,
        system_msg: str,
        user_msg: str,
        examples: list[tuple[str, str]] = None,
        n: int = 1,
        max_tokens: int = 2048,
        entry_id: int = 0
):
    return {
        "custom_id": "%s_%s_%s_%s" % (run_id, task, pkg, entry_id),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model['name'],
            "n": n,
            "max_tokens": max_tokens,
            "messages": prepare_messages(system_msg, user_msg, examples)
        }
    }


def prepare_messages(system_msg: str, user_msg: str, examples: list[tuple[str, str]] = None):
    if examples is None:
        examples = []

    # map the examples to the correct json format
    examples = [({"role": "user", "content": e[0]}, {"role": "assistant", "content": e[1]}) for e in examples]

    # generate the messages list for the API call
    messages = [{"role": "system", "content": system_msg}]
    for example in examples:
        messages.extend(example)
    messages.append({"role": "user", "content": user_msg})

    return messages


def run_batch(run_id, task: str):
    load_dotenv()
    client = OpenAI()

    batch_input_file = client.files.create(
        file=open(f"output/{run_id}/batch/{task}/batch_input.jsonl", "rb"),
        purpose="batch"
    )

    batch_metadata = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "fixing headlines"
        }
    )

    batch_metadata_json = json.dumps(json.loads(batch_metadata.model_dump_json()), indent=4)

    # write the batch metadata to a file
    with open(f"output/{run_id}/batch/{task}/batch_metadata.json", "w") as f:
        f.write(batch_metadata_json)


def retrieve_batch_result_entry(run_id, task: str, entry_id: str, batch_results_file: str = "batch_results.jsonl"):
    # iterate through all lines of the batch-results.jsonl file and return the one with the matching custom_id as a dict
    with open(f"output/{run_id}/batch/{task}/{batch_results_file}", "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"Error decoding line: {line}")
                raise json.JSONDecodeError
            if entry['custom_id'] == entry_id:
                # check for errors in the response
                if entry['error'] is not None:
                    print(f"Error for entry {entry_id}: {entry['error']}")
                    return None, 0
                elif entry['response']['status_code'] != 200:
                    print(f"Bad status code for entry {entry_id}: {entry['response']['status_code']}")
                    return None, 0
                elif entry['response']['body']['choices'] is None or len(entry['response']['body']['choices']) == 0:
                    print(f"No choices for entry {entry_id}")
                    return None, 0
                elif entry['response']['body']['choices'][0]['finish_reason'] != "stop":
                    print(f"Finish reason not 'stop' for entry {entry_id}: {entry['response']['body']['choices'][0]['finish_reason']}")
                    return None, 0
                # return the content of the first choice
                else:
                    prompt_cost = entry['response']['body']['usage']['prompt_tokens'] * models['gpt-4o-mini']['input_price'] / 2
                    completion_cost = entry['response']['body']['usage']['completion_tokens'] * models['gpt-4o-mini']['output_price'] / 2
                    output = entry['response']['body']['choices'][0]['message']['content']

                    return output, prompt_cost + completion_cost

    print(f"Entry {entry_id} not found")
    return None, 0


def check_batch_status(run_id, task: str, batch_metadata_file: str = "batch_metadata.json"):
    load_dotenv()
    client = OpenAI()

    with open(f"output/{run_id}/batch/{task}/{batch_metadata_file}", "r") as f:
        batch_metadata = json.load(f)

    batch_status = client.batches.retrieve(batch_metadata['id'])

    batch_status_json = json.dumps(json.loads(batch_status.model_dump_json()), indent=4)

    # write the batch status to a file
    with open(f"output/{run_id}/batch/{task}/{batch_metadata_file}", "w") as f:
        f.write(f"{batch_status_json}")

    if batch_status.status == "completed":
        print("Batch completed, getting results...")
    else:
        print("Batch not completed yet, status: %s" % batch_status.status)

    return batch_status


def get_batch_results(run_id, task: str, batch_metadata_file: str = "batch_metadata.json"):
    load_dotenv()
    client = OpenAI()

    with open(f"output/{run_id}/batch/{task}/{batch_metadata_file}", "r") as f:
        batch_metadata = json.load(f)

    batch_results = client.files.content(batch_metadata['output_file_id']).text

    with open(f"output/{run_id}/batch/{task}/batch_results.jsonl", "w") as f:
        f.write(batch_results)

    return batch_results