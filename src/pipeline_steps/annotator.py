import json
from typing import Union

from src import util
from src.llm_connectors.api_ollama import ApiOllama
from src.llm_connectors.api_openai import ApiOpenAI
from src.state_manager import BaseStateManager


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
        parallel_prompt: bool = False,
):
    policy = util.load_policy_json(f'{in_folder}/{pkg}.json')
    if policy is None:
        return None

    output = []
    total_cost = 0
    total_time = 0

    if parallel_prompt:
        results, total_cost, total_time = client.prompt_parallel(
            pkg=pkg,
            task=task,
            user_msgs=[json.dumps(passage) for passage in policy],
            model=model,
            response_format='json_schema',
            max_tokens=8192
        )

        # iterate over each result and add the annotations to the policy
        for index, result in enumerate(results):
            try:
                policy[index]['annotations'] = json.loads(result)['annotations']
            except json.JSONDecodeError as e:
                policy[index]['annotations'] = []
                await state_manager.raise_error(error_message=str(e))
            output.append(policy[index])
    else:
        # iterate over each passage in the policy and annotate it or retrieve the annotation from the batch result
        for index, passage in enumerate(policy):
            await state_manager.update_state(file_progress=index / len(policy))

            if use_batch_result:
                result, cost, time = client.retrieve_batch_result_entry(task, f"{run_id}_{task}_{pkg}_{index}")
            else:
                result, cost, time = client.prompt(
                    pkg=pkg,
                    task=task,
                    model=model,
                    response_format='json_schema',
                    user_msg=json.dumps(passage),
                    max_tokens=8192
                )

            total_cost += cost
            total_time += time

            # add the annotations to the policy
            try:
                passage['annotations'] = json.loads(result)['annotations']
            except json.JSONDecodeError as e:
                passage['annotations'] = []
                await state_manager.raise_error(error_message=str(e))
            output.append(passage)

    await state_manager.update_state(
        file_progress=1,
        message=f"Cost: {total_cost:.2f} USD, Time: {total_time:.2f} seconds"
    )

    util.write_to_file(f"{out_folder}/{pkg}.json", json.dumps(output, indent=4))


def prepare_batch(
        run_id: str,
        pkg: str,
        task: str,
        in_folder: str,
        state_manager: BaseStateManager,
        client: Union[ApiOpenAI, ApiOllama],
        model: dict
):
    policy = util.load_policy_json(f'{in_folder}/{pkg}.json')
    if policy is None:
        return None

    entries = []
    policy_length = len(policy)

    for index, passage in enumerate(policy):
        state_manager.update_state(file_progress=index / policy_length)

        entry = client.prepare_batch_entry(
            pkg=pkg,
            task=task,
            model=model,
            user_msg=json.dumps(passage),
            response_format='json_schema',
            max_tokens=8192,
            entry_id=index
        )
        entries.append(entry)

    state_manager.update_state(file_progress=1)

    return entries
