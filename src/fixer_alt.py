import tiktoken
import markdown as md
from typing import Union

from src import util
from src.api_ollama import ApiOllama
from src import api_ollama


def execute(
        run_id: str,
        pkg: str,
        in_folder: str,
        out_folder: str,
        task: str,
        client: Union[ApiOllama],
        model: dict = None,
        use_batch_result: bool = False,
        parallel_prompt: bool = False
):
    print(f">>> Fixing {pkg}...")
    file_path = f"{in_folder}/{pkg}.html"

    policy = util.read_from_file(file_path)
    model = model or api_ollama.models.get('reader-lm0.5b')

    encoding = tiktoken.get_encoding(model.get('encoding'))
    input_token_count = len(encoding.encode(policy))
    print(f"Markdown conversion input token count: {input_token_count}")

    client.setup()
    markdown, cost, time = client.prompt(
        pkg=pkg,
        task=task,
        model=model,
        system_msg='',
        user_msg=policy,
        max_tokens=input_token_count,
        top_k=20,
        repeat_penalty=1.2,
        context_window=input_token_count + 5000,
    )

    output_token_count = len(encoding.encode(markdown))
    print(f"Markdown conversion output token count: {output_token_count}")

    model = api_ollama.models.get('llama3b')

    reduced_markdown, cost, time = client.prompt(
        pkg=pkg,
        task=task,
        model=model,
        system_msg='Given the markdown, remove all of the content in it that does not pertain directly to the policy, for example images, links, and other non-policy content.'
                   'Return the exact input for all of the content that is kept. Your output should be a markdown formatted string that only contains the policy content.'
                   'Do not provide any other output other than the reduced markdown. You may also provide the original markdown unchanged if you believe that the policy content is already isolated.'
                   'Conversely, if there is no policy content in the markdown, you should return an empty string.',
        user_msg=markdown,
        max_tokens=output_token_count,
        top_k=20,
        context_window=output_token_count + 5000,
    )

    html = md.markdown(reduced_markdown)

    util.write_to_file(f"{out_folder}/{pkg}.html", html)

def prepare_batch(
        pkg: str,
        in_folder: str,
        task: str,
        client: Union[ApiOllama],
        model: dict
):
    # Function required for pipeline compatibility but not used in this context
    return None


def _replace_long_text_with_placeholders(soup):
    # Function required for pipeline compatibility but not used in this context
    pass


def _identify_headlines(
        run_id: str,
        pkg: str,
        html: str,
        task: str,
        client: Union[ApiOllama],
        model: dict,
        use_batch: bool
):
    # Function required for pipeline compatibility but not used in this context
    return None
