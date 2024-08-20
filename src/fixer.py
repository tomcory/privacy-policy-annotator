import os
import ast
import logging

from ollama import AsyncClient
from bs4 import BeautifulSoup, NavigableString

from src import api_wrapper, util


def replace_long_text_with_placeholders(soup: BeautifulSoup):
    # replace all text content longer than 10 words with a placeholder
    for element in soup.find_all():
        for child in element.children:
            if isinstance(child, NavigableString):
                words = child.split()
                if len(words) > 10:
                    child.replace_with('...')

    # replace the text content of all li elements with a placeholder
    for element in soup.find_all('li'):
        if len(list(element.children)) == 1:
            element.string = "..."

    # if a tag's next sibling is of the same type (e.g. li) and both tags only have '...' as text, remove this tag
    for element in soup.find_all():
        if element.next_sibling is not None and element.next_sibling.name == element.name:
            if element.string == "..." and element.next_sibling.string == "...":
                element.decompose()

    return soup

class Fixer:
    def __init__(self, run_id: str, pkg: str, ollama_client: AsyncClient, use_batch: bool = False):
        self.task = "fixer"
        self.model = api_wrapper.models[os.environ.get('FIXER_MODEL', 'llama8b')]
        self.in_folder = f"output/{run_id}/accepted"
        self.out_folder = f"output/{run_id}/fixed"

        self.run_id = run_id
        self.pkg = pkg
        self.ollama_client = ollama_client
        self.use_batch = use_batch

    def execute(self):
        print(f">>> Fixing {self.pkg}...")
        logging.info(f"Fixing {self.pkg}...")
        file_path = f"{self.in_folder}/{self.pkg}.html"

        policy = util.read_from_file(file_path)
        soup = BeautifulSoup(policy, 'html.parser')

        print("Identifying headlines...")
        mini_soup = replace_long_text_with_placeholders(BeautifulSoup(str(soup), 'html.parser'))
        headlines = self.identify_headlines(self.run_id, self.pkg, str(mini_soup), self.use_batch)

        if headlines is None:
            print(f"> No headlines found for {self.pkg}.")
            return soup

        print("Fixing headlines...")

        for headline in headlines:
            element = soup.find(lambda tag: headline[0].strip() in tag.get_text().strip() and tag.name not in ['li', 'td', 'th'])
            if element is not None:
                element.name = headline[2]
            else:
                logging.warning(f"Element not found! Headline: {headline}")
                print(f"Element not found! Headline: {headline[0]}, Current tag: {headline[1]}, H-tag: {headline[2]}")

        print(f"> Fixed {len(headlines)} headlines.")

        fixed_policy = soup.prettify()

        file_name = f"{self.out_folder}/{self.pkg}.html"
        util.write_to_file(file_name, fixed_policy)

    def identify_headlines(self, run_id: str, pkg: str, html: str, use_batch: bool = False) -> (str, float):
        logging.info(f"Identifying headlines for {pkg}...")

        if use_batch:
            print("Using batch result...")
            output, inference_time = api_wrapper.retrieve_batch_result_entry(run_id, self.task, f"{run_id}_{self.task}_{pkg}_0")
        else:
            with open(f'{os.path.join(os.getcwd(), "system-prompts/fixer_system_prompt.md")}', 'r') as f:
                system_message = f.read()

            # TODO: figure out why the LLM sometimes outputs additional text accompanying the list
            output, inference_time = api_wrapper.prompt(
                run_id=run_id,
                pkg=pkg,
                task=self.task,
                model=self.model,
                ollama_client=self.ollama_client,
                system_msg=system_message,
                user_msg=html,
                options={"max_tokens": 1024},
                json_format=False
            )

        if output is None:
            output = "ERROR"

        print(f"Inference time: {inference_time} s")

        # write the pkg and cost to "output/costs-detector.csv"
        with open(f"output/{run_id}/{self.model}_responses/inference_times_fixer.csv", "a") as f:
            f.write(f"{pkg},{inference_time}\n")

        with open(f"output/{run_id}/{self.model}_responses/fixer/{pkg}.txt", "w") as f:
            f.write(output)

        # parse "output" into a list of 3-tuples: (headline, current_tag, h_tag)
        # split the string into lines and then split each line by the comma character
        # h_tag is the last entry
        # current_tag the second to last entry
        # reconstruct the headline by joining the remaining entries with commas
        headlines = []

        if output == "ERROR":
            return headlines

        output = ast.literal_eval(output)

        for line in output:
            if line == "":
                continue
            try:
                headline, current_tag, h_tag = line.rsplit(",", 2)
                current_tag = current_tag.strip().replace("<", "").replace(">", "")
                h_tag = h_tag.strip().replace("<", "").replace(">", "")
                headlines.append((headline, current_tag, h_tag))
                logging.info(f"Appended new headline: {headline}, {current_tag}, {h_tag}")
                # print(f"Headline: {headline}, Current tag: {current_tag}, H-tag: {h_tag}")
            except Exception:
                print(f"Error parsing line: {line}")

        return headlines

    def skip(self):
        print(">>> Skipping fixing %s..." % self.pkg)
        logging.info("Skipping fixing %s..." % self.pkg)
        util.copy_folder(self.in_folder, self.out_folder)
