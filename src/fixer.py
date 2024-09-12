import os
import ast
import logging
import tiktoken

from bs4 import BeautifulSoup, NavigableString

from src import api_wrapper, util
from src.api_wrapper import ApiWrapper


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
    def __init__(self, run_id: str, pkg: str, llm_api: ApiWrapper, model: str, use_batch: bool = False):
        self.task = "fixer"
        self.model = model
        self.in_folder = f"output/{run_id}/accepted"
        self.out_folder = f"output/{run_id}/fixed"

        self.run_id = run_id
        self.pkg = pkg
        self.llm_api = llm_api
        self.use_batch = use_batch

    def execute(self):
        # print(f"\n>>> Fixing {self.pkg}...")
        logging.info(f"Fixing {self.pkg}...")
        file_path = f"{self.in_folder}/{self.pkg}.html"

        policy = util.read_from_file(file_path)
        soup = BeautifulSoup(policy, 'html.parser')

#         print("Identifying headlines...")
        mini_soup = replace_long_text_with_placeholders(BeautifulSoup(str(soup), 'html.parser'))

        token_count = len(tiktoken.get_encoding('cl100k_base').encode(str(mini_soup)))
        logging.info(f"Token count for headline fixing of package \"{self.pkg}\": {token_count}")
        if token_count > 5000:
            self.skip()
            logging.info("Skipping headline fixing because of high token count of policy")
            return

        headlines = self.identify_headlines(self.run_id, self.pkg, str(mini_soup), self.use_batch)

        if headlines is None:
#             print(f"> No headlines found for {self.pkg}.")
            return soup

#         print("Fixing headlines...")

        for headline in headlines:
            element = soup.find(lambda tag: headline[0].strip() in tag.get_text().strip() and tag.name not in ['li', 'td', 'th'])
            if element is not None:
                element.name = headline[2]
            else:
                logging.warning(f"Element not found! Headline: {headline}")
#                 print(f"Element not found! Headline: {headline[0]}, Current tag: {headline[1]}, H-tag: {headline[2]}")

#         print(f"> Fixed {len(headlines)} headlines.")

        fixed_policy = soup.prettify()

        file_name = f"{self.out_folder}/{self.pkg}.html"
        util.write_to_file(file_name, fixed_policy)

    def identify_headlines(self, run_id: str, pkg: str, html: str, use_batch: bool = False) -> (str, float):
        logging.info(f"Identifying headlines for {pkg}...")

        if use_batch:
#             print("Using batch result...")
            output, inference_time = api_wrapper.retrieve_batch_result_entry(run_id, self.task, f"{run_id}_{self.task}_{pkg}_0")
        else:
            system_message = util.read_from_file(f'{os.path.join(os.getcwd(), "system-prompts/fixer_system_prompt.md")}')

            output, inference_time = self.llm_api.prompt(
                run_id=run_id,
                pkg=pkg,
                task=self.task,
                model=self.model,
                system_msg=system_message,
                user_msg=html,
                max_tokens=2048,
                context_window=6144,
                json_format=False
            )
            try:
                # remove backticks since sometimes it will try to adhere to the system prompt a little too much
                output = output.replace("`", "")
                # only keep the output starting at the first opening square bracket and ending at the last closing square bracket
                output = output[output.find("[") + 1:output.rfind("]")].strip()
            except SyntaxError:
#                 print(f"Error parsing output of fixer: {output}")
                logging.error(f"Error parsing output of fixer: {output}")
                # retry once
                output, inference_time = self.llm_api.prompt(
                    run_id=run_id,
                    pkg=pkg,
                    task=self.task,
                    model=self.model,
                    system_msg=system_message,
                    user_msg=html,
                    max_tokens=2048,
                    context_window=6144,
                    json_format=False
                )
                try:
                    output = output.replace("`", "")
                    output = output[output.find("[") + 1:output.rfind("]")].strip()
                except Exception as e:
#                     print(f"Error parsing output of fixer: {e}")
                    logging.error(f"Error parsing output of fixer: {e}")
                    output = None
            except Exception as e:
#                 print(f"Error in fixer: {e}")
                logging.error(f"Error in fixer: {e}")
                output = None

        if output is None:
            output = "ERROR"

#         print(f"Headline detection time: {inference_time} s\n")
        logging.info(f"Headline detection time: {inference_time} s")

        # write the pkg and cost to "output/costs-detector.csv"
        util.add_to_file(f"output/{run_id}/{self.model}_responses/inference_times_fixer.csv", f"{pkg},{inference_time}\n")

        util.write_to_file(f"output/{run_id}/{self.model}_responses/fixer/{pkg}.txt", output)

        # parse "output" into a list of 3-tuples: (headline, current_tag, h_tag)
        # split the string into lines and then split each line by the comma character
        # h_tag is the last entry
        # current_tag the second to last entry
        # reconstruct the headline by joining the remaining entries with commas
        headlines = []

        if output == "ERROR":
            return headlines

        try:
            output = ast.literal_eval(output)
        except Exception as e:
#             print(f"Error parsing output of fixer: {e}")
            logging.error(f"Error parsing output of fixer: {e}", exc_info=True)
            return headlines

        for line in output:
            if line == "":
                continue
            try:
                headline, current_tag, h_tag = line.rsplit(",", 2)
                current_tag = current_tag.strip().replace("<", "").replace(">", "")
                h_tag = h_tag.strip().replace("<", "").replace(">", "")
                headlines.append((headline, current_tag, h_tag))
                logging.debug(f"Appended new headline: {headline}, {current_tag}, {h_tag}")
                # print(f"Headline: {headline}, Current tag: {current_tag}, H-tag: {h_tag}")
            except Exception:
#                 print(f"Error parsing line: {line}")
                logging.error(f"Error parsing line: {line}")
                continue

        return headlines

    def skip(self):
#         print("\n>>> Skipping fixing %s..." % self.pkg)
        logging.info("Skipping fixing %s..." % self.pkg)
        util.copy_file(f"{self.in_folder}/{self.pkg}.html", f"{self.out_folder}/{self.pkg}.html")
