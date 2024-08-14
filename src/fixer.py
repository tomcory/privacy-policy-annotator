from bs4 import BeautifulSoup, NavigableString

from src import api_wrapper, util


system_msg = '''Your task is to analyze a simplified HTML document of a privacy policy, identify headlines 
that do not have the correct tag (e.g. <p> instead of <h2>), and determine the correct h* tag for the 
headline. Some headlines may already have a <h*> tag, but of the wrong level (e.g. <h4> instead of <h3>; 
determine the correct level for these headlines. To shorten the input, the content of all elements with more 
than ten words has been replaced with "[...]"; disregard these elements. Thoroughly consider whether a given 
HTML element is a headline, as some elements are simply short sentences, table cells or list items. Make sure 
not to include these non-headlines! Output a list of all headlines, their current HTML tag, and the h-tag the 
headline should have in the context of the given document (h1, h2, h3, 54, h5, h6). Format your response as a 
list where each entry is on a new line formatted as "<headline>,<current tag of the element>,<h*-tag>". Do 
not number the entries.'''.replace('\n', '')

task = "fixer"
model = api_wrapper.models['gpt-4o']


def execute(run_id: str, pkg: str, in_folder: str, out_folder: str, use_batch: bool = False):
    print(f">>> Fixing {pkg}...")
    file_path = f"{in_folder}/{pkg}.html"

    policy = util.read_from_file(file_path)
    soup = BeautifulSoup(policy, 'html.parser')

    print("Identifying headlines...")
    mini_soup = replace_long_text_with_placeholders(BeautifulSoup(str(soup), 'html.parser'))
    headlines = identify_headlines(run_id, pkg, str(mini_soup), use_batch)

    if headlines is None:
        print(f"> No headlines found for {pkg}.")
        return soup

    print("Fixing headlines...")

    for headline in headlines:
        # find the element in the soup that corresponds to the headline and is not a li, td or th element
        # strip the element's text content and compare it to the headline pattern
        element = soup.find(lambda tag: tag.get_text().strip() == headline[0].strip() and tag.name not in ['li', 'td', 'th'])
        if element is not None:
            # replace the element's tag with the correct one
            element.name = headline[2]
        else:
            print(f"Element not found! Headline: {headline[0]}, Current tag: {headline[1]}, H-tag: {headline[2]}")

    print(f"> Fixed {len(headlines)} headlines.")

    fixed_policy = soup.prettify()

    file_name = f"{out_folder}/{pkg}.html"
    util.write_to_file(file_name, fixed_policy)


def prepare_batch(run_id: str, pkg: str, in_folder: str):
    html = util.read_from_file(f"{in_folder}/{pkg}.html")
    if html is None or html == "":
        return None

    soup = BeautifulSoup(html, 'html.parser')
    mini_soup = replace_long_text_with_placeholders(BeautifulSoup(str(soup), 'html.parser'))

    batch_entry = api_wrapper.prepare_batch_entry(
        model=model,
        system_msg=system_msg,
        user_msg=str(mini_soup),
        run_id=run_id,
        pkg=pkg,
        task=task
    )

    return [batch_entry]


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


def identify_headlines(run_id: str, pkg: str, html: str, use_batch: bool = False) -> (str, float):

    if use_batch:
        print("Using batch result...")
        output, cost = api_wrapper.retrieve_batch_result_entry(run_id, task, f"{run_id}_{task}_{pkg}_0")
    else:
        output, cost = api_wrapper.prompt(
            run_id=run_id,
            pkg=pkg,
            task=task,
            model=model,
            system_msg=system_msg,
            user_msg=html
        )

    if output is None:
        output = "ERROR"

    print(f"Cost: {cost}€")

    # write the pkg and cost to "output/costs-detector.csv"
    with open(f"output/{run_id}/gpt_responses/costs_fixer.csv", "a") as f:
        f.write(f"{pkg},{cost}\n")

    with open(f"output/{run_id}/gpt_responses/fixer/{pkg}.txt", "w") as f:
        f.write(output)

    # parse "output" into a list of 3-tuples: (headline, current_tag, h_tag)
    # split the string into lines and then split each line by the comma character
    # h_tag is the last entry
    # current_tag the second to last entry
    # reconstruct the headline by joining the remaining entries with commas
    headlines = []

    if output == "ERROR":
        return headlines

    for line in output.split("\n"):
        if line == "":
            continue
        try:
            headline, current_tag, h_tag = line.rsplit(",", 2)
            current_tag = current_tag.strip().replace("<", "").replace(">", "")
            h_tag = h_tag.strip().replace("<", "").replace(">", "")
            headlines.append((headline, current_tag, h_tag))
            # print(f"Headline: {headline}, Current tag: {current_tag}, H-tag: {h_tag}")
        except Exception:
            print(f"Error parsing line: {line}")

    return headlines
