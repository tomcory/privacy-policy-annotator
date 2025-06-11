import re

from bs4 import NavigableString, Comment, BeautifulSoup, Tag

from src import util
from src.llm_connectors.api_base import ApiBase
from src.pipeline_steps.pipeline_step import PipelineStep
from src.state_manager import BaseStateManager


class Cleaner(PipelineStep):
    def __init__(
            self,
            run_id: str,
            skip: bool,
            in_folder: str,
            out_folder: str,
            state_manager: BaseStateManager,
            batch_input_file: str = "batch_input.json",
            batch_metadata_file: str = "batch_metadata.json",
            batch_results_file: str = "batch_results.jsonl",
            batch_errors_file: str = "batch_errors.jsonl",
            is_batch_step: bool = False,
            parallel_prompt: bool = False,
            model: str = None,
            client: ApiBase = None
    ):
        super().__init__(
            run_id=run_id,
            task='clean',
            details='',
            skip=skip,
            is_llm_step=False,
            is_batch_step=is_batch_step,
            in_folder=in_folder,
            out_folder=out_folder,
            state_manager=state_manager,
            batch_input_file=batch_input_file,
            batch_metadata_file=batch_metadata_file,
            batch_results_file=batch_results_file,
            batch_errors_file=batch_errors_file,
            parallel_prompt=parallel_prompt,
            model=model,
            client=client
        )

        self.search_strings = ['.*nav.*', '.*header.*', '.*footer.*', '.*[cC]ookie.*', '.*banner.*', '.*popup.*',
                          '.*toolbar.*', '.*modal.*', '.*dialog.*', '.*overlay.*', '.*consent.*', '.*dropdown.*',
                          '.*onetrust.*', '.*didomi.*', '.*country-selector.*']

        self.unwanted_tags = ['head', 'title', 'meta', 'link', 'style', 'script', 'noscript', 'iframe', 'object', 'nav',
                         'footer']

        self.inline_tags = ['a', 'abbr', 'acronym', 'b', 'bdi', 'bdo', 'big', 'button', 'center',
                       'cite', 'code', 'data', 'del', 'dfn', 'em', 'font', 'i', 'img', 'input',
                       'ins', 'kbd', 'label', 'map', 'mark', 'meter', 'noscript', 'object',
                       'output', 'picture', 'progress', 'q', 'ruby', 's', 'samp', 'script',
                       'select', 'small', 'span', 'strong', 'sub', 'sup', 'svg', 'template',
                       'textarea', 'time', 'tt', 'u', 'var', 'wbr']

        self.highlight_tags = ['b', 'strong', 'i', 'em', 'big', 'u']

        self.headline_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']

        self.page_source = None
        self.soup = None

    async def execute(self, pkg: str ):
        file_path = f"{self.in_folder}/{pkg}.html"

        self.page_source = util.read_from_file(file_path)

        self.simplify_soup()

        if self.soup is not None:
            cleaned = self.soup.prettify()

            if len(cleaned) > 0:
                util.write_to_file(f"{self.out_folder}/{pkg}.html", cleaned)
            else:
                with open(f"../output/{self.run_id}/log/empty.txt", 'a') as empty_file: #TODO: replace with logger
                    empty_file.write(f"{self.run_id} {pkg} clean\n")
                with open(f"../output/empty.txt", 'a') as global_empty_file: #TODO: replace with logger
                    global_empty_file.write(f"{self.run_id} {pkg} clean\n")
        else:
            #TODO: implement error handling/logging for empty soup
            pass

    async def prepare_batch(self, pkg: str):
        raise NotImplementedError("This is not an LLM step, batch processing is not supported.")


    def simplify_soup(self):
        """
            Extract the policy text from the page source using BeautifulSoup.
            """

        self.soup = BeautifulSoup(self.page_source, 'html.parser')
        self.remove_surrounding_structure()
        self.unwrap_inline_elements()
        self.replace_br_tags()
        self.parse_pre_tags()
        self.replace_special_characters()
        self.remove_attributes()
        self.concatenate_text()
        self.wrap_nonterminal_text()
        self.flatten_structure()
        self.remove_empty_elements()

    def remove_surrounding_structure(self):

        # print('Removing surrounding structure...')

        # remove all comments
        comments = self.soup.find_all(string=lambda text: isinstance(text, Comment))
        i = 0
        for comment in comments:
            comment.extract()
            i += 1
        # print('> Removed %d comments.' % i)

        # remove all unwanted tags
        i = 0
        for tag in self.unwanted_tags:
            for element in self.soup.find_all(tag):
                element.decompose()
                i += 1

        # remove all tags that have classes, roles or ids related to nav, header, footer, modals, banners, etc.
        removal_counter = 0
        for s in self.search_strings:
            regex = re.compile(pattern=s, flags=re.I)
            match_classes = self.soup.find_all("div", {"class": regex})
            match_classes.reverse()
            for match_class in match_classes:
                # print('Removing element based on class %s' % match_class['class'])
                match_class.decompose()
                removal_counter += 1
                pass
            match_roles = self.soup.find_all("div", {"role": regex})
            match_roles.reverse()
            for match_role in match_roles:
                # print('Removing element based on role %s' % match_role['role'])
                match_role.decompose()
                removal_counter += 1
                pass
            match_ids = self.soup.find_all('div', id=regex)
            match_ids.reverse()
            for match_id in match_ids:
                # print('Removing element based on id %s' % match_id['id'])
                match_id.decompose()
                removal_counter += 1
                pass

        # if there's a main element, reduce the soup to that
        policy = self.soup.find("main")
        if policy is None:
            # if there's an article element, reduce the soup to that
            policy = self.soup.find("article")
            if policy is None:
                # some sites use the class 'article' instead of the tag itself
                policy = self.soup.find("div", {"class": re.compile('.*[aA]rticle.*')})
                if policy is None:
                    # some sites use the role 'article' instead of the tag itself
                    policy = self.soup.find("div", {"role": re.compile('.*[aA]rticle.*')})
                    if policy is None:
                        # some sites use the id 'article' instead of the tag itself
                        policy = self.soup.find("div", id=re.compile('.*[aA]rticle.*'))
                        if policy is None:
                            # some sites use the class 'main' instead of the tag itself
                            policy = self.soup.find("div", {"class": re.compile('.*[mM]ain.*')})
                            if policy is None:
                                # some sites use the role 'main' instead of the tag itself
                                policy = self.soup.find("div", {"role": re.compile('.*[mM]ain.*')})
                                if policy is None:
                                    # print('> No main container found.')
                                    pass
                                else:
                                    # print('Main (class) found, reducing soup...')
                                    pass
                            else:
                                # print('Main (role) found, reducing soup...')
                                pass
                        else:
                            # print('Article (id) found, reducing soup...')
                            pass
                    else:
                        # print('Article (class) found, reducing soup...')
                        pass
                else:
                    # print('Article (role) found, reducing soup...')
                    pass
            else:
                # print('Article (tag) found, reducing soup...')
                pass
        else:
            # print('Main (tag) found, reducing soup...')
            pass

        if policy is not None:
            # remove all elements of the soup except the policy container and its children
            self.soup = BeautifulSoup(str(policy), 'html.parser')


    def unwrap_inline_elements(self):
        # Find all headline tags (h1 through h6)
        headlines = self.soup.find_all(self.headline_tags)

        # Determine the highest-order headline tag
        least_significant_headline = min(5, max((int(tag.name[1]) for tag in headlines), default=1))

        pseudo_headline_counter = 0
        inline_counter = 0

        # unwrap all inline tags as defined by the list above, transforming pseudo-headlines into proper headline tags
        for tag in self.inline_tags:
            for inline_element in self.soup.find_all(tag):
                if False and len(inline_element.parent.contents) == 1 and inline_element.name in self.highlight_tags and \
                        inline_element.parent.name not in headlines:
                    replacement_tag = 'h' + str(least_significant_headline + 1)
                    h4_element = self.soup.new_tag(replacement_tag)
                    h4_element.string = inline_element.text
                    inline_element.replace_with(h4_element)
                    pseudo_headline_counter += 1
                else:
                    inline_element.unwrap()
                    inline_counter += 1

    def parse_pre_tags(self):
        # Find all <pre> tags
        pre_tags = self.soup.find_all('pre')

        # List of white-space CSS styles to look for
        white_space_styles = ['pre', 'pre-line', 'pre-wrap', 'break-spaces']

        # Regex pattern to match white-space styles with or without spaces after the colon
        pattern = re.compile(r'white-space:\s*({})'.format('|'.join(white_space_styles)))

        # Find all tags with specified white-space styles
        tags_with_white_space = self.soup.find_all(lambda tag: tag.has_attr('style') and pattern.search(tag['style']))

        # Combine the results
        all_relevant_tags = pre_tags + tags_with_white_space

        if all_relevant_tags:
            # print('Parsing <pre> tags (found %d)...' % len(all_relevant_tags))
            pass
        for pre in all_relevant_tags:
            try:
                # if the element has non-NavigableString children, skip it
                if any(isinstance(child, Tag) for child in pre.children):
                    continue

                # count the number of line breaks in the content
                line_breaks = pre.get_text().count('\n')
                # print('Found %d line breaks in <pre> content...' % line_breaks)
                # Split content based on two or more consecutive line breaks, and insert each chunk as a <p> element
                chunks = re.split(r'\n', pre.get_text())
                # print('Splitting %d chunks...' % len(chunks))
                for chunk in chunks:
                    new_p = self.soup.new_tag('p')
                    new_p.string = chunk
                    pre.insert_before(new_p)
            except Exception as e:
                pass

            # Remove the original <pre> element
            pre.decompose()


    def replace_br_tags(self):
        # find all <br> tags and wrap all previous siblings in a p tag and all next siblings in a p tag
        for br in self.soup.find_all('br'):
            # wrap all previous siblings in a p tag
            previous_siblings = list(br.previous_siblings)
            if previous_siblings:
                previous_p = self.soup.new_tag('p')
                for sibling in previous_siblings:
                    previous_p.insert(0, sibling.extract())
                br.insert_before(previous_p)

            # wrap all next siblings in a p tag
            next_siblings = list(br.next_siblings)
            if next_siblings:
                next_p = self.soup.new_tag('p')
                for sibling in next_siblings:
                    next_p.append(sibling.extract())
                br.insert_after(next_p)

            # remove the <br> tag
            br.decompose()

    def remove_attributes(self):
        # remove all attributes from each tag
        for element in self.soup.find_all():
            if element.name in ['td', 'th']:
                # For <td> elements, preserve 'rowspan' and 'colspan' if they exist
                rowspan = element.get('rowspan')
                colspan = element.get('colspan')
                element.attrs = {}
                if rowspan:
                    element.attrs['rowspan'] = rowspan
                else:
                    element.attrs['rowspan'] = 1
                if colspan:
                    element.attrs['colspan'] = colspan
                else:
                    element.attrs['colspan'] = 1
            else:
                # For all other elements, remove all attributes
                element.attrs = {}

    def replace_special_characters(self):
        for element in self.soup.find_all():
            if isinstance(element, NavigableString):
                # replace <br> with \n
                # element.replace_with(element.replace('<br>', ' '))
                element.replace_with(element.replace(' ', ' '))
                element.replace_with(element.replace('​', ' '))

    def concatenate_text(self):
        # concatenate the text of each element
        for element in self.soup.find_all():
            # text can be split into multiple content entries; concatenate adjacent entries
            i = 0
            while i < len(element.contents):
                content = element.contents[i]

                # if the content is a NavigableString, start concatenation
                if isinstance(content, NavigableString):
                    concatenated_text = content

                    # continue concatenating adjacent NavigableStrings
                    while i + 1 < len(element.contents) and (isinstance(element.contents[i + 1], NavigableString) or
                                                             element.contents[i + 1].name == 'br'):
                        if element.contents[i + 1].name == 'br':
                            concatenated_text += '\n'
                        else:
                            concatenated_text += element.contents[i + 1]
                        del element.contents[i + 1]  # Remove the next item as it's concatenated

                    # normalize whitespace in the concatenated text
                    concatenated_text = ' '.join(concatenated_text.split())

                    # replace the original string with the concatenated one
                    element.contents[i].replace_with(concatenated_text)

                i += 1  # Move to the next content item

    def wrap_nonterminal_text(self):
        # print('Wrapping non-terminal text in <p> elements...')

        # if an element has direct text as well as children, wrap the text content in <p> elements
        for element in self.soup.find_all():
            if len(element.contents) > 1:
                for child in element.contents:
                    if isinstance(child, NavigableString) and child.strip():
                        # print("Wrapping non-terminal text in %s" % element.name)
                        new_p = self.soup.new_tag("p")
                        new_p.string = child
                        child.replace_with(new_p)

    def flatten_structure(self):
        # flatten the soup, unwrapping all elements that don't have text directly within them
        allowed_wrapper_tags = ['ul', 'ol', 'li', 'table', 'thead', 'tbody', 'tr', 'td', 'th']
        elements = [element for element in self.soup.find_all() if element.name is not None]
        for element in elements:
            # Check if tag has no text of its own
            if (not any(isinstance(child, NavigableString) and child.strip() for child in element.children)) and (
                    element.name not in allowed_wrapper_tags):
                element.unwrap()

        # flatten all inner wrapper tags that only contain a single child element, removing the inner tag
        flattenable_wrapper_tags = ['li', 'tr', 'td', 'th']
        for tag_name in flattenable_wrapper_tags:
            for tag in self.soup.find_all(tag_name):
                # Check if the tag has exactly one child element
                if len(tag.contents) == 1 and isinstance(tag.contents[0], Tag):
                    child = tag.contents[0]
                    # Replace the tag's content with the child's content
                    if len(child.contents) > 0:
                        tag.clear()
                        tag.append(child.contents[0])

    def remove_empty_elements(self):
        # remove all empty tags
        for element in self.soup.find_all():
            if not element.contents:
                element.decompose()
            # remove all tags that only contain whitespace
            elif re.match(r'^\s*$', element.get_text()):
                element.decompose()

    def close_headline_gaps(self):
        # increase the level of headline tags to remove all "level gaps" in the hierarchy
        # example: if a headline tag h3 follows an h1, it will be transformed into an h2
        last_tag_level = 1
        last_tag_level_original = 1
        for tag in self.headline_tags:
            tag_level = int(tag.name[1])
            if tag_level > last_tag_level + 1:
                last_tag_level_original = tag_level
                tag_level = last_tag_level + 1
                tag.name = 'h' + str(tag_level)
            elif tag_level == last_tag_level_original:
                tag_level = last_tag_level
                tag.name = 'h' + str(tag_level)
            last_tag_level = tag_level
