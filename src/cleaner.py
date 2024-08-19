import re
import logging
import traceback

import chardet
from bs4 import NavigableString, Comment, BeautifulSoup, Tag

from src import util


def remove_surrounding_structure(soup: BeautifulSoup):
    print('Removing surrounding structure...')

    # remove all comments
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    i = 0
    for comment in comments:
        comment.extract()
        i += 1
    print('> Removed %d comments.' % i)

    # remove all unwanted tags
    unwanted_tags = ['head', 'title', 'meta', 'link', 'style', 'script', 'noscript', 'iframe', 'object', 'nav', 'footer']
    i = 0
    for tag in unwanted_tags:
        for element in soup.find_all(tag):
            element.decompose()
            i += 1
    print('> Removed %d elements based on their tag name.' % i)

    # remove all tags that have classes, roles or ids related to nav, header, footer, modals, banners, etc.
    removal_counter = 0
    search_strings = ['.*nav.*', '.*header.*', '.*footer.*', '.*[cC]ookie.*', '.*banner.*', '.*popup.*',
                      '.*toolbar.*', '.*modal.*', '.*dialog.*', '.*overlay.*', '.*consent.*', '.*dropdown.*',
                      '.*onetrust.*', '.*didomi.*', '.*country-selector.*']
    for s in search_strings:
        regex = re.compile(pattern=s, flags=re.I)
        match_classes = soup.find_all("div", {"class": regex})
        match_classes.reverse()
        for match_class in match_classes:
            # print('Removing element based on class %s' % match_class['class'])
            match_class.decompose()
            removal_counter += 1
            pass
        match_roles = soup.find_all("div", {"role": regex})
        match_roles.reverse()
        for match_role in match_roles:
            # print('Removing element based on role %s' % match_role['role'])
            match_role.decompose()
            removal_counter += 1
            pass
        match_ids = soup.find_all('div', id=regex)
        match_ids.reverse()
        for match_id in match_ids:
            # print('Removing element based on id %s' % match_id['id'])
            match_id.decompose()
            removal_counter += 1
            pass

    print('> Removed %d elements based on their class, role or id.' % removal_counter)

    # if there's a main element, reduce the soup to that
    policy = soup.find("main")
    if policy is None:
        # if there's an article element, reduce the soup to that
        policy = soup.find("article")
        if policy is None:
            # some sites use the class 'article' instead of the tag itself
            policy = soup.find("div", {"class": re.compile('.*[aA]rticle.*')})
            if policy is None:
                # some sites use the role 'article' instead of the tag itself
                policy = soup.find("div", {"role": re.compile('.*[aA]rticle.*')})
                if policy is None:
                    # some sites use the id 'article' instead of the tag itself
                    policy = soup.find("div", id=re.compile('.*[aA]rticle.*'))
                    if policy is None:
                        # some sites use the class 'main' instead of the tag itself
                        policy = soup.find("div", {"class": re.compile('.*[mM]ain.*')})
                        if policy is None:
                            # some sites use the role 'main' instead of the tag itself
                            policy = soup.find("div", {"role": re.compile('.*[mM]ain.*')})
                            if policy is None:
                                print('> No main container found.')
                            else:
                                print('Main (class) found, reducing soup...')
                        else:
                            print('Main (role) found, reducing soup...')
                    else:
                        print('Article (id) found, reducing soup...')
                else:
                    print('Article (class) found, reducing soup...')
            else:
                print('Article (role) found, reducing soup...')
        else:
            print('Article (tag) found, reducing soup...')
    else:
        print('Main (tag) found, reducing soup...')

    if policy is not None:
        # print the tag that was found with its attributes
        print('Found tag: %s' % policy)
        # remove all elements of the soup except the policy container and its children
        soup = BeautifulSoup(str(policy), 'html.parser')

    print('> Reduced soup to the main content.')

    return soup


def unwrap_inline_elements(soup: BeautifulSoup):
    # Extended list of inline element tags to unwrap
    inline_tags = ['a', 'abbr', 'acronym', 'b', 'bdi', 'bdo', 'big', 'button', 'center',
                   'cite', 'code', 'data', 'del', 'dfn', 'em', 'font', 'i', 'img', 'input',
                   'ins', 'kbd', 'label', 'map', 'mark', 'meter', 'noscript', 'object',
                   'output', 'picture', 'progress', 'q', 'ruby', 's', 'samp', 'script',
                   'select', 'small', 'span', 'strong', 'sub', 'sup', 'svg', 'template',
                   'textarea', 'time', 'tt', 'u', 'var', 'wbr']

    highlight_tags = ['b', 'strong', 'i', 'em', 'big', 'u']

    headline_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']

    print('Transforming pseudo-headlines and unwrapping inline elements...')

    # Find all headline tags (h1 through h6)
    headlines = soup.find_all(headline_tags)

    # Determine the highest-order headline tag
    least_significant_headline = min(5, max((int(tag.name[1]) for tag in headlines), default=1))

    pseudo_headline_counter = 0
    inline_counter = 0

    # unwrap all inline tags as defined by the list above, transforming pseudo-headlines into proper headline tags
    for tag in inline_tags:
        for inline_element in soup.find_all(tag):
            if False and len(inline_element.parent.contents) == 1 and inline_element.name in highlight_tags and \
                    inline_element.parent.name not in headlines:
                replacement_tag = 'h' + str(least_significant_headline + 1)
                h4_element = soup.new_tag(replacement_tag)
                h4_element.string = inline_element.text
                inline_element.replace_with(h4_element)
                pseudo_headline_counter += 1
            else:
                inline_element.unwrap()
                inline_counter += 1

    print('> Unwrapped %d inline elements and transformed %d pseudo-headlines.' % (inline_counter, pseudo_headline_counter))

    return soup


def parse_pre_tags(soup: BeautifulSoup):
    # Find all <pre> tags
    pre_tags = soup.find_all('pre')

    # List of white-space CSS styles to look for
    white_space_styles = ['pre', 'pre-line', 'pre-wrap', 'break-spaces']

    # Regex pattern to match white-space styles with or without spaces after the colon
    pattern = re.compile(r'white-space:\s*({})'.format('|'.join(white_space_styles)))

    # Find all tags with specified white-space styles
    tags_with_white_space = soup.find_all(lambda tag: tag.has_attr('style') and pattern.search(tag['style']))

    # Combine the results
    all_relevant_tags = pre_tags + tags_with_white_space

    if all_relevant_tags:
        print('Parsing <pre> tags (found %d)...' % len(all_relevant_tags))
    for pre in all_relevant_tags:
        try:
            # if the element has non-NavigableString children, skip it
            if any(isinstance(child, Tag) for child in pre.children):
                continue

            # count the number of line breaks in the content
            line_breaks = pre.get_text().count('\n')
            print('Found %d line breaks in <pre> content...' % line_breaks)
            # Split content based on two or more consecutive line breaks, and insert each chunk as a <p> element
            chunks = re.split(r'\n', pre.get_text())
            print('Splitting %d chunks...' % len(chunks))
            for chunk in chunks:
                new_p = soup.new_tag('p')
                new_p.string = chunk
                pre.insert_before(new_p)
        except Exception as e:
            pass

        # Remove the original <pre> element
        pre.decompose()

    return soup


def replace_br_tags(soup: BeautifulSoup):
    # find all <br> tags and wrap all previous siblings in a p tag and all next siblings in a p tag
    for br in soup.find_all('br'):
        # wrap all previous siblings in a p tag
        previous_siblings = list(br.previous_siblings)
        if previous_siblings:
            previous_p = soup.new_tag('p')
            for sibling in previous_siblings:
                previous_p.insert(0, sibling.extract())
            br.insert_before(previous_p)

        # wrap all next siblings in a p tag
        next_siblings = list(br.next_siblings)
        if next_siblings:
            next_p = soup.new_tag('p')
            for sibling in next_siblings:
                next_p.append(sibling.extract())
            br.insert_after(next_p)

        # remove the <br> tag
        br.decompose()
    return soup


def remove_attributes(soup: BeautifulSoup):
    # remove all attributes from each tag
    for element in soup.find_all():
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

    return soup


def replace_special_characters(soup: BeautifulSoup):
    for element in soup.find_all():
        if isinstance(element, NavigableString):
            # replace <br> with \n
            # element.replace_with(element.replace('<br>', ' '))
            element.replace_with(element.replace(' ', ' '))
            element.replace_with(element.replace('​', ' '))
    return soup


def concatenate_text(soup: BeautifulSoup):
    print('Concatenating text content of elements...')

    # concatenate the text of each element
    for element in soup.find_all():
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

    return soup


def wrap_nonterminal_text(soup: BeautifulSoup):
    print('Wrapping non-terminal text in <p> elements...')

    # if an element has direct text as well as children, wrap the text content in <p> elements
    for element in soup.find_all():
        if len(element.contents) > 1:
            for child in element.contents:
                if isinstance(child, NavigableString) and child.strip():
                    print("Wrapping non-terminal text in %s" % element.name)
                    if element.name == "h2":
                        for child2 in element.children:
                            print("h2 child: %s" % child)
                    new_p = soup.new_tag("p")
                    new_p.string = child
                    child.replace_with(new_p)

    return soup


def flatten_structure(soup: BeautifulSoup):
    print('Flattening soup...')

    # flatten the soup, unwrapping all elements that don't have text directly within them
    allowed_wrapper_tags = ['ul', 'ol', 'li', 'table', 'thead', 'tbody', 'tr', 'td', 'th']
    elements = [element for element in soup.find_all() if element.name is not None]
    for element in elements:
        # Check if tag has no text of its own
        if (not any(isinstance(child, NavigableString) and child.strip() for child in element.children)) and (
                element.name not in allowed_wrapper_tags):
            element.unwrap()

    # flatten all inner wrapper tags that only contain a single child element, removing the inner tag
    flattenable_wrapper_tags = ['li', 'tr', 'td', 'th']
    for tag_name in flattenable_wrapper_tags:
        for tag in soup.find_all(tag_name):
            # Check if the tag has exactly one child element
            if len(tag.contents) == 1 and isinstance(tag.contents[0], Tag):
                child = tag.contents[0]
                # Replace the tag's content with the child's content
                tag.clear()
                tag.append(child.contents[0])

    return soup


def remove_empty_elements(soup: BeautifulSoup):
    # remove all empty tags
    for element in soup.find_all():
        if not element.contents:
            element.decompose()
        # remove all tags that only contain whitespace
        elif re.match(r'^\s*$', element.get_text()):
            element.decompose()

    return soup


def close_headline_gaps(soup: BeautifulSoup, headline_tags: list[str]):
    # increase the level of headline tags to remove all "level gaps" in the hierarchy
    # example: if a headline tag h3 follows an h1, it will be transformed into an h2
    last_tag_level = 1
    last_tag_level_original = 1
    for tag in headline_tags:
        tag_level = int(tag.name[1])
        if tag_level > last_tag_level + 1:
            last_tag_level_original = tag_level
            tag_level = last_tag_level + 1
            tag.name = 'h' + str(tag_level)
        elif tag_level == last_tag_level_original:
            tag_level = last_tag_level
            tag.name = 'h' + str(tag_level)
        last_tag_level = tag_level

    return soup


def simplify_soup(page_source):
    """
        Extract the policy text from the page source using BeautifulSoup.

        Args:
            page_source (str): The HTML source of the page.

        Returns:
            BeautifulSoup: The isolated and cleaned main content of the page source or None if an error occurred.
        """

    try:
        soup = BeautifulSoup(page_source, 'html.parser')
        soup = remove_surrounding_structure(soup)
        soup = unwrap_inline_elements(soup)
        soup = replace_br_tags(soup)
        soup = parse_pre_tags(soup)
        soup = replace_special_characters(soup)
        soup = remove_attributes(soup)
        soup = concatenate_text(soup)
        soup = wrap_nonterminal_text(soup)
        soup = flatten_structure(soup)
        soup = remove_empty_elements(soup)

        print('Soup simplified.')

        return soup
    except Exception as e:
        print(
            f'An error occurred while simplifying the soup: {type(e)}.')
        traceback.print_exc()
        return None


def clean_policy(page) -> str:
    try:
        print('Attempting to clean the soup...')
        policy = simplify_soup(page)
        if policy is None:
            print('Cleaning the soup failed.')
            raise Exception

        print('Cleaning the soup succeeded.')
        return policy.prettify()

    except Exception as e:
        print('Error: Unknown exception %s' % e)
        return ''


class Cleaner:
    def __init__(self, run_id: str, pkg: str, use_batch: bool = False):
        self.in_folder = f"output/{run_id}/original"
        self.out_folder = f"output/{run_id}/cleaned"

        self.run_id = run_id
        self.pkg = pkg
        self.use_batch = use_batch

    def execute(self):
        print(">>> Cleaning %s..." % self.pkg)
        logging.info("Cleaning %s..." % self.pkg)
        file_path = "%s/%s.html" % (self.in_folder, self.pkg)

        # Detect the encoding of the file
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        policy = util.read_from_file(file_path, encoding)
        cleaned = simplify_soup(policy).prettify()

        if len(cleaned) > 0:
            util.write_to_file("%s/%s.html" % (self.out_folder, self.pkg), cleaned)

    def skip(self):
        print(">>> Skipping cleaning %s..." % self.pkg)
        logging.info("Skipping cleaning %s..." % self.pkg)
        util.copy_folder(self.in_folder, self.out_folder)
