import json

import tiktoken
from bs4 import BeautifulSoup, NavigableString, Tag, PageElement

from src import util

BLOCK_TYPES = {
    'td': 'table_cell',
    'th': 'table_header',
    'li': 'list_item',
    'p': 'text',
    'div': 'text',
    'h1': 'headline',
    'h2': 'headline',
    'h3': 'headline',
    'h4': 'headline',
    'h5': 'headline',
    'h6': 'headline'
}


def execute(run_id: str, pkg: str, in_folder: str, out_folder: str, use_batch: bool = False):
    print(">>> Parsing %s..." % pkg)
    file_path = "%s/%s.html" % (in_folder, pkg)

    html = util.read_from_file(file_path)
    output = parse_to_json(html)

    file_path = "%s/%s.json" % (out_folder, pkg)
    util.write_to_file(file_path, output)


def parse_to_json(html: str):
    if len(html) > 0:
        parser = Parser(html)
        return json.dumps(parser.parse(), indent=2)
    else:
        print('Cannot parse an empty HTML document!')


class Parser:
    def __init__(self, html_content, tokenizer_encoding='cl100k_base'):
        self.html_content = html_content
        self.blocks = []
        self.headline_levels = [-1] * 6
        self.context = []
        self.predecessor = None
        self.table_x = 0
        self.table_y = 0
        self.table_header = []
        self.table_row_header = None
        self.encoder = tiktoken.get_encoding(tokenizer_encoding)

    def parse(self):
        if not self.blocks:
            soup = BeautifulSoup(self.html_content, 'html.parser')
            self.__handle_children(soup)
        return self.blocks

    def __handle_children(self, element):
        for child in element.contents:
            self.__handle_element(child)
            if isinstance(child, Tag):
                self.predecessor = child

    def __handle_element(self, element: PageElement):
        if isinstance(element, Tag):
            self.__handle_tag(element)
        elif isinstance(element, NavigableString) and len(element.strip()) > 0:
            self.__handle_string(element)

    def __handle_string(self, element: NavigableString):
        parent = element.parent.name
        block_type = BLOCK_TYPES.get(parent) if parent in BLOCK_TYPES else 'UNKNOWN_' + parent
        block = {
            'type': block_type,
            'context': self.context.copy(),
            'passage': element.text.strip()
        }
        self.blocks.append(block)

    def __handle_tag(self, element: Tag):
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            self.__handle_headline(element)
        elif element.name in ['ol', 'ul', 'table']:
            self.__handle_block(element)
        elif element.name == 'tr':
            self.__handle_table_row(element)
        elif element.name in ['td', 'th']:
            self.__handle_table_cell(element)
        else:
            self.__handle_children(element)

    def __handle_headline(self, element: Tag):
        # if there's already a context entry for this headline level, trim the context accordingly
        previous_index, _ = self.__find_context_entry(element.name)
        if previous_index >= 0:
            self.context = self.context[:previous_index]

        self.context.append({'text': element.text.strip(), 'type': element.name})

    def __handle_block(self, element):
        context_length = len(self.context)

        # if there's a preceding paragraph, add it to the context
        if self.predecessor is not None and self.predecessor.name in ['p', 'div']:
            context_type = 'table_intro' if element.name == 'table' else 'list_intro'
            self.context.append({'text': self.predecessor.text.strip(), 'type': context_type})

        self.__handle_children(element)
        self.context = self.context[:context_length]

        if element.name == 'table':
            # reset table traversal variables
            self.table_x = 0
            self.table_y = 0
            self.table_header = []
            self.table_row_header = None

    def __handle_table_row(self, element):
        # add this row as the new table header if it's the first row of the table
        if self.table_y == 0:
            # append each child of the row <colspan> times
            for element in [e for e in element.contents if isinstance(e, Tag)]:
                colspan = int(element.attrs['colspan'] if element.attrs['colspan'] else '1')
                for _ in range(colspan):
                    self.table_header.append(element.text.strip())

        self.__handle_children(element)
        self.table_y += 1
        self.table_x = 0

    def __handle_table_cell(self, element):
        context_length = len(self.context)

        # add this cell as the new row header if it's the first cell of the row
        if self.table_x == 0:
            self.table_row_header = element.text.strip()

        # add the current column header to the context if it's not too long
        if self.table_y > 0:
            if self.table_x < len(self.table_header):
                col_header = self.table_header[self.table_x]
                if len(self.encoder.encode(col_header)) < 30:
                    self.context.append({'text': col_header, 'type': 'table_column_header'})

        # add the current row header to the context if it's not too long
        if self.table_x > 0:
            if len(self.encoder.encode(self.table_row_header)) < 30:
                self.context.append({'text': self.table_row_header, 'type': 'table_row_header'})

        self.__handle_children(element)
        self.context = self.context[:context_length]
        colspan = int(element.attrs['colspan'] if element.attrs['colspan'] else '1')
        self.table_x += colspan

    def __find_context_entry(self, target_type: str):
        for index, element in enumerate(self.context):
            if element['type'] == target_type:
                return index, element
        return -1, None
