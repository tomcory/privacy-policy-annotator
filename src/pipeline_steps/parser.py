import json

from bs4 import BeautifulSoup, NavigableString, Tag, PageElement

from src import util
from src.llm_connectors.api_base import ApiBase
from src.pipeline_steps.pipeline_step import PipelineStep
from src.state_manager import BaseStateManager


class Parser(PipelineStep):

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
            task='parse',
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

        self.BLOCK_TYPES = {
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

        self.html_content = None
        self.blocks = []
        self.headline_levels = [-1] * 6
        self.context = []
        self.predecessor = None
        self.table_x = 0
        self.table_y = 0
        self.table_header = []
        self.table_row_header = None

    async def execute(self, pkg: str):
        try:
            self.html_content = util.read_from_file(f"{self.in_folder}/{pkg}.html")
            if len(self.html_content) > 0:
                parsed_structure = self._parse()
                output = json.dumps(parsed_structure, indent=2)
                util.write_to_file(f"{self.out_folder}/{pkg}.json", output)
            else:
                #TODO: handle empty HTML
                pass
        except Exception as e:
            await self.state_manager.raise_error(error_message=str(e))

    async def prepare_batch(self, pkg: str):
        pass

    def _parse(self):
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
        block_type = self.BLOCK_TYPES.get(parent) if parent in self.BLOCK_TYPES else 'UNKNOWN_' + parent
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
        # add the headline's text as a passage
        self.__handle_children(element)
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
