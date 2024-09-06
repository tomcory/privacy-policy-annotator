from src.api_wrapper import ApiWrapper
from src.base_processor import BaseProcessor

class Annotator(BaseProcessor):
    def __init__(self, run_id: str, pkg: str, llm_api: ApiWrapper, model: str, use_batch: bool = False):
        super().__init__(run_id, pkg, llm_api, model, "annotator", use_batch)
