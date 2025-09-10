import json
from typing import Dict, List

from src import util
from src.llm_connectors.api_base import ApiBase
from src.pipeline_steps.pipeline_step import PipelineStep
from src.state_manager import BaseStateManager
from src.rag.rag_injector import RAGInjector
from src.rag.rag_injector_opp import OPPRAGInjector


class LLMClassifier(PipelineStep):
    def __init__(
            self,
            run_id: str,
            skip: bool,
            is_batch_step: bool,
            in_folder: str,
            out_folder: str,
            state_manager: BaseStateManager,
            batch_input_file: str = "batch_input.json",
            batch_metadata_file: str = "batch_metadata.json",
            batch_results_file: str = "batch_results.jsonl",
            batch_errors_file: str = "batch_errors.jsonl",
            parallel_prompt: bool = False,
            model: str = None,
            client: ApiBase = None,
            use_rag: bool = True,
            max_examples_per_requirement: int = 2,
            min_example_confidence: float = 0.5,
            use_opp_115: bool = False
    ):
        super().__init__(
            run_id=run_id,
            task='classify',
            details='Classify passages for GDPR requirement presence using LLM',
            skip=skip,
            is_llm_step=True,  # This is an LLM step
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
        
        # Define the 21 GDPR requirements
        self.requirement_types = [
            'Controller Name', 'Controller Contact', 'DPO Contact',
            'Data Categories', 'Processing Purpose', 'Legal Basis for Processing',
            'Legitimate Interests for Processing', 'Source of Data',
            'Data Retention Period', 'Data Recipients', 'Third-country Transfers',
            'Mandatory Data Disclosure', 'Automated Decision-Making',
            'Right to Access', 'Right to Rectification', 'Right to Erasure',
            'Right to Restrict', 'Right to Object', 'Right to Portability',
            'Right to Withdraw Consent', 'Right to Lodge Complaint'
        ]
        
        # RAG configuration
        self.use_rag = True
        self.min_examples = 3
        self.max_examples = 5
        self.min_example_confidence = 0.5
        self.use_opp_115 = use_opp_115
        
        # Initialize RAG injector
        self.rag_injector = None
        if self.use_rag:
            if self.use_opp_115:
                self.rag_injector = OPPRAGInjector(database_type='classify')
            else:
                self.rag_injector = RAGInjector(database_type='classify')

    def _build_classification_prompt(self, passage: dict) -> str:
        """Build a classification prompt with the passage and context."""
        
        # Start with the passage
        prompt = f"**Passage:**\n{passage.get('passage', '')}"
        
        # Add context information if available
        context_items = passage.get('context', [])
        if context_items:
            context_text = "\n\nContext:\n" + "\n".join([
                f"- {item.get('text', '')} ({item.get('type', 'unknown')})" 
                for item in context_items
            ])
            prompt = context_text + "\n\n" + prompt
        
        return prompt

    def _get_rag_examples(self, passage: dict) -> list[tuple[str, str]]:
        """Get RAG examples in the format expected by the API connector based on passage similarity."""
        if not self.use_rag or not self.rag_injector:
            examples = []
        else:
            examples = self.rag_injector.get_similar_examples_for_api(
                passage=passage,
                max_examples=self.max_examples,
                min_confidence=self.min_example_confidence
            )
        return examples

    async def execute(self, pkg: str):
        """Execute classification for a single package using LLM."""
        try:
            # Try to load from JSONL first, fallback to JSON for backward compatibility
            policy = util.load_policy_jsonl(f'{self.in_folder}/{pkg}.jsonl')
            if policy is None:
                policy = util.load_policy_json(f'{self.in_folder}/{pkg}.json')
            if policy is None:
                return None

            total_cost = 0
            total_time = 0

            if self.parallel_prompt:
                # parallel processing: process all passages in the policy at once
                results, total_cost, total_time = self.client.prompt_parallel(
                    pkg=pkg,
                    task=self.task,
                    user_msgs=[json.dumps(passage) for passage in policy],
                    model=self.model,
                    response_format='json',
                    max_tokens=1024
                )

                # Handle None values from client
                if total_cost is None:
                    total_cost = 0.0
                if total_time is None:
                    total_time = 0.0

                # iterate over each result and add the classifications to the policy
                for index, result in enumerate(results):
                    try:
                        predicted_labels = self._process_llm_classification(result)
                        policy[index]['predicted_labels'] = predicted_labels
                    except Exception as e:
                        # Fallback to empty classifications if LLM fails
                        policy[index]['predicted_labels'] = []
                        await self.state_manager.raise_error(error_message=str(e))
                    util.append_to_file(f"{self.out_folder}/{pkg}.jsonl", json.dumps(policy[index]))
            else:
                # sequential processing: iterate over each passage in the policy and classify it
                for index, passage in enumerate(policy):

                    await self.state_manager.update_state(file_progress=index / len(policy))

                    if self.is_batch_step:
                        # process the batch result entry
                        result, cost, time = self.client.retrieve_batch_result_entry(
                            task=self.task,
                            entry_id=f"{self.run_id}_{self.task}_{pkg}_{index}",
                            batch_results_file=self.batch_results_file
                        )
                    else:
                        # Build the user message with passage and context
                        user_msg = json.dumps(passage)
                        # Get RAG examples for the API connector
                        examples = self._get_rag_examples(passage)

                        # process the passage with the LLM
                        result, cost, time = self.client.prompt(
                            pkg=pkg,
                            task=self.task,
                            model=self.model,
                            response_format='json_schema',
                            user_msg=user_msg,
                            examples=examples,
                            max_tokens=1024
                        )

                    # Handle None values from client
                    if cost is None:
                        cost = 0.0
                    if time is None:
                        time = 0.0
                    
                    total_cost += cost
                    total_time += time

                    # add the classifications to the passage
                    try:
                        labels = util.parse_llm_json_response(result, expected_key='labels')

                        # remove "Other" from the result
                        result = [label for label in labels if label != "Other"]

                        # Correct any hallucinated labels
                        predicted_labels = self._process_llm_classification(labels)
                        passage['predicted_labels'] = predicted_labels

                        print("--------------------------------")
                        print(json.dumps(passage))
                        print("--------------------------------")
                    except Exception as e:
                        # Fallback to empty classifications if LLM fails
                        passage['predicted_labels'] = []
                        await self.state_manager.raise_error(error_message=str(e))
                    
                    # Append each passage individually to JSONL file
                    util.append_to_file(f"{self.out_folder}/{pkg}.jsonl", json.dumps(passage))

            await self.state_manager.update_state(
                file_progress=1,
                message=f"Cost: {total_cost:.2f} USD, Time: {total_time:.2f} seconds"
            )

        except Exception as e:
            await self.state_manager.raise_error(error_message=str(e))
            util.write_to_file(f"../../output/{self.run_id}/log/failed_classify.txt", pkg)
            return None

    async def prepare_batch(self, pkg: str):
        """Prepare batch entry for classification."""
        # Try to load from JSONL first, fallback to JSON for backward compatibility
        policy = util.load_policy_jsonl(f'{self.in_folder}/{pkg}.jsonl')
        if policy is None:
            policy = util.load_policy_json(f'{self.in_folder}/{pkg}.json')
        if policy is None:
            return None

        entries = []
        policy_length = len(policy)

        for index, passage in enumerate(policy):
            await self.state_manager.update_state(file_progress=index / policy_length)
            
            entry = self.client.prepare_batch_entry(
                pkg=pkg,
                task=self.task,
                model=self.model,
                user_msg=json.dumps(passage),
                examples=self._get_rag_examples(passage),
                response_format='json_schema',
                max_tokens=1024,
                entry_id=index
            )
            entries.append(entry)

        await self.state_manager.update_state(file_progress=1)

        return entries

    def _process_llm_classification(self, llm_result: str) -> List[str]:
        """
        Process the LLM classification result and return a list of predicted labels.
        The LLM returns a JSON array of requirement names, we correct any hallucinations.
        """
        try:
            # Handle None or empty result
            if llm_result is None:
                return []
            
            # Parse the LLM result using the centralized utility function
            if isinstance(llm_result, str):
                predicted_requirements = util.parse_llm_json_response(llm_result, expected_key='labels')
            else:
                predicted_requirements = llm_result

            # Ensure it's a list
            if not isinstance(predicted_requirements, list):
                try:
                    predicted_requirements = util.parse_llm_json_response(llm_result, expected_key='labels')
                    if isinstance(predicted_requirements, dict):
                        predicted_requirements = predicted_requirements.get('labels', [])
                    else:
                        predicted_requirements = []
                except Exception as e:
                    print(f"Error parsing llm_result: {str(e)}")
                    return []

            # Filter out None values and ensure all elements are strings
            if isinstance(predicted_requirements, list):
                predicted_requirements = [
                    str(req) for req in predicted_requirements 
                    if req is not None and str(req).strip()
                ]

            # Correct any hallucinated labels
            corrected_requirements = util.correct_labels(predicted_requirements, use_opp_115=self.use_opp_115)
            
            return corrected_requirements

        except Exception as e:
            print(f"Error in _process_llm_classification: {str(e)}")
            # If parsing fails, return empty list
            return [] 