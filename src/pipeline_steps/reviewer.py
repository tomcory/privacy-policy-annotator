import json
import os
import sys

from src import util
from src.llm_connectors.api_base import ApiBase
from src.pipeline_steps.pipeline_step import PipelineStep
from src.state_manager import BaseStateManager
from src.rag.rag_injector import RAGInjector
from src.rag.rag_injector_opp import OPPRAGInjector


class Reviewer(PipelineStep):
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
            min_example_confidence: float = 0.8,
            use_opp_115: bool = False
    ):
        super().__init__(
            run_id=run_id,
            task='review',
            details='Review and correct annotations for GDPR compliance',
            skip=skip,
            is_llm_step=True,
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
        
        self.use_rag = use_rag
        self.max_examples_per_requirement = max_examples_per_requirement
        self.min_example_confidence = min_example_confidence
        self.use_opp_115 = use_opp_115
        
        # Initialize RAG injector if enabled
        self.rag_injector = None
        if self.use_rag:
            if self.use_opp_115:
                self.rag_injector = OPPRAGInjector(database_type='annotate')
            else:
                self.rag_injector = RAGInjector(database_type='annotate')

    def _load_rag_system_prompt(self) -> str:
        """Load the RAG system prompt template."""
        _, system_msg, _ = util.prepare_prompt_messages(
            self.client.api, self.task, None, None, None, use_opp_115=self.use_opp_115
        )
        return system_msg

    def _extract_requirements_from_annotations(self, passage: dict) -> list[str]:
        """
        Extract requirements from the passage's existing annotations for RAG examples.
        
        Args:
            passage: Dictionary containing passage data with annotations field
            
        Returns:
            List of requirement labels found in the annotations
        """
        annotations = passage.get('annotations', [])
        requirements = set()
        
        for annotation in annotations:
            if isinstance(annotation, dict) and 'requirement' in annotation:
                requirements.add(annotation['requirement'])
            else:
                print("Annotation is not a dictionary or does not have a requirement field!")
                sys.exit(1)
        
        return list(requirements)

    def _build_rag_prompt(self, passage: dict, target_requirements: list) -> str:
        """Build a RAG-enhanced prompt for specific requirements using the system prompt template."""
        if not self.use_rag or not self.rag_injector:
            # Load the basic system prompt without RAG
            return self.rag_system_prompt

        # Inject RAG materials (labels and legal background) into the system prompt
        return self.rag_injector.inject_rag_materials_into_prompt(
            base_prompt=self.rag_system_prompt,
            target_requirements=target_requirements
        )

    def _get_rag_examples(self, passage: dict, target_requirements: list) -> list[tuple[str, str]]:
        """Get RAG examples in the format expected by the API connector."""
        if not self.rag_injector:
            print("No RAG injector found!")
            sys.exit(1)
        
        # Get existing similar examples - use a higher number for similarity-based examples
        # This should be independent of the per-requirement limit
        # Use a reasonable confidence threshold
        debug_confidence = min(self.min_example_confidence, 0.5)  # Use lower of 0.8 or 0.5
        similar_examples = self.rag_injector.get_similar_examples_for_api(
            passage=passage,
            max_examples=5,  # Get more similarity-based examples
            min_confidence=debug_confidence
        )
        
        # Get label-specific examples (2 per requirement)
        label_specific_examples = self.rag_injector.get_label_specific_examples_for_api(
            passage=passage,
            target_requirements=target_requirements,
            examples_per_requirement=2,
            min_confidence=debug_confidence
        )
        
        # Combine both sets of examples
        all_examples = similar_examples + label_specific_examples
        
        return all_examples

    async def execute(self, pkg: str):
        """Execute review for a single package."""
        try:
            # Try to load from JSONL first, fallback to JSON for backward compatibility
            policy = util.load_policy_jsonl(f'{self.in_folder}/{pkg}.jsonl')
            if policy is None:
                policy = util.load_policy_json(f'{self.in_folder}/{pkg}.json')
            if policy is None:
                return None
            
            # Load the RAG system prompt template
            self.rag_system_prompt = self._load_rag_system_prompt()

            total_cost = 0
            total_time = 0

            if self.parallel_prompt:
                # For parallel processing, we need to prepare RAG examples for each passage
                user_msgs = []
                system_msgs = []
                examples_list = []
                
                for passage in policy:
                    # Extract requirements from existing annotations
                    target_requirements = self._extract_requirements_from_annotations(passage)

                    # remove "Other" from the result
                    annotations = [annotation for annotation in annotations if annotation['requirement'] != "Other"]
                    
                    # Build RAG-enhanced system prompt
                    system_msg = self._build_rag_prompt(passage, target_requirements)
                    system_msgs.append(system_msg)
                    
                    # Get RAG examples
                    examples = self._get_rag_examples(passage, target_requirements)
                    examples_list.append(examples)
                    
                    user_msgs.append(json.dumps(passage))

                results, total_cost, total_time = self.client.prompt_parallel(
                    pkg=pkg,
                    task=self.task,
                    user_msgs=user_msgs,
                    system_msgs=system_msgs,
                    examples_list=examples_list,
                    model=self.model,
                    response_format='json_schema',
                    max_tokens=8192
                )

                # iterate over each result and add the annotations to the policy
                for index, result in enumerate(results):
                    try:
                        annotations = util.parse_llm_json_response(result, expected_key='annotations')
                        # Correct any hallucinated requirement labels in annotations
                        corrected_annotations = util.correct_annotations(annotations, use_opp_115=self.use_opp_115)
                        policy[index]['annotations'] = corrected_annotations
                    except Exception as e:
                        policy[index]['annotations'] = []
                        print(f"Error processing annotation result: {str(e)}")
                        # Don't raise error for individual failures, just log and continue
                
                # Write all results as JSONL
                util.write_jsonl_file(f"{self.out_folder}/{pkg}.jsonl", policy)
            else:
                # iterate over each passage in the policy and review it or retrieve the review from the batch result
                for index, passage in enumerate(policy):
                    await self.state_manager.update_state(file_progress=index / len(policy))

                    if self.is_batch_step:
                        result, cost, time = self.client.retrieve_batch_result_entry(
                            task=self.task,
                            entry_id=f"{self.run_id}_{self.task}_{pkg}_{index}",
                            batch_results_file=self.batch_results_file
                        )
                    else:
                        system_msg = None
                        examples = None
                        
                        if self.use_rag:
                            # Extract requirements from existing annotations
                            target_requirements = self._extract_requirements_from_annotations(passage)

                            # Get RAG examples for the API connector
                            examples = self._get_rag_examples(passage, target_requirements)
                            system_msg = self._build_rag_prompt(passage, target_requirements)
                        
                        result, cost, time = self.client.prompt(
                            pkg=pkg,
                            task=self.task,
                            model=self.model,
                            system_msg=system_msg,
                            user_msg=json.dumps(passage),
                            examples=examples,
                            response_format='json_schema',
                            max_tokens=8192
                        )

                    total_cost += cost
                    total_time += time

                    # add the annotations to the passage
                    try:
                        annotations = util.parse_llm_json_response(result, expected_key='annotations')

                        # remove "Other" from the result
                        annotations = [annotation for annotation in annotations if annotation['requirement'] != "Other"]

                        # Correct any hallucinated requirement labels in annotations
                        corrected_annotations = util.correct_annotations(annotations, use_opp_115=self.use_opp_115)
                        passage['annotations'] = corrected_annotations

                        print("--------------------------------")
                        print(json.dumps(passage))
                        print("--------------------------------")
                    except Exception as e:
                        passage['annotations'] = []
                        print(f"Error processing annotation result: {str(e)}")
                        # Don't raise error for individual failures, just log and continue
                    
                    # Append each passage individually to JSONL file
                    util.append_to_file(f"{self.out_folder}/{pkg}.jsonl", json.dumps(passage))

            await self.state_manager.update_state(
                file_progress=1,
                message=f"Cost: {total_cost:.2f} USD, Time: {total_time:.2f} seconds"
            )

        except Exception as e:
            await self.state_manager.raise_error(error_message=str(e))
            util.write_to_file(f"../../output/{self.run_id}/log/failed_review.txt", pkg)
            return None


    async def prepare_batch(self, pkg: str):
        """Prepare batch entry for review."""
        try:
            # Try to load from JSONL first, fallback to JSON for backward compatibility
            policy = util.load_policy_jsonl(f'{self.in_folder}/{pkg}.jsonl')
            if policy is None:
                policy = util.load_policy_json(f'{self.in_folder}/{pkg}.json')
            if policy is None:
                return None
            
            # Load the RAG system prompt template
            self.rag_system_prompt = self._load_rag_system_prompt()

            entries = []
            policy_length = len(policy)

            for index, passage in enumerate(policy):
                await self.state_manager.update_state(file_progress=index / policy_length)
                
                # Extract requirements from existing annotations
                target_requirements = self._extract_requirements_from_annotations(passage)
                
                # Get RAG examples for the API connector
                examples = self._get_rag_examples(passage, target_requirements)
                
                entry = self.client.prepare_batch_entry(
                    pkg=pkg,
                    task=self.task,
                    model=self.model,
                    system_msg=self._build_rag_prompt(passage, target_requirements),
                    user_msg=passage,
                    examples=examples,
                    response_format='json_schema',
                    max_tokens=8192,
                    entry_id=index
                )
                entries.append(entry)

            await self.state_manager.update_state(file_progress=1)
            return entries

        except Exception as e:
            await self.state_manager.raise_error(error_message=str(e))
            return None 