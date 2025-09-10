from typing import Dict, List, Optional, Any
from src.rag.dummy_database import DummyDatabaseConnector, AnnotationExample
from src.rag.vector_database_opp import OPPVectorDatabaseConnector
import json


class OPPRAGInjector:
    """
    OPP-specific RAG injector that retrieves category-specific labels, legal background, and examples, and injects them into prompts.
    """
    
    def __init__(self, database_connector = None, use_vector_db: bool = True, database_type: str = 'annotate'):
        """
        Initialize the OPP RAG injector.
        
        Args:
            database_connector: Database connector for retrieving examples and background
            use_vector_db: Whether to use the vector database (default) or dummy database
            database_type: Type of database to use ('annotate' or 'classify')
        """
        if database_connector is not None:
            print(f"✅ Using provided OPP database connector for RAG examples (type: {database_type})")
            self.database = database_connector
        elif use_vector_db:
            try:
                self.database = OPPVectorDatabaseConnector()
                print(f"✅ Using OPP vector database for RAG examples (type: {database_type})")
            except Exception as e:
                print(f"⚠️  Failed to load OPP vector database, falling back to dummy database: {e}")
                self.database = DummyDatabaseConnector()
        else:
            print("⚠️  Using dummy database for OPP RAG examples")
            self.database = DummyDatabaseConnector()
        
        self.database_type = database_type

    def _replace_placeholders(self, text: str) -> str:
        """
        Replace all occurrences of "[...]" with "PLACEHOLDER" in the given text.
        
        Args:
            text: The text to process
            
        Returns:
            Text with "[...]" replaced by "PLACEHOLDER"
        """
        if isinstance(text, str):
            return text.replace("[...]", "PLACEHOLDER")
        return text

    def _replace_placeholders_in_annotations(self, annotations: list) -> list:
        """
        Replace "[...]" with "PLACEHOLDER" specifically in annotation "value" fields.
        
        Args:
            annotations: List of annotation objects (modified in-place)
            
        Returns:
            The same list with "[...]" replaced by "PLACEHOLDER" in annotation "value" fields
        """
        if not isinstance(annotations, list):
            return annotations
        
        for annotation in annotations:
            if isinstance(annotation, dict) and 'value' in annotation:
                if isinstance(annotation['value'], str):
                    annotation['value'] = self._replace_placeholders(annotation['value'])
        
        return annotations

    def inject_rag_materials_into_prompt(
        self,
        base_prompt: str,
        target_requirements: List[str]
    ) -> str:
        """
        Inject RAG materials (categories, background, examples) into a prompt.
        
        Args:
            base_prompt: The base prompt template
            target_requirements: List of OPP requirements to find examples/background for
        
        Returns:
            Enhanced prompt with RAG materials injected
        """
        # Format categories for the prompt
        categories_text = self._format_categories_for_prompt(target_requirements)
        
        # Retrieve and format legal background
        background_text = self._format_background_for_prompt(target_requirements)

        # Replace placeholders in the prompt
        enhanced_prompt = base_prompt
        enhanced_prompt = enhanced_prompt.replace("{{RAG_LABELS}}", categories_text)
        enhanced_prompt = enhanced_prompt.replace("{{RAG_BACKGROUND}}", background_text)
        
        return enhanced_prompt

    def _format_categories_for_prompt(self, categories: List[str]) -> str:
        """
        Format the list of OPP categories for the RAG_LABELS placeholder.
        Creates a targeted subset of the full OPP-115 categories list.
        """
        if not categories:
            return "(No relevant categories predicted for this passage.)"
        
        # Create a formatted list with category descriptions
        formatted_categories = []
        for category in categories:
            # Get category info from the database
            category_info = self.database.retrieve_opp_category_info(category)
            if category_info:
                description = category_info.get('description', '')
                formatted_categories.append(f'"{category}": {description}')
            else:
                formatted_categories.append(f'"{category}": OPP-115 Privacy Practice Category')
        
        return "\n".join(formatted_categories)

    def _format_background_for_prompt(self, categories: List[str]) -> str:
        """
        Retrieve and format legal background for each category for the RAG_BACKGROUND placeholder.
        """
        background_blocks = []

        if not categories:
            return "(No categories provided, no legal background available.)"
        
        for category in categories:
            background = self.database.retrieve_legal_background(category)
            if background:
                background_blocks.append(f"### {category}:\n{background}")
        if not background_blocks:
            return "(No legal background available for these categories.)"
        return "\n\n".join(background_blocks)

    def get_similar_examples_for_api(
        self,
        passage: dict,
        min_examples: int = 3,
        max_examples: int = 5,
        min_confidence: float = 0.5
    ) -> List[tuple[str, str]]:
        """
        Get similar examples based on passage and context similarity for the API connector.
        
        Args:
            passage: Dictionary containing 'passage' and optional 'context' keys
            max_examples: Maximum number of examples to return
            min_confidence: Minimum confidence threshold for examples
            
        Returns:
            List of tuples (user_message, assistant_message) for the API connector
        """
        if not self.database:
            print("No OPP database found")
            return []
        
        # Get the passage text and context for similarity search
        passage_text = passage.get('passage', '')

        # add context to the passage text
        context_items = passage.get('context', [])
        if context_items:
            context_text = "; ".join([item.get('text', '') for item in context_items])
            passage_text = f"{context_text} {passage_text}"
        
        # Search for similar examples in the database based purely on similarity
        # Use a higher top_k to account for confidence filtering
        similar_examples = self.database.search_similar_examples(
            query_text=passage_text,
            example_type=self.database_type,
            top_k=max_examples * 3  # Get more examples to filter by confidence
        )

        # sort the examples by similarity score
        similar_examples.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Filter by confidence and convert to API format
        api_examples = []
        for example in similar_examples:
            # Filter by confidence threshold if we have enough examples
            if example['similarity_score'] < min_confidence:
                if len(api_examples) < min_examples:
                    continue
                else:
                    continue
                
            # Create user message (passage) - OPP format only has passage
            user_msg = {
                "passage": example['metadata']['passage']
            }
            user_msg = json.dumps(user_msg)
            
            # Create assistant message based on database type
            if self.database_type == 'classify':
                output = example['metadata'].get('labels', [])
            else:
                output = example['metadata'].get('annotations', [])
                # Apply placeholder replacement specifically to annotation "value" fields
                if isinstance(output, list):
                    self._replace_placeholders_in_annotations(output)
            assistant_msg = json.dumps(output)
            
            api_examples.append((user_msg, assistant_msg))
            
            # Stop if we have enough examples
            if len(api_examples) >= max_examples:
                break
        
        return api_examples

    def get_category_specific_examples_for_api(
        self,
        passage: dict,
        target_categories: List[str],
        examples_per_category: int = 2,
        min_confidence: float = 0.5
    ) -> List[tuple[str, str]]:
        """
        Get examples specific to each predicted OPP category for the API connector.
        
        Args:
            passage: Dictionary containing 'passage' and optional 'context' keys
            target_categories: List of categories to find examples for
            examples_per_category: Number of examples to retrieve per category
            min_confidence: Minimum confidence threshold for examples
            
        Returns:
            List of tuples (user_message, assistant_message) for the API connector
        """
        if not self.database:
            return []
        
        # Retrieve examples for each category
        examples_by_category = self.database.retrieve_examples_for_requirements(
            requirements=target_categories,
            passage_text=passage.get('passage', ''),
            max_examples_per_requirement=examples_per_category,
            min_confidence=min_confidence,
            example_type=self.database_type
        )
        
        # Convert to API format
        api_examples = []
        for category, examples in examples_by_category.items():
            for example in examples:
                # Create user message (passage) - OPP format only has passage
                user_msg = {
                    "passage": example.passage
                }
                # Replace placeholders in passage text
                if isinstance(user_msg['passage'], str):
                    user_msg['passage'] = self._replace_placeholders(user_msg['passage'])
                user_msg = json.dumps(user_msg)
                
                # Create assistant message based on database type
                if self.database_type == 'classify':
                    # For classify examples, return the labels
                    labels = example.labels if example.labels else [category]
                    assistant_msg = json.dumps(labels)
                else:  # annotate
                    # For annotate examples, return the annotations
                    annotations = []
                    if example.annotations:
                        for annotation in example.annotations:
                            annotations.append({
                                "requirement": annotation.get('requirement', category),
                                "value": annotation.get('value', example.annotation_value),
                                "performed": annotation.get('performed', example.performed)
                            })
                    else:
                        # Fallback if no annotations available
                        annotations.append({
                            "requirement": category,
                            "value": example.annotation_value,
                            "performed": example.performed
                        })
                    # Apply placeholder replacement specifically to annotation "value" fields
                    self._replace_placeholders_in_annotations(annotations)
                    assistant_msg = json.dumps(annotations)
                
                api_examples.append((user_msg, assistant_msg))
        
        return api_examples

    def get_label_specific_examples_for_api(
        self,
        passage: dict,
        target_requirements: List[str],
        examples_per_requirement: int = 2,
        min_confidence: float = 0.5
    ) -> List[tuple[str, str]]:
        """
        Alias for get_category_specific_examples_for_api to maintain compatibility with pipeline steps.
        
        Args:
            passage: Dictionary containing 'passage' and optional 'context' keys
            target_requirements: List of requirements to find examples for (alias for target_categories)
            examples_per_requirement: Number of examples to retrieve per requirement
            min_confidence: Minimum confidence threshold for examples
            
        Returns:
            List of tuples (user_message, assistant_message) for the API connector
        """
        return self.get_category_specific_examples_for_api(
            passage=passage,
            target_categories=target_requirements,
            examples_per_category=examples_per_requirement,
            min_confidence=min_confidence
        )

    def get_available_opp_categories(self) -> List[Dict[str, Any]]:
        """
        Get all available OPP categories from the database.
        
        Returns:
            List of OPP category information
        """
        if hasattr(self.database, 'get_available_opp_categories'):
            return self.database.get_available_opp_categories()
        else:
            # Fallback for dummy database
            return [
                {"label": "First Party Collection/Use", "description": "Privacy practice describing data collection or data use by the company/organization"},
                {"label": "Third Party Sharing/Collection", "description": "Privacy practice describing data sharing with third parties or data collection by third parties"},
                {"label": "User Choice/Control", "description": "Practice that describes general choices and control options available to users"},
                {"label": "User Access, Edit and Deletion", "description": "Privacy practice that allows users to access, edit or delete their data"},
                {"label": "Data Retention", "description": "Privacy practice specifying the retention period for collected user information"},
                {"label": "Data Security", "description": "Practice that describes how users' information is secured and protected"},
                {"label": "Policy Change", "description": "The company/organization's practices concerning policy changes"},
                {"label": "Do Not Track", "description": "Practices that explain if and how Do Not Track signals are honored"},
                {"label": "International and Specific Audiences", "description": "Specific audiences mentioned in the privacy policy with special provisions"}
            ] 