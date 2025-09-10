import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

@dataclass
class OPPAnnotationExample:
    """Represents an OPP example annotation with metadata."""
    passage: str
    requirement: str
    annotation_value: str
    performed: bool
    confidence: float
    source_policy: str
    difficulty_level: str = "normal"  # normal, difficult, edge_case
    similarity_score: float = 0.0
    context: List[Dict] = None
    annotations: List[Dict] = None
    labels: List[str] = None


class OPPVectorDatabaseConnector:
    """
    Vector database connector for OPP (Online Privacy Policy) data that uses SBERT embeddings 
    for similarity search to retrieve requirement-specific examples from the built vector database.
    """
    
    def __init__(self, db_dir: Path = None):
        """
        Initialize the OPP vector database connector.
        
        Args:
            db_dir: Directory containing the vector database files
        """
        if db_dir is None:
            # Default to the built OPP vector database location
            db_dir = Path(__file__).parent.parent.parent / "rag" / "examples_opp" / "vector_db"
        
        self.db_dir = Path(db_dir)
        self.model = None
        self.embeddings = {}
        self.metadata = {}
        self.db_info = {}
        
        self._load_database()
    
    def _load_database(self):
        """Load the OPP vector database."""
        logger.info(f"Loading OPP vector database from {self.db_dir}")
        
        # Load database info
        db_info_file = self.db_dir / "database_info.json"
        if db_info_file.exists():
            with open(db_info_file, 'r') as f:
                self.db_info = json.load(f)
            logger.info(f"Loaded OPP database info: {self.db_info}")
        
        # Load SBERT model
        model_name = self.db_info.get('model_name', 'all-MiniLM-L6-v2')
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded SBERT model: {model_name}")
        
        # Load embeddings and metadata for each type
        for example_type in ['annotate', 'classify']:
            # Load embeddings
            embeddings_file = self.db_dir / f"embeddings_{example_type}.npy"
            if embeddings_file.exists():
                self.embeddings[example_type] = np.load(embeddings_file)
                logger.info(f"Loaded {example_type} embeddings: {self.embeddings[example_type].shape}")
            
            # Load metadata
            metadata_file = self.db_dir / f"metadata_{example_type}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata[example_type] = json.load(f)
                logger.info(f"Loaded {example_type} metadata: {len(self.metadata[example_type])} entries")
    
    def search_similar_examples(
        self, 
        query_text: str, 
        example_type: str = 'annotate', 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar OPP examples using SBERT embeddings.
        
        Args:
            query_text: Text to search for similar examples
            example_type: 'annotate' or 'classify'
            top_k: Number of top results to return
            
        Returns:
            List of similar examples with metadata and similarity scores
        """
        if example_type not in self.embeddings:
            logger.warning(f"No embeddings found for example type: {example_type}")
            return []
        
        # Encode query
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        
        # Calculate similarities
        embeddings = self.embeddings[example_type]
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            metadata = self.metadata[example_type][idx]
            result = {
                'index': int(idx),
                'similarity_score': float(similarities[idx]),
                'metadata': metadata,
                'passage': metadata['passage']
            }
            
            # Add type-specific fields
            if example_type == 'classify':
                result['labels'] = metadata.get('labels', [])
            else:  # annotate
                result['annotations'] = metadata.get('annotations', [])
            
            results.append(result)
        
        return results
    
    def retrieve_examples(
        self, 
        requirement: str, 
        passage_text: str = None, 
        max_examples: int = 3,
        min_confidence: float = 0.5,  # Lower default threshold
        example_type: str = 'annotate'
    ) -> List[OPPAnnotationExample]:
        """
        Retrieve OPP examples for a specific requirement, optionally filtered by passage similarity.
        
        Args:
            requirement: The OPP requirement to find examples for
            passage_text: Optional passage text for similarity filtering
            max_examples: Maximum number of examples to return
            min_confidence: Minimum confidence threshold
            example_type: Type of examples to retrieve ('annotate' or 'classify')
            
        Returns:
            List of OPPAnnotationExample objects
        """
        all_examples = []
        
        # If we have passage text, use similarity search
        if passage_text:
            # Search only the specified example type
            results = self.search_similar_examples(passage_text, example_type, max_examples * 2)
                
            for result in results:
                if example_type == 'annotate':
                    # Handle annotate examples
                    annotations = result['metadata'].get('annotations', [])
                    if annotations:
                        for annotation in annotations:
                            example = OPPAnnotationExample(
                                passage=result['passage'],
                                requirement=requirement,  # Use the requested requirement
                                annotation_value=annotation.get('value', ''),
                                performed=True,
                                confidence=result['similarity_score'],
                                source_policy=result['metadata'].get('source_file', 'unknown'),
                                similarity_score=result['similarity_score'],
                                context=result['metadata'].get('context', []),
                                annotations=result['metadata'].get('annotations', [])
                            )
                            all_examples.append(example)
                    else:
                        # Fallback if no annotations found
                        example = OPPAnnotationExample(
                            passage=result['passage'],
                            requirement=requirement,
                            annotation_value='',
                            performed=False,
                            confidence=result['similarity_score'],
                            source_policy=result['metadata'].get('source_file', 'unknown'),
                            similarity_score=result['similarity_score'],
                            context=result['metadata'].get('context', [])
                        )
                        all_examples.append(example)
                
                elif example_type == 'classify':
                    # Handle classify examples
                    labels = result['metadata'].get('labels', [])
                    if labels:
                        for label in labels:
                            example = OPPAnnotationExample(
                                passage=result['passage'],
                                requirement=label,
                                annotation_value=label,
                                performed=True,
                                confidence=result['similarity_score'],
                                source_policy=result['metadata'].get('source_file', 'unknown'),
                                similarity_score=result['similarity_score'],
                                context=result['metadata'].get('context', []),
                                labels=result['metadata'].get('labels', [])
                            )
                            all_examples.append(example)
        
        # If no passage text or no examples found, get general examples for the requirement
        if not all_examples:
            # Search for examples that mention the requirement
            query_text = requirement.lower()
            
            results = self.search_similar_examples(query_text, example_type, max_examples)
            
            for result in results:
                if example_type == 'annotate':
                    annotations = result['metadata'].get('annotations', [])
                    if annotations:
                        for annotation in annotations:
                            # For annotate examples, use the requirement parameter since labels are often empty
                            example = OPPAnnotationExample(
                                passage=result['passage'],
                                requirement=requirement,  # Use the requested requirement
                                annotation_value=annotation.get('value', ''),
                                performed=True,
                                confidence=result['similarity_score'],
                                source_policy=result['metadata'].get('source_file', 'unknown'),
                                similarity_score=result['similarity_score'],
                                context=result['metadata'].get('context', []),
                                annotations=result['metadata'].get('annotations', [])
                            )
                            all_examples.append(example)
                
                elif example_type == 'classify':
                    # For classify examples, check if the requirement matches any labels
                    labels = result['metadata'].get('labels', [])
                    for label in labels:
                        if requirement.lower() in label.lower():
                            example = OPPAnnotationExample(
                                passage=result['passage'],
                                requirement=label,
                                annotation_value=label,
                                performed=True,
                                confidence=result['similarity_score'],
                                source_policy=result['metadata'].get('source_file', 'unknown'),
                                similarity_score=result['similarity_score'],
                                context=result['metadata'].get('context', []),
                                labels=result['metadata'].get('labels', [])
                            )
                            all_examples.append(example)
        
        # Filter by confidence and sort by similarity score
        filtered_examples = [
            ex for ex in all_examples 
            if ex.confidence >= min_confidence
        ]
        
        # Sort by similarity score (descending) and return top results
        filtered_examples.sort(key=lambda x: x.similarity_score, reverse=True)
        return filtered_examples[:max_examples]
    
    def retrieve_examples_for_requirements(
        self, 
        requirements: List[str], 
        passage_text: str = None,
        max_examples_per_requirement: int = 2,
        min_confidence: float = 0.5,  # Lower default threshold
        example_type: str = 'annotate'
    ) -> Dict[str, List[OPPAnnotationExample]]:
        """
        Retrieve OPP examples for multiple requirements.
        
        Args:
            requirements: List of OPP requirements to find examples for
            passage_text: Optional passage text for similarity filtering
            max_examples_per_requirement: Maximum examples per requirement
            
        Returns:
            Dictionary mapping requirements to lists of examples
        """
        examples_by_requirement = {}
        
        for requirement in requirements:
            examples = self.retrieve_examples(
                requirement=requirement,
                passage_text=passage_text,
                max_examples=max_examples_per_requirement,
                min_confidence=min_confidence,
                example_type=example_type
            )
            examples_by_requirement[requirement] = examples
        
        return examples_by_requirement
    
    def retrieve_legal_background(self, requirement: str) -> Optional[str]:
        """
        Retrieve legal background for an OPP requirement from the legal database.
        
        Args:
            requirement: The OPP requirement to get background for
            
        Returns:
            Legal background text or None
        """
        try:
            # Load the database index to find the correct file
            index_path = Path(__file__).parent.parent.parent / "rag" / "legal_opp" / "opp_database_index.json"
            if not index_path.exists():
                logger.warning(f"OPP legal database index not found at {index_path}")
                return None
            
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # Find the requirement in the index
            requirement_entry = None
            for req in index_data.get('categories_overview', []):
                if req.get('label') == requirement:
                    requirement_entry = req
                    break
            
            if not requirement_entry:
                logger.warning(f"OPP requirement '{requirement}' not found in legal database index")
                return None
            
            # Load the specific requirement file
            req_file = requirement_entry.get('file')
            if not req_file:
                logger.warning(f"No file specified for OPP requirement '{requirement}'")
                return None
            
            req_path = Path(__file__).parent.parent.parent / "rag" / "legal_opp" / req_file
            if not req_path.exists():
                logger.warning(f"OPP legal requirement file not found: {req_path}")
                return None
            
            with open(req_path, 'r', encoding='utf-8') as f:
                req_data = json.load(f)
            
            # Extract the relevant legal background information
            background_parts = []
            
            # Add description
            description = req_data.get('description')
            if description:
                background_parts.append(f"Description: {description}")
            
            # Add usage guidelines
            usage = req_data.get('usage')
            if usage:
                background_parts.append(f"Usage Guidelines: {usage}")
            
            if background_parts:
                return "\n\n".join(background_parts)
            else:
                return f"Legal background for {requirement} not available."
                
        except Exception as e:
            logger.error(f"Error retrieving legal background for '{requirement}': {e}")
            return f"Legal background for {requirement} not available."
    
    def retrieve_opp_category_info(self, requirement: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve OPP category information for a requirement from the legal database.
        
        Args:
            requirement: The OPP requirement to get category info for
            
        Returns:
            OPP category information or None
        """
        try:
            # Look for the category file that matches the requirement
            legal_dir = Path(__file__).parent.parent.parent / "rag" / "legal_opp"
            
            # Try to find the category file by searching through all opp_cat files
            for cat_file in legal_dir.glob("opp_cat_*.json"):
                with open(cat_file, 'r', encoding='utf-8') as f:
                    cat_data = json.load(f)
                
                # Check if this category matches the requirement
                if cat_data.get('label') == requirement:
                    return {
                        'category_file': cat_file.name,
                        'label': cat_data.get('label'),
                        'description': cat_data.get('description'),
                        'usage': cat_data.get('usage')
                    }
            
            logger.warning(f"OPP category for requirement '{requirement}' not found in legal database")
            return None
                
        except Exception as e:
            logger.error(f"Error retrieving OPP category info for '{requirement}': {e}")
            return None
    
    def get_available_opp_categories(self) -> List[Dict[str, Any]]:
        """
        Get all available OPP categories from the legal database.
        
        Returns:
            List of OPP category information
        """
        categories = []
        try:
            legal_dir = Path(__file__).parent.parent.parent / "rag" / "legal_opp"
            
            for cat_file in sorted(legal_dir.glob("opp_cat_*.json")):
                with open(cat_file, 'r', encoding='utf-8') as f:
                    cat_data = json.load(f)
                
                categories.append({
                    'category_file': cat_file.name,
                    'label': cat_data.get('label'),
                    'description': cat_data.get('description'),
                    'usage': cat_data.get('usage')
                })
            
            logger.info(f"Found {len(categories)} OPP categories")
            return categories
                
        except Exception as e:
            logger.error(f"Error retrieving OPP categories: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get statistics about the OPP vector database."""
        stats = {
            'total_examples': {},
            'embedding_dimensions': {},
            'model_name': self.db_info.get('model_name', 'unknown'),
            'database_type': 'OPP'
        }
        
        for example_type in ['annotate', 'classify']:
            if example_type in self.embeddings:
                stats['total_examples'][example_type] = self.embeddings[example_type].shape[0]
                stats['embedding_dimensions'][example_type] = self.embeddings[example_type].shape[1]
        
        return stats
    
    def add_example(self, example: OPPAnnotationExample):
        """
        Add a new OPP example to the database.
        Note: This would require rebuilding the vector database in a real implementation.
        """
        logger.warning("Adding OPP examples requires rebuilding the vector database. Use build_vector_db.py instead.")
        # In a real implementation, this would add to the database and update embeddings 