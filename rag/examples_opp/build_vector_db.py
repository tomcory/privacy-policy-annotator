#!/usr/bin/env python3
"""
Vector Database Builder for GDPR Examples

This script builds a vector database from the examples in examples-annotate.jsonl
and examples-classify.jsonl using SBERT embeddings for similarity search.
The embeddings are created from the "context" array and "passage" string fields.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

class VectorDatabaseBuilder:
    """Builds and manages vector database for GDPR examples."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the vector database builder.
        
        Args:
            model_name: SBERT model to use for embeddings
        """
        self.model_name = model_name
        self.model = None
        self.examples_data = {
            'annotate': [],
            'classify': []
        }
        self.embeddings = {
            'annotate': [],
            'classify': []
        }
        self.metadata = {
            'annotate': [],
            'classify': []
        }
        
        if SBERT_AVAILABLE:
            logger.info(f"Loading SBERT model: {model_name}")
            self.model = SentenceTransformer(model_name)
        else:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
    
    def load_examples(self, examples_dir: Path) -> None:
        """
        Load examples from both JSONL files.
        
        Args:
            examples_dir: Directory containing the example files
        """
        logger.info("Loading example files...")
        
        # Load annotate examples
        annotate_file = examples_dir / "examples-annotate.jsonl"
        if annotate_file.exists():
            self.examples_data['annotate'] = []
            with open(annotate_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            example = json.loads(line)
                            self.examples_data['annotate'].append(example)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON on line {line_num} in {annotate_file}: {e}")
            logger.info(f"Loaded {len(self.examples_data['annotate'])} annotate examples")
        else:
            logger.warning(f"Annotate examples file not found: {annotate_file}")
        
        # Load classify examples
        classify_file = examples_dir / "examples-classify.jsonl"
        if classify_file.exists():
            self.examples_data['classify'] = []
            with open(classify_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            example = json.loads(line)
                            self.examples_data['classify'].append(example)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON on line {line_num} in {classify_file}: {e}")
            logger.info(f"Loaded {len(self.examples_data['classify'])} classify examples")
        else:
            logger.warning(f"Classify examples file not found: {classify_file}")
    
    def extract_text_for_embedding(self, example: Dict[str, Any]) -> str:
        """
        Extract text from context array and passage for embedding.
        
        Args:
            example: Example dictionary containing context and passage
            
        Returns:
            Combined text string for embedding
        """
        # Extract text from context array
        context_texts = []
        if 'context' in example and isinstance(example['context'], list):
            for context_item in example['context']:
                if isinstance(context_item, dict) and 'text' in context_item:
                    context_texts.append(context_item['text'])
        
        # Combine context texts
        context_str = " ".join(context_texts)
        
        # Get passage text
        passage_str = example.get('passage', '')
        
        # Combine context and passage
        combined_text = f"{context_str} {passage_str}".strip()
        
        return combined_text
    
    def build_embeddings(self) -> None:
        """Build embeddings for all examples."""
        logger.info("Building embeddings...")
        
        for example_type in ['annotate', 'classify']:
            if not self.examples_data[example_type]:
                logger.warning(f"No {example_type} examples to process")
                continue
            
            logger.info(f"Processing {example_type} examples...")
            
            # Extract texts for embedding
            texts = []
            metadata_list = []
            
            for i, example in enumerate(self.examples_data[example_type]):
                text = self.extract_text_for_embedding(example)
                texts.append(text)
                
                # Store metadata with file reference
                metadata = {
                    'index': i,
                    'type': example_type,
                    'example_type': example.get('type', ''),
                    'context': example.get('context', []),
                    'passage': example.get('passage', ''),
                    'annotations': example.get('annotations', []),
                    'labels': example.get('labels', []),
                    'source_file': f"examples-{example_type}.jsonl",
                    'source_index': i  # Index in the original file
                }
                metadata_list.append(metadata)
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} {example_type} examples...")
            embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            
            self.embeddings[example_type] = embeddings
            self.metadata[example_type] = metadata_list
            
            logger.info(f"Generated {embeddings.shape[0]} embeddings for {example_type} examples")
    
    def save_vector_database(self, output_dir: Path) -> None:
        """
        Save the vector database to files.
        
        Args:
            output_dir: Directory to save the vector database files
        """
        output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Saving vector database to {output_dir}")
        
        # Save embeddings
        for example_type in ['annotate', 'classify']:
            if len(self.embeddings[example_type]) > 0:
                # Save embeddings as numpy array
                embeddings_file = output_dir / f"embeddings_{example_type}.npy"
                np.save(embeddings_file, self.embeddings[example_type])
                logger.info(f"Saved {example_type} embeddings: {embeddings_file}")
                
                # Save metadata
                metadata_file = output_dir / f"metadata_{example_type}.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata[example_type], f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {example_type} metadata: {metadata_file}")
        
        # Save combined database info
        db_info = {
            'model_name': self.model_name,
            'embedding_dimension': self.embeddings['annotate'].shape[1] if len(self.embeddings['annotate']) > 0 else None,
            'total_examples': {
                'annotate': len(self.examples_data['annotate']),
                'classify': len(self.examples_data['classify'])
            },
            'total_embeddings': {
                'annotate': self.embeddings['annotate'].shape[0] if len(self.embeddings['annotate']) > 0 else 0,
                'classify': self.embeddings['classify'].shape[0] if len(self.embeddings['classify']) > 0 else 0
            },
            'created_at': '2024-01-15'
        }
        
        db_info_file = output_dir / "database_info.json"
        with open(db_info_file, 'w', encoding='utf-8') as f:
            json.dump(db_info, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved database info: {db_info_file}")
    
    def create_search_interface(self, output_dir: Path) -> None:
        """
        Create a search interface script for the vector database.
        
        Args:
            output_dir: Directory containing the vector database files
        """
        search_script = output_dir / "search_examples.py"
        
        script_content = '''#!/usr/bin/env python3
"""
Example Search Interface

This script provides a search interface for the GDPR examples vector database.
It uses SBERT embeddings to find the most similar examples.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExampleSearcher:
    """Search interface for GDPR examples vector database."""
    
    def __init__(self, db_dir: Path, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the searcher.
        
        Args:
            db_dir: Directory containing the vector database files
            model_name: SBERT model name
        """
        self.db_dir = db_dir
        self.model = SentenceTransformer(model_name)
        self.embeddings = {}
        self.metadata = {}
        self.db_info = {}
        
        self._load_database()
    
    def _load_database(self):
        """Load the vector database."""
        # Load database info
        db_info_file = self.db_dir / "database_info.json"
        if db_info_file.exists():
            with open(db_info_file, 'r') as f:
                self.db_info = json.load(f)
        
        # Load embeddings and metadata for each type
        for example_type in ['annotate', 'classify']:
            embeddings_file = self.db_dir / f"embeddings_{example_type}.npy"
            metadata_file = self.db_dir / f"metadata_{example_type}.json"
            
            if embeddings_file.exists() and metadata_file.exists():
                self.embeddings[example_type] = np.load(embeddings_file)
                with open(metadata_file, 'r') as f:
                    self.metadata[example_type] = json.load(f)
                
                logger.info(f"Loaded {example_type} database: {self.embeddings[example_type].shape[0]} examples")
    
    def search(self, query: str, example_type: str = 'annotate', top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar examples.
        
        Args:
            query: Search query text
            example_type: Type of examples to search ('annotate' or 'classify')
            top_k: Number of top results to return
            
        Returns:
            List of similar examples with similarity scores
        """
        if example_type not in self.embeddings:
            raise ValueError(f"Example type '{example_type}' not found in database")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Calculate similarities
        similarities = np.dot(self.embeddings[example_type], query_embedding.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Prepare results
        results = []
        for idx in top_indices:
            result = {
                'index': int(idx),
                'similarity_score': float(similarities[idx]),
                'metadata': self.metadata[example_type][idx],
                'context_text': ' '.join([item.get('text', '') for item in self.metadata[example_type][idx]['context']]),
                'passage': self.metadata[example_type][idx]['passage']
            }
            results.append(result)
        
        return results
    
    def search_both_types(self, query: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search both annotate and classify examples.
        
        Args:
            query: Search query text
            top_k: Number of top results to return per type
            
        Returns:
            Dictionary with results for both types
        """
        results = {}
        for example_type in ['annotate', 'classify']:
            if example_type in self.embeddings:
                results[example_type] = self.search(query, example_type, top_k)
        
        return results

def main():
    """Example usage of the search interface."""
    db_dir = Path(__file__).parent
    searcher = ExampleSearcher(db_dir)
    
    # Example search
    query = "We collect your personal data for marketing purposes"
    print(f"Searching for: '{query}'")
    
    results = searcher.search_both_types(query, top_k=3)
    
    for example_type, type_results in results.items():
        print(f"\\n{example_type.upper()} Results:")
        for i, result in enumerate(type_results, 1):
            print(f"  {i}. Score: {result['similarity_score']:.4f}")
            print(f"     Passage: {result['passage'][:100]}...")
            print(f"     Labels: {result['metadata']['labels']}")
            print()

if __name__ == "__main__":
    main()
'''
        
        with open(search_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Make it executable
        search_script.chmod(0o755)
        logger.info(f"Created search interface: {search_script}")

def main():
    """Main function to build the vector database."""
    logger.info("Starting vector database construction...")
    
    # Get paths
    script_dir = Path(__file__).parent
    examples_dir = script_dir  # Examples are in the same directory as the script
    output_dir = script_dir / "vector_db"
    
    # Check if example files exist
    annotate_file = examples_dir / "examples-annotate.jsonl"
    classify_file = examples_dir / "examples-classify.jsonl"
    
    if not annotate_file.exists():
        logger.error(f"Annotate examples file not found: {annotate_file}")
        return
    
    if not classify_file.exists():
        logger.error(f"Classify examples file not found: {classify_file}")
        return
    
    # Initialize builder
    try:
        builder = VectorDatabaseBuilder()
    except ImportError as e:
        logger.error(f"Failed to initialize vector database builder: {e}")
        logger.error("Please install sentence-transformers: pip install sentence-transformers")
        return
    
    # Load examples
    builder.load_examples(examples_dir)
    
    # Build embeddings
    builder.build_embeddings()
    
    # Save vector database
    builder.save_vector_database(output_dir)
    
    # Create search interface
    builder.create_search_interface(output_dir)
    
    logger.info("Vector database construction completed successfully!")

if __name__ == "__main__":
    main() 