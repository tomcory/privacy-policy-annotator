#!/usr/bin/env python3
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
        print(f"\n{example_type.upper()} Results:")
        for i, result in enumerate(type_results, 1):
            print(f"  {i}. Score: {result['similarity_score']:.4f}")
            print(f"     Passage: {result['passage'][:100]}...")
            print(f"     Labels: {result['metadata']['labels']}")
            print()

if __name__ == "__main__":
    main()
