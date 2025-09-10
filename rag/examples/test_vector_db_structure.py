#!/usr/bin/env python3
"""
Test Vector Database Structure

This script tests the structure of the vector database and provides usage examples
without requiring the full SBERT installation.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

def test_database_structure():
    """Test the structure of the vector database."""
    print("Testing Vector Database Structure")
    print("=" * 40)
    
    # Check if vector_db directory exists
    db_dir = Path(__file__).parent / "vector_db"
    if not db_dir.exists():
        print("❌ Vector database directory not found")
        print("   Run 'python build_vector_db.py' to create the database")
        return False
    
    print(f"✅ Vector database directory found: {db_dir}")
    
    # Check database info
    db_info_file = db_dir / "database_info.json"
    if db_info_file.exists():
        with open(db_info_file, 'r') as f:
            db_info = json.load(f)
        
        print(f"\n📊 Database Information:")
        print(f"   Model: {db_info.get('model_name', 'Unknown')}")
        print(f"   Embedding Dimension: {db_info.get('embedding_dimension', 'Unknown')}")
        print(f"   Created: {db_info.get('created_at', 'Unknown')}")
        
        total_examples = db_info.get('total_examples', {})
        total_embeddings = db_info.get('total_embeddings', {})
        
        print(f"\n📈 Statistics:")
        for example_type in ['annotate', 'classify']:
            examples = total_examples.get(example_type, 0)
            embeddings = total_embeddings.get(example_type, 0)
            print(f"   {example_type.capitalize()}: {examples} examples, {embeddings} embeddings")
    else:
        print("❌ Database info file not found")
        return False
    
    # Check embeddings files
    print(f"\n🔍 Checking Files:")
    for example_type in ['annotate', 'classify']:
        embeddings_file = db_dir / f"embeddings_{example_type}.npy"
        metadata_file = db_dir / f"metadata_{example_type}.json"
        
        if embeddings_file.exists():
            try:
                embeddings = np.load(embeddings_file)
                print(f"   ✅ {example_type} embeddings: {embeddings.shape}")
            except Exception as e:
                print(f"   ❌ {example_type} embeddings: Error loading - {e}")
        else:
            print(f"   ❌ {example_type} embeddings: File not found")
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                print(f"   ✅ {example_type} metadata: {len(metadata)} entries")
            except Exception as e:
                print(f"   ❌ {example_type} metadata: Error loading - {e}")
        else:
            print(f"   ❌ {example_type} metadata: File not found")
    
    # Check search interface
    search_file = db_dir / "search_examples.py"
    if search_file.exists():
        print(f"   ✅ Search interface: {search_file}")
    else:
        print(f"   ❌ Search interface: File not found")
    
    return True

def show_usage_examples():
    """Show usage examples for the vector database."""
    print(f"\n📖 Usage Examples")
    print("=" * 40)
    
    print("1. Building the Vector Database:")
    print("   python build_vector_db.py")
    print()
    
    print("2. Using the Search Interface:")
    print("   from pathlib import Path")
    print("   from vector_db.search_examples import ExampleSearcher")
    print()
    print("   db_dir = Path('rag/vector_db')")
    print("   searcher = ExampleSearcher(db_dir)")
    print()
    print("   # Search annotate examples")
    print("   results = searcher.search('We collect personal data for marketing', 'annotate', top_k=5)")
    print()
    print("   # Search both types")
    print("   results = searcher.search_both_types('data retention period', top_k=3)")
    print()
    
    print("3. Direct File Access:")
    print("   import numpy as np")
    print("   import json")
    print()
    print("   # Load embeddings")
    print("   embeddings = np.load('rag/vector_db/embeddings_annotate.npy')")
    print("   metadata = json.load(open('rag/vector_db/metadata_annotate.json'))")
    print()
    print("   # Calculate similarity manually")
    print("   from sentence_transformers import SentenceTransformer")
    print("   model = SentenceTransformer('all-MiniLM-L6-v2')")
    print("   query_embedding = model.encode(['your query text'])")
    print("   similarities = np.dot(embeddings, query_embedding.T).flatten()")
    print("   top_indices = np.argsort(similarities)[::-1][:5]")

def show_integration_example():
    """Show integration example for the pipeline."""
    print(f"\n🔗 Pipeline Integration Example")
    print("=" * 40)
    
    integration_code = '''
# Example integration with your pipeline

class RAGExampleRetriever:
    def __init__(self, db_dir: Path):
        self.searcher = ExampleSearcher(db_dir)
    
    def get_similar_examples(self, passage: str, context: List[Dict], 
                           example_type: str = 'annotate', top_k: int = 5):
        """
        Get similar examples for a given passage and context.
        
        Args:
            passage: The passage text to find similar examples for
            context: The context array for the passage
            example_type: 'annotate' or 'classify'
            top_k: Number of similar examples to retrieve
            
        Returns:
            List of similar examples with metadata
        """
        # Combine context and passage for search
        context_text = ' '.join([item.get('text', '') for item in context])
        search_text = f"{context_text} {passage}".strip()
        
        # Search for similar examples
        results = self.searcher.search(search_text, example_type, top_k)
        
        return results

# Usage in your pipeline
def process_passage(passage: str, context: List[Dict], pipeline_step: str):
    """
    Process a passage with RAG-enhanced examples.
    
    Args:
        passage: The passage to process
        context: The context for the passage
        pipeline_step: 'annotate' or 'classify'
    """
    retriever = RAGExampleRetriever(Path('rag/vector_db'))
    
    # Get similar examples
    similar_examples = retriever.get_similar_examples(
        passage, context, 
        example_type=pipeline_step, 
        top_k=5
    )
    
    # Use examples in your LLM prompt
    example_context = []
    for example in similar_examples:
        example_context.append({
            'passage': example['passage'],
            'context': example['metadata']['context'],
            'annotations': example['metadata']['annotations'],
            'labels': example['metadata']['labels'],
            'similarity_score': example['similarity_score']
        })
    
    # Include example_context in your LLM prompt
    return example_context
'''
    
    print(integration_code)

def main():
    """Main function."""
    print("Vector Database Structure Test")
    print("=" * 40)
    
    # Test structure
    if test_database_structure():
        print(f"\n✅ Vector database structure is valid!")
    else:
        print(f"\n❌ Vector database structure has issues!")
        print("   Please run 'python build_vector_db.py' to create the database")
        return
    
    # Show usage examples
    show_usage_examples()
    
    # Show integration example
    show_integration_example()
    
    print(f"\n🎉 Test completed successfully!")

if __name__ == "__main__":
    main() 