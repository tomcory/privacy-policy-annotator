#!/usr/bin/env python3
"""
Vector Database Demo

This script demonstrates how the vector database would work for RAG-enhanced
GDPR example retrieval, without requiring the actual database to be built.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

class MockVectorDatabase:
    """Mock vector database for demonstration purposes."""
    
    def __init__(self):
        self.examples_dir = Path(__file__).parent / "examples"
        self.annotate_examples = []
        self.classify_examples = []
        self._load_examples()
    
    def _load_examples(self):
        """Load a small sample of examples for demonstration."""
        # Load a few examples from each file for demo
        annotate_file = self.examples_dir / "examples-annotate.json"
        classify_file = self.examples_dir / "examples-classify.json"
        
        if annotate_file.exists():
            with open(annotate_file, 'r', encoding='utf-8') as f:
                all_annotate = json.load(f)
                self.annotate_examples = all_annotate[:5]  # First 5 examples
        
        if classify_file.exists():
            with open(classify_file, 'r', encoding='utf-8') as f:
                all_classify = json.load(f)
                self.classify_examples = all_classify[:5]  # First 5 examples
    
    def extract_text_for_search(self, example: Dict[str, Any]) -> str:
        """Extract text from context and passage for search."""
        context_texts = []
        if 'context' in example and isinstance(example['context'], list):
            for context_item in example['context']:
                if isinstance(context_item, dict) and 'text' in context_item:
                    context_texts.append(context_item['text'])
        
        context_str = " ".join(context_texts)
        passage_str = example.get('passage', '')
        return f"{context_str} {passage_str}".strip()
    
    def simple_similarity_search(self, query: str, example_type: str = 'annotate', top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Simple keyword-based similarity search for demonstration.
        In the real implementation, this would use SBERT embeddings.
        """
        examples = self.annotate_examples if example_type == 'annotate' else self.classify_examples
        
        # Simple keyword matching for demo
        query_lower = query.lower()
        scored_examples = []
        
        for i, example in enumerate(examples):
            text = self.extract_text_for_search(example).lower()
            
            # Simple scoring based on keyword overlap
            score = 0
            for word in query_lower.split():
                if word in text:
                    score += 1
            
            if score > 0:
                scored_examples.append({
                    'index': i,
                    'similarity_score': score,
                    'metadata': {
                        'type': example_type,
                        'example_type': example.get('type', ''),
                        'context': example.get('context', []),
                        'passage': example.get('passage', ''),
                        'annotations': example.get('annotations', []),
                        'labels': example.get('labels', [])
                    },
                    'context_text': ' '.join([item.get('text', '') for item in example.get('context', [])]),
                    'passage': example.get('passage', '')
                })
        
        # Sort by score and return top_k
        scored_examples.sort(key=lambda x: x['similarity_score'], reverse=True)
        return scored_examples[:top_k]

def demonstrate_rag_retrieval():
    """Demonstrate RAG example retrieval."""
    print("🔍 RAG Vector Database Demo")
    print("=" * 50)
    
    # Initialize mock database
    db = MockVectorDatabase()
    
    print(f"📊 Loaded {len(db.annotate_examples)} annotate examples")
    print(f"📊 Loaded {len(db.classify_examples)} classify examples")
    print()
    
    # Demo queries
    demo_queries = [
        "We collect personal data for marketing purposes",
        "Data retention period and storage",
        "Third party data sharing",
        "User rights and access to data"
    ]
    
    for query in demo_queries:
        print(f"🔎 Query: '{query}'")
        print("-" * 40)
        
        # Search annotate examples
        annotate_results = db.simple_similarity_search(query, 'annotate', top_k=2)
        print(f"📝 Annotate Results ({len(annotate_results)} found):")
        for i, result in enumerate(annotate_results, 1):
            print(f"  {i}. Score: {result['similarity_score']}")
            print(f"     Passage: {result['passage'][:80]}...")
            if result['metadata']['annotations']:
                print(f"     Annotations: {len(result['metadata']['annotations'])} items")
            print()
        
        # Search classify examples
        classify_results = db.simple_similarity_search(query, 'classify', top_k=2)
        print(f"🏷️  Classify Results ({len(classify_results)} found):")
        for i, result in enumerate(classify_results, 1):
            print(f"  {i}. Score: {result['similarity_score']}")
            print(f"     Passage: {result['passage'][:80]}...")
            if result['metadata']['labels']:
                print(f"     Labels: {result['metadata']['labels']}")
            print()
        
        print("=" * 50)
        print()

def demonstrate_pipeline_integration():
    """Demonstrate how this would integrate with the pipeline."""
    print("🔗 Pipeline Integration Demo")
    print("=" * 50)
    
    # Mock pipeline step
    def mock_pipeline_step(passage: str, context: List[Dict], step_type: str):
        """Mock pipeline step that uses RAG examples."""
        db = MockVectorDatabase()
        
        # Combine context and passage for search
        context_text = ' '.join([item.get('text', '') for item in context])
        search_text = f"{context_text} {passage}".strip()
        
        # Get similar examples
        similar_examples = db.simple_similarity_search(search_text, step_type, top_k=3)
        
        print(f"🎯 Processing passage for {step_type} step:")
        print(f"   Passage: {passage[:60]}...")
        print(f"   Context: {context_text[:60]}...")
        print(f"   Found {len(similar_examples)} similar examples")
        
        # Format examples for LLM prompt
        example_context = []
        for example in similar_examples:
            example_context.append({
                'passage': example['passage'],
                'context': example['metadata']['context'],
                'annotations': example['metadata']['annotations'],
                'labels': example['metadata']['labels'],
                'similarity_score': example['similarity_score']
            })
        
        print(f"   Example context prepared for LLM prompt")
        return example_context
    
    # Demo pipeline usage
    demo_passages = [
        {
            'passage': "We may share your personal information with third-party service providers who assist us in operating our website and providing services to you.",
            'context': [
                {'text': 'Privacy Policy', 'type': 'h1'},
                {'text': 'Data Sharing', 'type': 'h2'}
            ],
            'step_type': 'annotate'
        },
        {
            'passage': "Your personal data will be retained for a period of 3 years from the date of collection, after which it will be securely deleted.",
            'context': [
                {'text': 'Data Retention Policy', 'type': 'h1'},
                {'text': 'Retention Periods', 'type': 'h2'}
            ],
            'step_type': 'classify'
        }
    ]
    
    for demo in demo_passages:
        print(f"\n📋 Pipeline Demo:")
        result = mock_pipeline_step(
            demo['passage'], 
            demo['context'], 
            demo['step_type']
        )
        print(f"   ✅ RAG examples retrieved and formatted")
        print()

def show_real_implementation_steps():
    """Show the steps to implement the real vector database."""
    print("🚀 Real Implementation Steps")
    print("=" * 50)
    
    steps = [
        "1. Install dependencies: pip install sentence-transformers numpy torch",
        "2. Build vector database: python build_vector_db.py",
        "3. Verify database: python test_vector_db_structure.py",
        "4. Use in pipeline: from vector_db.search_examples import ExampleSearcher",
        "5. Integrate with RAG injector for enhanced LLM prompts"
    ]
    
    for step in steps:
        print(f"   {step}")
    
    print(f"\n📁 Expected file structure after building:")
    print(f"   rag/vector_db/")
    print(f"   ├── embeddings_annotate.npy")
    print(f"   ├── embeddings_classify.npy")
    print(f"   ├── metadata_annotate.json")
    print(f"   ├── metadata_classify.json")
    print(f"   ├── database_info.json")
    print(f"   └── search_examples.py")
    
    print(f"\n⚡ Performance expectations:")
    print(f"   - Embedding generation: ~1000 examples/second")
    print(f"   - Similarity search: ~1000 queries/second")
    print(f"   - Memory usage: ~1.5MB per 1000 examples")

def main():
    """Main demo function."""
    print("🎯 GDPR Examples Vector Database Demo")
    print("=" * 60)
    print("This demo shows how the vector database would work for RAG-enhanced")
    print("GDPR example retrieval in your annotation pipeline.")
    print()
    
    # Check if example files exist
    examples_dir = Path(__file__).parent / "examples"
    if not examples_dir.exists():
        print("❌ Examples directory not found!")
        print("   Please ensure examples-annotate.json and examples-classify.json exist")
        return
    
    # Run demonstrations
    demonstrate_rag_retrieval()
    demonstrate_pipeline_integration()
    show_real_implementation_steps()
    
    print(f"\n🎉 Demo completed!")
    print(f"   To build the real vector database, run: python build_vector_db.py")

if __name__ == "__main__":
    main() 