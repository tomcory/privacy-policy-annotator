#!/usr/bin/env python3
"""
Vector Database Overview

This script provides a comprehensive overview of the vector database functionality
for GDPR examples RAG-enhanced annotation pipeline.
"""

import json
from pathlib import Path
from typing import Dict, Any

def show_file_structure():
    """Show the complete file structure for the vector database system."""
    print("📁 Vector Database File Structure")
    print("=" * 50)
    
    rag_dir = Path(__file__).parent
    
    # Core scripts
    print("🔧 Core Scripts:")
    core_scripts = [
        "build_vector_db.py",
        "test_vector_db_structure.py", 
        "demo_vector_db.py",
        "overview.py"
    ]
    
    for script in core_scripts:
        script_path = rag_dir / script
        if script_path.exists():
            print(f"   ✅ {script}")
        else:
            print(f"   ❌ {script} (missing)")
    
    # Documentation
    print(f"\n📚 Documentation:")
    docs = [
        "README_vector_db.md",
        "requirements_vector_db.txt"
    ]
    
    for doc in docs:
        doc_path = rag_dir / doc
        if doc_path.exists():
            print(f"   ✅ {doc}")
        else:
            print(f"   ❌ {doc} (missing)")
    
    # Example files
    print(f"\n📋 Example Files:")
    examples_dir = rag_dir / "examples"
    if examples_dir.exists():
        example_files = [
            "examples-annotate.json",
            "examples-classify.json"
        ]
        
        for example_file in example_files:
            file_path = examples_dir / example_file
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"   ✅ {example_file} ({size_mb:.1f}MB)")
            else:
                print(f"   ❌ {example_file} (missing)")
    else:
        print(f"   ❌ examples/ directory (missing)")
    
    # Generated files (after building)
    print(f"\n🏗️  Generated Files (after building):")
    vector_db_dir = rag_dir / "vector_db"
    if vector_db_dir.exists():
        generated_files = [
            "embeddings_annotate.npy",
            "embeddings_classify.npy", 
            "metadata_annotate.json",
            "metadata_classify.json",
            "database_info.json",
            "search_examples.py"
        ]
        
        for gen_file in generated_files:
            file_path = vector_db_dir / gen_file
            if file_path.exists():
                if file_path.suffix == '.npy':
                    # For numpy files, show shape info
                    try:
                        import numpy as np
                        data = np.load(file_path)
                        print(f"   ✅ {gen_file} (shape: {data.shape})")
                    except:
                        print(f"   ✅ {gen_file}")
                else:
                    size_kb = file_path.stat().st_size / 1024
                    print(f"   ✅ {gen_file} ({size_kb:.1f}KB)")
            else:
                print(f"   ❌ {gen_file} (not built yet)")
    else:
        print(f"   ❌ vector_db/ directory (not built yet)")
        print(f"   💡 Run 'python build_vector_db.py' to create")

def show_example_statistics():
    """Show statistics about the example files."""
    print(f"\n📊 Example File Statistics")
    print("=" * 50)
    
    examples_dir = Path(__file__).parent / "examples"
    
    if not examples_dir.exists():
        print("❌ Examples directory not found")
        return
    
    for example_type in ['annotate', 'classify']:
        file_path = examples_dir / f"examples-{example_type}.json"
        
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    examples = json.load(f)
                
                # Count examples with different types
                type_counts = {}
                annotation_counts = []
                label_counts = []
                
                for example in examples:
                    # Count example types
                    ex_type = example.get('type', 'unknown')
                    type_counts[ex_type] = type_counts.get(ex_type, 0) + 1
                    
                    # Count annotations (for annotate examples)
                    if example_type == 'annotate' and 'annotations' in example:
                        annotation_counts.append(len(example['annotations']))
                    
                    # Count labels (for classify examples)
                    if example_type == 'classify' and 'labels' in example:
                        label_counts.append(len(example['labels']))
                
                print(f"📝 {example_type.capitalize()} Examples:")
                print(f"   Total examples: {len(examples):,}")
                print(f"   File size: {file_path.stat().st_size / (1024 * 1024):.1f}MB")
                
                if type_counts:
                    print(f"   Example types:")
                    for ex_type, count in sorted(type_counts.items()):
                        print(f"     - {ex_type}: {count:,}")
                
                if annotation_counts:
                    avg_annotations = sum(annotation_counts) / len(annotation_counts)
                    print(f"   Average annotations per example: {avg_annotations:.1f}")
                
                if label_counts:
                    avg_labels = sum(label_counts) / len(label_counts)
                    print(f"   Average labels per example: {avg_labels:.1f}")
                
                print()
                
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
        else:
            print(f"❌ {file_path} not found")

def show_usage_workflow():
    """Show the complete usage workflow."""
    print(f"🔄 Complete Usage Workflow")
    print("=" * 50)
    
    workflow_steps = [
        {
            "step": 1,
            "title": "Install Dependencies",
            "command": "pip install sentence-transformers numpy torch transformers",
            "description": "Install required packages for SBERT embeddings"
        },
        {
            "step": 2,
            "title": "Build Vector Database",
            "command": "python build_vector_db.py",
            "description": "Generate embeddings and create searchable database"
        },
        {
            "step": 3,
            "title": "Verify Database",
            "command": "python test_vector_db_structure.py",
            "description": "Check that all files are properly created"
        },
        {
            "step": 4,
            "title": "Test Demo",
            "command": "python demo_vector_db.py",
            "description": "See how the system works with sample queries"
        },
        {
            "step": 5,
            "title": "Integrate with Pipeline",
            "command": "from vector_db.search_examples import ExampleSearcher",
            "description": "Use in your annotation pipeline for RAG-enhanced prompts"
        }
    ]
    
    for step_info in workflow_steps:
        print(f"{step_info['step']}. {step_info['title']}")
        print(f"   Command: {step_info['command']}")
        print(f"   Purpose: {step_info['description']}")
        print()

def show_integration_example():
    """Show a complete integration example."""
    print(f"🔗 Complete Integration Example")
    print("=" * 50)
    
    integration_code = '''
# Complete integration example for your pipeline

from pathlib import Path
from typing import List, Dict, Any
from vector_db.search_examples import ExampleSearcher

class RAGEnhancedPipeline:
    def __init__(self, vector_db_dir: Path):
        """Initialize the RAG-enhanced pipeline."""
        self.searcher = ExampleSearcher(vector_db_dir)
    
    def process_annotate_step(self, passage: str, context: List[Dict]) -> Dict[str, Any]:
        """Process a passage for annotation with RAG examples."""
        # Get similar examples
        similar_examples = self.searcher.search(
            self._combine_context_passage(context, passage),
            'annotate',
            top_k=5
        )
        
        # Format examples for LLM prompt
        example_context = self._format_examples_for_prompt(similar_examples)
        
        # Include in your LLM prompt
        prompt = self._build_annotate_prompt(passage, context, example_context)
        
        return {
            'passage': passage,
            'context': context,
            'similar_examples': similar_examples,
            'prompt': prompt
        }
    
    def process_classify_step(self, passage: str, context: List[Dict]) -> Dict[str, Any]:
        """Process a passage for classification with RAG examples."""
        # Get similar examples
        similar_examples = self.searcher.search(
            self._combine_context_passage(context, passage),
            'classify',
            top_k=5
        )
        
        # Format examples for LLM prompt
        example_context = self._format_examples_for_prompt(similar_examples)
        
        # Include in your LLM prompt
        prompt = self._build_classify_prompt(passage, context, example_context)
        
        return {
            'passage': passage,
            'context': context,
            'similar_examples': similar_examples,
            'prompt': prompt
        }
    
    def _combine_context_passage(self, context: List[Dict], passage: str) -> str:
        """Combine context and passage for search."""
        context_text = ' '.join([item.get('text', '') for item in context])
        return f"{context_text} {passage}".strip()
    
    def _format_examples_for_prompt(self, examples: List[Dict]) -> str:
        """Format examples for inclusion in LLM prompt."""
        formatted_examples = []
        for example in examples:
            formatted_examples.append(f"""
Example {len(formatted_examples) + 1} (Similarity: {example['similarity_score']:.3f}):
Context: {' '.join([item.get('text', '') for item in example['metadata']['context']])}
Passage: {example['passage']}
Annotations: {example['metadata']['annotations']}
Labels: {example['metadata']['labels']}
""")
        return '\n'.join(formatted_examples)
    
    def _build_annotate_prompt(self, passage: str, context: List[Dict], examples: str) -> str:
        """Build annotation prompt with RAG examples."""
        return f"""
You are an expert GDPR compliance annotator. Based on the following examples and the passage to annotate, identify all GDPR transparency requirements.

EXAMPLES:
{examples}

PASSAGE TO ANNOTATE:
Context: {' '.join([item.get('text', '') for item in context])}
Text: {passage}

Please identify all GDPR transparency requirements in the passage above.
"""
    
    def _build_classify_prompt(self, passage: str, context: List[Dict], examples: str) -> str:
        """Build classification prompt with RAG examples."""
        return f"""
You are an expert GDPR compliance classifier. Based on the following examples and the passage to classify, identify which GDPR transparency requirements are present.

EXAMPLES:
{examples}

PASSAGE TO CLASSIFY:
Context: {' '.join([item.get('text', '') for item in context])}
Text: {passage}

Please classify which GDPR transparency requirements are present in the passage above.
"""

# Usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RAGEnhancedPipeline(Path('rag/vector_db'))
    
    # Process a passage for annotation
    result = pipeline.process_annotate_step(
        "We collect your personal data for marketing and analytics purposes",
        [
            {'text': 'Privacy Policy', 'type': 'h1'},
            {'text': 'Data Collection', 'type': 'h2'}
        ]
    )
    
    print("RAG-enhanced prompt generated successfully!")
'''
    
    print(integration_code)

def main():
    """Main overview function."""
    print("🎯 GDPR Examples Vector Database - Complete Overview")
    print("=" * 60)
    print("This overview shows the complete vector database system for")
    print("RAG-enhanced GDPR example retrieval in your annotation pipeline.")
    print()
    
    # Show file structure
    show_file_structure()
    
    # Show example statistics
    show_example_statistics()
    
    # Show usage workflow
    show_usage_workflow()
    
    # Show integration example
    show_integration_example()
    
    print("🎉 Overview completed!")
    print("   Next steps:")
    print("   1. Install dependencies: pip install sentence-transformers numpy torch")
    print("   2. Build database: python build_vector_db.py")
    print("   3. Test system: python demo_vector_db.py")
    print("   4. Integrate with your pipeline")

if __name__ == "__main__":
    main() 