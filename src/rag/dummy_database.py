import json
import random
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class AnnotationExample:
    """Represents an example annotation with metadata."""
    passage: str
    requirement: str
    annotation_value: str
    performed: bool
    confidence: float
    source_policy: str
    difficulty_level: str = "normal"  # normal, difficult, edge_case


class DummyDatabaseConnector:
    """
    A dummy database connector that simulates a vector database with requirement-specific examples.
    In a real implementation, this would connect to a vector database like FAISS, Chroma, or Pinecone.
    """
    
    def __init__(self):
        self.examples = self._create_dummy_examples()
        
    def _create_dummy_examples(self) -> List[AnnotationExample]:
        """Create dummy examples for each GDPR requirement."""
        examples = []
        
        # Data Categories examples
        examples.extend([
            AnnotationExample(
                passage="We collect your email address, phone number, and postal address for account management.",
                requirement="Data Categories",
                annotation_value="email address",
                performed=True,
                confidence=0.95,
                source_policy="example_policy_1"
            ),
            AnnotationExample(
                passage="Your device identifiers, IP address, and browser information are collected for analytics.",
                requirement="Data Categories",
                annotation_value="device identifiers",
                performed=True,
                confidence=0.92,
                source_policy="example_policy_2"
            ),
            AnnotationExample(
                passage="We do not collect sensitive personal data such as health information or financial records.",
                requirement="Data Categories",
                annotation_value="sensitive personal data",
                performed=False,
                confidence=0.88,
                source_policy="example_policy_3"
            )
        ])
        
        # Processing Purpose examples
        examples.extend([
            AnnotationExample(
                passage="Your data is used to provide customer support and respond to your inquiries.",
                requirement="Processing Purpose",
                annotation_value="provide customer support",
                performed=True,
                confidence=0.94,
                source_policy="example_policy_1"
            ),
            AnnotationExample(
                passage="We process your information to improve our services and develop new features.",
                requirement="Processing Purpose",
                annotation_value="improve our services",
                performed=True,
                confidence=0.91,
                source_policy="example_policy_2"
            ),
            AnnotationExample(
                passage="Your data is used for marketing purposes, including sending promotional emails.",
                requirement="Processing Purpose",
                annotation_value="marketing purposes",
                performed=True,
                confidence=0.89,
                source_policy="example_policy_4"
            )
        ])
        
        # Legal Basis examples
        examples.extend([
            AnnotationExample(
                passage="We process your data based on your explicit consent for marketing communications.",
                requirement="Legal Basis for Processing",
                annotation_value="your explicit consent",
                performed=True,
                confidence=0.96,
                source_policy="example_policy_1"
            ),
            AnnotationExample(
                passage="Data processing is necessary for the performance of our service contract with you.",
                requirement="Legal Basis for Processing",
                annotation_value="performance of our service contract",
                performed=True,
                confidence=0.93,
                source_policy="example_policy_2"
            ),
            AnnotationExample(
                passage="We rely on legitimate interests to process data for fraud prevention.",
                requirement="Legal Basis for Processing",
                annotation_value="legitimate interests",
                performed=True,
                confidence=0.90,
                source_policy="example_policy_3"
            )
        ])
        
        # Right to Access examples
        examples.extend([
            AnnotationExample(
                passage="You have the right to request access to all personal data we hold about you.",
                requirement="Right to Access",
                annotation_value="right to request access",
                performed=True,
                confidence=0.95,
                source_policy="example_policy_1"
            ),
            AnnotationExample(
                passage="You may request a copy of your personal data by contacting our data protection officer.",
                requirement="Right to Access",
                annotation_value="request a copy of your personal data",
                performed=True,
                confidence=0.92,
                source_policy="example_policy_2"
            ),
            AnnotationExample(
                passage="We will provide you with access to your data within 30 days of your request.",
                requirement="Right to Access",
                annotation_value="within 30 days of your request",
                performed=True,
                confidence=0.88,
                source_policy="example_policy_3"
            )
        ])
        
        # Data Retention examples
        examples.extend([
            AnnotationExample(
                passage="We retain your personal data for 3 years after your last interaction with us.",
                requirement="Data Retention Period",
                annotation_value="3 years after your last interaction",
                performed=True,
                confidence=0.94,
                source_policy="example_policy_1"
            ),
            AnnotationExample(
                passage="Account data is kept for 7 years to comply with legal obligations.",
                requirement="Data Retention Period",
                annotation_value="7 years to comply with legal obligations",
                performed=True,
                confidence=0.91,
                source_policy="example_policy_2"
            ),
            AnnotationExample(
                passage="Marketing data is retained until you withdraw your consent.",
                requirement="Data Retention Period",
                annotation_value="until you withdraw your consent",
                performed=True,
                confidence=0.89,
                source_policy="example_policy_4"
            )
        ])
        
        # Right to Erasure examples
        examples.extend([
            AnnotationExample(
                passage="You have the right to request deletion of your personal data at any time.",
                requirement="Right to Erasure",
                annotation_value="right to request deletion",
                performed=True,
                confidence=0.95,
                source_policy="example_policy_1"
            ),
            AnnotationExample(
                passage="We will delete your data upon request, subject to legal retention requirements.",
                requirement="Right to Erasure",
                annotation_value="delete your data upon request",
                performed=True,
                confidence=0.92,
                source_policy="example_policy_2"
            ),
            AnnotationExample(
                passage="You can request erasure by contacting our support team or using the account settings.",
                requirement="Right to Erasure",
                annotation_value="request erasure by contacting our support team",
                performed=True,
                confidence=0.88,
                source_policy="example_policy_3"
            )
        ])
        
        return examples
    
    def retrieve_examples(
        self, 
        requirement: str, 
        passage_text: str = None, 
        max_examples: int = 3,
        min_confidence: float = 0.8
    ) -> List[AnnotationExample]:
        """
        Retrieve examples for a specific requirement.
        
        Args:
            requirement: The GDPR requirement to find examples for
            passage_text: Optional passage text for similarity matching (not used in dummy)
            max_examples: Maximum number of examples to return
            min_confidence: Minimum confidence threshold for examples
            
        Returns:
            List of AnnotationExample objects
        """
        # Filter examples by requirement and confidence
        filtered_examples = [
            ex for ex in self.examples 
            if ex.requirement == requirement and ex.confidence >= min_confidence
        ]
        
        # In a real implementation, you would use semantic similarity here
        # For now, we'll just return random examples up to max_examples
        if len(filtered_examples) > max_examples:
            # Sort by confidence and take the top examples
            filtered_examples.sort(key=lambda x: x.confidence, reverse=True)
            filtered_examples = filtered_examples[:max_examples]
        
        return filtered_examples
    
    def retrieve_examples_for_requirements(
        self, 
        requirements: List[str], 
        passage_text: str = None,
        max_examples_per_requirement: int = 2
    ) -> Dict[str, List[AnnotationExample]]:
        """
        Retrieve examples for multiple requirements.
        
        Args:
            requirements: List of GDPR requirements
            passage_text: Optional passage text for similarity matching
            max_examples_per_requirement: Maximum examples per requirement
            
        Returns:
            Dictionary mapping requirements to lists of examples
        """
        result = {}
        
        for requirement in requirements:
            examples = self.retrieve_examples(
                requirement=requirement,
                passage_text=passage_text,
                max_examples=max_examples_per_requirement
            )
            result[requirement] = examples
        
        return result
    
    def add_example(self, example: AnnotationExample):
        """Add a new example to the database."""
        self.examples.append(example)
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        stats = {}
        for example in self.examples:
            req = example.requirement
            if req not in stats:
                stats[req] = {"count": 0, "avg_confidence": 0.0}
            stats[req]["count"] += 1
            stats[req]["avg_confidence"] += example.confidence
        
        # Calculate averages
        for req in stats:
            stats[req]["avg_confidence"] /= stats[req]["count"]
        
        return stats 