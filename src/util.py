import json
import os
import shutil
import sys
from typing import List, Dict, Optional
from difflib import SequenceMatcher
import re
import numpy as np

from sentence_transformers import SentenceTransformer

import chardet

def _detect_encoding(file_path: str) -> str:
    """
    Detects the encoding of a file.
    Args:
        file_path (str): The path to the file.
    Returns:
        str: The detected encoding of the file.
    """
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def write_to_file(file_name: str, text: str):
    """
    Writes text to a file, creating the file if it does not exist.
    Args:
        file_name (str): The name of the file to write to.
        text (str): The text to write to the file.
    """
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(text)


def append_to_file(file_name: str, text: str):
    """
    Appends text to a file, creating the file if it does not exist.
    Args:
        file_name (str): The name of the file to append to.
        text (str): The text to append to the file.
    """
    if os.path.exists(file_name):
        encoding = _detect_encoding(file_name)
    else:
        encoding = 'utf-8'

    with open(file_name, 'a', encoding=encoding) as file:
        file.write(text + '\n')


def read_from_file(file_path: str):
    """
    Reads a file and returns its content as a string, detecting the encoding automatically.
    Args:
        file_path (str): The path to the file to read.
    Returns:
        str | None: The content of the file as a string or None if the file does not exist.
    """
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist.")
        return None

    # Read the file with the detected encoding
    content = ''
    with open(file_path, 'r', encoding=_detect_encoding(file_path)) as file:
        for line in file:
            content += line
        return content

def read_json_file(file_path: str):
    """
    Reads a JSON file and returns its content.
    Args:
        file_path (str): The path to the JSON file.
    Returns:
        dict | None: The content of the JSON file as a dictionary or None if the file does not exist.
    """
    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r', encoding=_detect_encoding(file_path)) as file:
        return json.load(file)


def prepare_output(run_id: str, output_folder: str = '../../output', overwrite: bool = False):
    if overwrite and os.path.exists('output'):
        shutil.rmtree('output')

    path_list = [
        f'{output_folder}',
        f'{output_folder}/{run_id}',
        f'{output_folder}/{run_id}/log',
        f'{output_folder}/{run_id}/policies',
        f'{output_folder}/{run_id}/policies/pkgs',
        f'{output_folder}/{run_id}/policies/metadata',
        f'{output_folder}/{run_id}/policies/html',
        f'{output_folder}/{run_id}/policies/cleaned',
        f'{output_folder}/{run_id}/policies/detected',
        f'{output_folder}/{run_id}/policies/classified',
        f'{output_folder}/{run_id}/policies/detected/rejected',
        f'{output_folder}/{run_id}/policies/detected/unknown',
        f'{output_folder}/{run_id}/policies/json',
        f'{output_folder}/{run_id}/policies/annotated-standalone',
        f'{output_folder}/{run_id}/policies/annotated-twostep',
        f'{output_folder}/{run_id}/policies/reviewed-standalone',
        f'{output_folder}/{run_id}/policies/reviewed-twostep',
        f'{output_folder}/{run_id}/batch',
        f'{output_folder}/{run_id}/batch/detect',
        f'{output_folder}/{run_id}/batch/annotate',
        f'{output_folder}/{run_id}/batch/review'
    ]

    for path in path_list:
        if not os.path.exists(path):
            os.mkdir(path)


def load_policy_json(file_path: str):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"The file {file_path} does not contain valid JSON.")
        return None


def load_policy_jsonl(file_path: str):
    """
    Reads a JSONL file and returns its content as a list of dictionaries.
    Args:
        file_path (str): The path to the JSONL file.
    Returns:
        list | None: The content of the JSONL file as a list of dictionaries or None if the file does not exist.
    """
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist.")
        return None

    try:
        items = []
        with open(file_path, 'r', encoding=_detect_encoding(file_path)) as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        item = json.loads(line)
                        items.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON on line {line_num} in {file_path}: {e}")
                        continue
        return items
    except Exception as e:
        print(f"Error reading JSONL file {file_path}: {e}")
        return None


def write_jsonl_file(file_path: str, items: list):
    """
    Writes a list of items to a JSONL file, with each item on a separate line.
    Args:
        file_path (str): The path to the JSONL file to write.
        items (list): The list of items to write to the file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in items:
            file.write(json.dumps(item) + '\n')


def log_prompt_result(
        run_id: str,
        task: str,
        pkg: str,
        model_name: str,
        output_format: str,
        cost: float,
        processing_time: float,
        outputs: list
):
    # create the output folder if it does not exist
    folder_path = f"../output/{run_id}/{model_name}_responses"
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(f"{folder_path}/{task}", exist_ok=True)

    # log the cost, processing time and response
    with open(f"{folder_path}/costs_{task}.csv", "a") as f:
        f.write(f"{pkg},{cost}\n")
    with open(f"{folder_path}/times_{task}.csv", "a") as f:
        f.write(f"{pkg},{processing_time}\n")

    if len(outputs) > 1:
        for i, output in enumerate(outputs):
            with open(f"{folder_path}/{task}/{pkg}_{i}.{output_format}", "a") as f:
                f.write(output + '\n')
    else:
        with open(f"{folder_path}/{task}/{pkg}.{output_format}", "a") as f:
            f.write(outputs[0] + '\n')


def prepare_prompt_messages(
        api: str,
        task: str,
        user_msg: str = None,
        system_msg: str = None,
        examples: list[tuple[str, str]] = None,
        response_schema: dict = None,
        bundle_system_msg: bool = True,
        use_opp_115: bool = False
) -> (list[dict], str, dict, int, int, int):
    """
    Prepare the prompt messages for the API call.
    Args:
        api (str): The API to use (e.g., 'anthropic', 'openai').
        task (str): The task for which the prompt is prepared.
        user_msg (str): The user message to include in the prompt.
        system_msg (str): The system message to include in the prompt.
        examples (list[tuple[str, str]]): A list of example pairs (user message, assistant message).
        response_schema (dict): The response schema to use for the prompt.
        bundle_system_msg (bool): Whether to bundle the system message with the examples.
    Returns:
        tuple: A tuple containing:
            - messages (list[dict]): The prepared messages for the API call.
            - system_msg (str): The system message used in the prompt.
            - response_schema (dict): The response schema used in the prompt.
            - system_len (int): The length of the system message.
            - user_len (int): The length of the user message.
            - example_len (int): The total length of all example messages.
    """
    if not system_msg or not examples or not response_schema:
        sys_msg, exs, res_schema = _load_prompts_from_files(task, api, use_opp_115)

        if not system_msg:
            if sys_msg is None:
                raise ValueError(f"No system prompt specified for task '{task}' and API '{api}'.")
            else:
                system_msg = sys_msg
        if not examples:
            examples = exs
        if not response_schema:
            response_schema = res_schema

    # map the examples to the correct json format
    es = [(
        {"role": "user", "content": e[0]}, # (e[0].replace('\n', ''))
        {"role": "assistant", "content": e[1]} # (e[1].replace('\n', ''))
    ) for e in examples]

    examples = es

    # generate the messages list for the API call
    messages = []
    if bundle_system_msg:
        messages.append({"role": "system", "content": system_msg})
    for example in examples:
        messages.extend(example)
    if user_msg is not None:
        messages.append({"role": "user", "content": user_msg})

    return messages, system_msg, response_schema


def _load_prompts_from_files(task: str, api: str, use_opp_115: bool = False) -> (str, list[tuple[str, str]], dict):
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # Choose prompts directory based on annotation scheme
        prompts_base = "prompts_opp" if use_opp_115 else "prompts"
        prompts_folder = f"{dir_path}/../{prompts_base}/{api}/{task}"

        # get all file names in the prompts folder
        file_names = os.listdir(prompts_folder)

        system_prompt = None
        examples = {}
        response_schema = None
        for file_name in file_names:
            if file_name.startswith("response_schema."):
                #print(f"Loading response schema from {file_name}")
                response_schema = read_from_file(f"{prompts_folder}/{file_name}")
                # minify the json schema
                response_schema = json.loads(response_schema)
            elif file_name.startswith("system."):
                #print(f"Loading system prompt from {file_name}")
                system_prompt = read_from_file(f"{prompts_folder}/{file_name}")
                if file_name.endswith("json"):
                    system_prompt = json.dumps(json.loads(system_prompt))
            elif file_name.startswith("example_"):
                name_split = file_name.split("_")
                index = name_split[1]
                if examples.get(index) is None:
                    examples[index] = (None, None)
                file_type = name_split[-1].split(".")
                file_type = file_type[0]
                if file_type == "user":
                    #print(f"Loading user example from {file_name}")
                    msg = read_from_file(f"{prompts_folder}/{file_name}")
                    if file_name.endswith("json"):
                        msg = json.dumps(json.loads(msg))
                    examples[str(index)] = (msg, examples[str(index)][1])
                elif file_type == "assistant":
                    #print(f"Loading assistant example from {file_name}")
                    msg = read_from_file(f"{prompts_folder}/{file_name}")
                    if file_name.endswith("json"):
                        msg = json.dumps(json.loads(msg))
                    examples[str(index)] = (examples[str(index)][0], msg)
                else:
                    raise ValueError(f"Unknown file type in example: {file_type}. Expected 'user' or 'assistant'.")

        # parse examples to a list of tuples
        examples = [(user, assistant) for user, assistant in examples.values()]

        return system_prompt, examples, response_schema

    except Exception as e:
        print(f"Error loading prompts: {e}", file=sys.stderr)
        return None, [], None

# Valid GDPR transparency requirements
valid_labels = [
    'Controller Name', 'Controller Contact', 'DPO Contact',
    'Data Categories', 'Processing Purpose', 'Legal Basis for Processing',
    'Legitimate Interests for Processing', 'Source of Data',
    'Data Retention Period', 'Data Recipients', 'Third-country Transfers',
    'Mandatory Data Disclosure', 'Automated Decision-Making',
    'Right to Access', 'Right to Rectification', 'Right to Erasure',
    'Right to Restrict', 'Right to Object', 'Right to Portability',
    'Right to Withdraw Consent', 'Right to Lodge Complaint'
]

# Valid OPP-115 privacy practice categories
valid_opp_categories = [
    'First Party Collection/Use',
    'Third Party Sharing/Collection', 
    'User Choice/Control',
    'User Access, Edit and Deletion',
    'Data Retention',
    'Data Security',
    'Policy Change',
    'Do Not Track',
    'International and Specific Audiences'
]

# Global SBERT model and cached embeddings for semantic similarity
_sbert_model = None
_valid_label_embeddings = None
_embeddings_calculated = False
_valid_opp_embeddings = None
_opp_embeddings_calculated = False

def _find_closest_valid_label(label: str, use_opp_115: bool = False) -> str:
    """
    Find the closest valid label for a given label using semantic similarity.
    
    Args:
        label: The label to find a match for
        use_opp_115: Whether to use OPP-115 categories instead of GDPR labels
    
    Returns:
        The most similar valid label
    """
    # Handle None or invalid input
    if label is None:
        return "Unknown"
    if not isinstance(label, str):
        return "Unknown"
    
    # Choose the appropriate valid labels based on scheme
    if use_opp_115:
        valid_categories = valid_opp_categories
        global _valid_opp_embeddings, _opp_embeddings_calculated
        embeddings_var = _valid_opp_embeddings
        calculated_var = _opp_embeddings_calculated
    else:
        valid_categories = valid_labels
        global _valid_label_embeddings, _embeddings_calculated
        embeddings_var = _valid_label_embeddings
        calculated_var = _embeddings_calculated
    
    if label in valid_categories:
        return label
    
    global _sbert_model
    
    # Initialize SBERT model if not already done
    if _sbert_model is None:
        _sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Loaded SBERT model for semantic label matching")
    
    # Calculate embeddings only once
    if not calculated_var:
        embeddings_var = _sbert_model.encode(valid_categories, convert_to_numpy=True)
        if use_opp_115:
            _valid_opp_embeddings = embeddings_var
            _opp_embeddings_calculated = True
            print(f"✅ Cached embeddings for {len(valid_categories)} OPP categories")
        else:
            _valid_label_embeddings = embeddings_var
            _embeddings_calculated = True
            print(f"✅ Cached embeddings for {len(valid_categories)} valid labels")
    
    # Use semantic similarity with SBERT embeddings
    # Encode the input label
    label_embedding = _sbert_model.encode([label], convert_to_numpy=True)
    
    # Calculate cosine similarities
    similarities = np.dot(embeddings_var, label_embedding.T).flatten()
    
    # Find the best match
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    best_match = valid_categories[best_idx]
    
    # Only use semantic match if similarity is above threshold
    if best_score > 0.3:  # Adjustable threshold
        return best_match
    else:
        print(f"⚠️  Semantic similarity too low ({best_score:.3f}) for '{label}', using string similarity")
        # Fallback to string similarity
        best_match = None
        best_score = 0.0
        
        for valid_category in valid_categories:
            # Simple similarity using sequence matcher
            score = SequenceMatcher(None, label.lower(), valid_category.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = valid_category
        
        return best_match if best_match else label

def correct_labels(labels: List[str], use_opp_115: bool = False) -> List[str]:
    """
    Correct a list of labels by replacing invalid ones with the closest valid label.
    
    Args:
        labels: List of labels to correct
        use_opp_115: Whether to use OPP-115 categories instead of GDPR labels
    
    Returns:
        List of corrected labels
    """
    # Handle None or invalid input
    if labels is None:
        return []
    
    # Choose the appropriate valid labels based on scheme
    valid_categories = valid_opp_categories if use_opp_115 else valid_labels
    
    corrected = []
    for label in labels:
        # Handle None or non-string labels
        if label is None:
            continue
        if not isinstance(label, str):
            continue
            
        if label in valid_categories:
            corrected.append(label)
        else:
            closest = _find_closest_valid_label(label, use_opp_115)
            corrected.append(closest)
            if closest != label:
                print(f"🔄 Corrected '{label}' → '{closest}'")
    return corrected

def get_similarity_scores(label: str, use_opp_115: bool = False) -> Dict[str, float]:
    """
    Get similarity scores between a label and all valid labels for debugging.
    
    Args:
        label: The label to compare
        use_opp_115: Whether to use OPP-115 categories instead of GDPR labels
    
    Returns:
        Dictionary mapping valid labels to their similarity scores
    """
    # Handle None or invalid input
    if label is None:
        return {}
    if not isinstance(label, str):
        return {}
    
    # Choose the appropriate valid labels based on scheme
    if use_opp_115:
        valid_categories = valid_opp_categories
        global _valid_opp_embeddings, _opp_embeddings_calculated
        embeddings_var = _valid_opp_embeddings
        calculated_var = _opp_embeddings_calculated
    else:
        valid_categories = valid_labels
        global _valid_label_embeddings, _embeddings_calculated
        embeddings_var = _valid_label_embeddings
        calculated_var = _embeddings_calculated
    
    if label in valid_categories:
        return {label: 1.0}
    
    global _sbert_model
    
    # Initialize SBERT model if not already done
    if _sbert_model is None:
        _sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Calculate embeddings only once
    if not calculated_var:
        embeddings_var = _sbert_model.encode(valid_categories, convert_to_numpy=True)
        if use_opp_115:
            _valid_opp_embeddings = embeddings_var
            _opp_embeddings_calculated = True
        else:
            _valid_label_embeddings = embeddings_var
            _embeddings_calculated = True
    
    scores = {}
    
    # Use semantic similarity
    label_embedding = _sbert_model.encode([label], convert_to_numpy=True)
    similarities = np.dot(embeddings_var, label_embedding.T).flatten()
    
    for i, valid_category in enumerate(valid_categories):
        scores[valid_category] = float(similarities[i])
    
    return scores

def correct_annotations(annotations: List[Dict], use_opp_115: bool = False) -> List[Dict]:
    """
    Correct annotations by fixing invalid requirement labels.
    
    Args:
        annotations: List of annotation dictionaries with 'requirement' field
        use_opp_115: Whether to use OPP-115 categories instead of GDPR labels
    
    Returns:
        List of corrected annotations
    """
    # Choose the appropriate valid labels based on scheme
    valid_categories = valid_opp_categories if use_opp_115 else valid_labels
    
    corrected = []
    for annotation in annotations:
        if isinstance(annotation, dict) and 'requirement' in annotation:
            requirement = annotation['requirement']
            if requirement in valid_categories:
                corrected.append(annotation)
            else:
                closest = _find_closest_valid_label(requirement, use_opp_115)
                corrected_annotation = annotation.copy()
                corrected_annotation['requirement'] = closest
                corrected.append(corrected_annotation)
                if closest != requirement:
                    print(f"🔄 Corrected annotation '{requirement}' → '{closest}'")
        else:
            corrected.append(annotation)
    return corrected


def parse_llm_json_response(response: str, expected_key: str = None) -> List[Dict]:
    """
    Robustly parse LLM JSON response with multiple fallback strategies.
    
    Args:
        response: Raw LLM response string
        expected_key: Expected key in the JSON response (e.g., 'annotations', 'labels')
        
    Returns:
        List of parsed items or empty list if parsing fails
    """
    if not response or not isinstance(response, str):
        return []
    
    # Clean the response
    cleaned_response = response.strip()
    
    # Remove markdown formatting
    if cleaned_response.startswith('```json'):
        cleaned_response = cleaned_response[7:]
    elif cleaned_response.startswith('```'):
        cleaned_response = cleaned_response[3:]
    
    if cleaned_response.endswith('```'):
        cleaned_response = cleaned_response[:-3]
    
    cleaned_response = cleaned_response.strip()
    
    if not cleaned_response:
        return []
    
    # Try multiple parsing strategies
    parsing_strategies = [
        # Strategy 1: Direct JSON parsing
        lambda: json.loads(cleaned_response),
        
        # Strategy 2: Try to find JSON object/array in the response
        lambda: _extract_json_from_text(cleaned_response),
        
        # Strategy 3: Try to fix common JSON issues
        lambda: _fix_and_parse_json(cleaned_response),
    ]
    
    for i, strategy in enumerate(parsing_strategies):
        try:
            result = strategy()
            if result:
                # If we have an expected key, try to extract it
                if expected_key and isinstance(result, dict):
                    if expected_key in result and isinstance(result[expected_key], list):
                        return result[expected_key]
                    else:
                        continue
                # If no expected key or result is already a list
                elif isinstance(result, list):
                    return result
                elif isinstance(result, dict):
                    # Try common keys
                    for key in ['annotations', 'labels', 'requirements', 'categories', 'predictions']:
                        if key in result and isinstance(result[key], list):
                            return result[key]
                    # If no valid key found, return empty list
                    return []
                else:
                    continue
        except Exception as e:
            print(f"Strategy {i+1} failed: {str(e)}")
            continue
    
    print(f"All parsing strategies failed for response: {cleaned_response[:100]}...")
    return []


def _extract_json_from_text(text: str):
    """Extract JSON object/array from text that might contain other content."""
    import re
    
    # Look for JSON object pattern
    object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(object_pattern, text)
    
    for match in matches:
        try:
            result = json.loads(match)
            if result:
                return result
        except:
            continue
    
    # Look for JSON array pattern
    array_pattern = r'\[[^\]]*\]'
    matches = re.findall(array_pattern, text)
    
    for match in matches:
        try:
            result = json.loads(match)
            if isinstance(result, list):
                return result
        except:
            continue
    
    return None


def _fix_and_parse_json(text: str):
    """Try to fix common JSON issues and parse."""
    import re
    
    # Fix common issues
    fixed_text = text
    
    # Fix unescaped quotes in strings
    fixed_text = re.sub(r'([^\\])"([^"]*?)([^\\])"', r'\1"\2\3"', fixed_text)
    
    # Fix trailing commas
    fixed_text = re.sub(r',(\s*[}\]])', r'\1', fixed_text)
    
    # Fix missing quotes around keys
    fixed_text = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed_text)
    
    try:
        result = json.loads(fixed_text)
        return result
    except:
        pass
    
    return None

