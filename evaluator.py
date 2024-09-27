import os
import re
import sys
import nltk
import json
import gensim
import logging
import argparse
import rapidfuzz
import urllib.request
import rapidfuzz.utils
import numpy as np
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from gensim.models.fasttext import FastText
from typing import List, Dict

from src import api_wrapper

# Define constants for file paths
POLICY_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVALUATION_DIR = os.path.join(POLICY_ANALYSIS_DIR, 'evaluation')
DATA_DIR = os.path.join(EVALUATION_DIR, 'embedding_data')
MODELS_DIR = os.path.join(EVALUATION_DIR, 'embedding_models')
FASTTEXT_DIR = os.path.join(MODELS_DIR, 'fasttext')


class ModelNotFoundError(Exception):
    """Exception raised when a model is not found."""
    pass


class ModelManager:
    def __init__(self, fasttext_model_name="wiki-news-300d-1M-subword"):
        self.fasttext_model_name = fasttext_model_name
        self.fasttext_model = None

        # Define model paths
        self.fasttext_pretrained_path = os.path.join(FASTTEXT_DIR, self.fasttext_model_name, f"{self.fasttext_model_name}.bin")
        self.fasttext_custom_path = os.path.join(FASTTEXT_DIR, self.fasttext_model_name, f"{self.fasttext_model_name}-custom.model")
        self.fasttext_finetuned_path = os.path.join(FASTTEXT_DIR, self.fasttext_model_name, f"{self.fasttext_model_name}-finetuned.model")

    def download_pretrained_fasttext_model(self):
        print("Downloading FastText model...")
        os.makedirs(os.path.join(FASTTEXT_DIR, self.fasttext_model_name), exist_ok=True)
        url = f'https://dl.fbaipublicfiles.com/fasttext/vectors-english/{self.fasttext_model_name}.bin.zip'
        zip_path = os.path.join(FASTTEXT_DIR, self.fasttext_model_name, f'{self.fasttext_model_name}.bin.zip')
        bin_path = os.path.join(FASTTEXT_DIR, self.fasttext_model_name, f'{self.fasttext_model_name}.bin')

        """ alternative pretrained model on a larger common crawl corpus"""
        # url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip'
        # zip_path = os.path.join(FASTTEXT_DIR, self.fasttext_model_name, 'crawl-300d-2M-subword.bin.zip')
        # bin_path = os.path.join(FASTTEXT_DIR, self.fasttext_model_name, 'crawl-300d-2M-subword.bin')

        # Download the zip file if it does not exist
        if not os.path.exists(zip_path):
            print("Downloading pre-trained FastText binary model...")
            urllib.request.urlretrieve(url, zip_path)
            print("Download complete.")

        # Extract the bin file if it does not exist
        if not os.path.exists(bin_path):
            print("Extracting FastText binary model...")
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(FASTTEXT_DIR, self.fasttext_model_name))
            print("Extraction complete.")

        self.fasttext_pretrained_path = bin_path
        print(f"Pre-trained FastText model available at {self.fasttext_pretrained_path}.")

    def load_pretrained_fasttext_model(self):
        if self.fasttext_model is not None:
            print("FastText model already loaded.")
            return self.fasttext_model

        if not os.path.exists(self.fasttext_pretrained_path):
            print("Pre-trained FastText model not found. Downloading...")
            self.download_pretrained_fasttext_model()
        else:
            print("Loading pre-trained FastText model...")

        self.fasttext_model = gensim.models.fasttext.load_facebook_model(self.fasttext_pretrained_path)
        print("Pre-trained FastText model loaded successfully.")

        return self.fasttext_model

    def load_finetuned_fasttext_model(self):
        if self.fasttext_model is not None:
            print("FastText model already loaded.")
            return self.fasttext_model

        if os.path.exists(self.fasttext_finetuned_path):
            print("Loading fine-tuned FastText model...")
            self.fasttext_model = FastText.load(self.fasttext_finetuned_path)
            print("Fine-tuned FastText model loaded successfully.")
            return self.fasttext_model
        else:
            print(f"Warning: Fine-tuned FastText model not found. Path: {self.fasttext_finetuned_path}")
            sys.exit(1)

    def load_custom_fasttext_model(self):
        if self.fasttext_model is not None:
            print("FastText model already loaded.")
            return self.fasttext_model

        if os.path.exists(self.fasttext_custom_path):
            print("Loading custom FastText model...")
            self.fasttext_model = FastText.load(self.fasttext_custom_path)
            print("Custom FastText model loaded successfully.")
            return self.fasttext_model
        else:
            print("Warning: Custom FastText model not found.")
            sys.exit(1)

    def fine_tune_fasttext_model(self, dataset_file: str):
        # Ensure directories exist
        os.makedirs(os.path.join(FASTTEXT_DIR, self.fasttext_model_name), exist_ok=True)

        # Load and tokenize sentences
        sentences = load_sentences(dataset_file)
        tokenized_data = [sentence.strip().split() for sentence in sentences if sentence.strip()]
        print(f"Fine-tuning on: {tokenized_data[:5]}")

        # Load or download pre-trained model
        self.load_pretrained_fasttext_model()

        # Set training parameters
        self.fasttext_model.sg = 1
        self.fasttext_model.min_count = 1
        self.fasttext_model.workers = 4
        self.fasttext_model.window = 10

        # Build the vocabulary with your dataset
        print("Building vocabulary with custom dataset...")
        self.fasttext_model.build_vocab(corpus_iterable=tokenized_data, update=True)
        print(f"Vocabulary updated. Number of unique tokens: {len(self.fasttext_model.wv)}")

        # Fine-tune the model
        print("Fine-tuning FastText model...")
        self.fasttext_model.train(
            corpus_iterable=tokenized_data,
            total_examples=len(tokenized_data),
            epochs=15,
            compute_loss=True
        )
        print("FastText model fine-tuned successfully.")

        # Save the fine-tuned model
        self.fasttext_model.save(self.fasttext_finetuned_path)
        print(f"FastText model saved successfully at {self.fasttext_finetuned_path}.")

    def train_fasttext_model(self, dataset, vector_size=600, window=10, min_count=1, epochs=10, sg=1):
        """
        Train a new FastText model from scratch on the given dataset.
        
        :param dataset: List of lists of strings, where each inner list represents a sentence
        :param vector_size: Dimensionality of the word vectors (default: 600)
        :param window: Maximum distance between the current and predicted word within a sentence (default: 10)
        :param min_count: Ignores all words with total frequency lower than this (default: 1)
        :param epochs: Number of iterations over the corpus (default: 10)
        :param sg: Training algorithm: 1 for skip-gram; otherwise CBOW (default: 1)
        """
        print("Training new FastText model...")

        # Initialize and train the FastText model
        self.fasttext_model = FastText(
            sentences=dataset,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=os.cpu_count(),  # Use all available CPU cores
            epochs=epochs,
            sg=sg
        )

        print("FastText model training completed.")

        # Save the trained model
        os.makedirs(os.path.dirname(self.fasttext_custom_path), exist_ok=True)
        self.fasttext_model.save(self.fasttext_custom_path)
        print(f"FastText model saved successfully at {self.fasttext_custom_path}.")

    def get_fasttext_phrase_embedding(self, phrase: str) -> np.ndarray:
        words = phrase.strip().split()
        if not words:
            logging.warning(f"Empty phrase: '{phrase}'. Continuing with an empty vector.")
            return np.zeros(self.fasttext_model.vector_size)

        vectors = [self.fasttext_model.wv[word] for word in words]

        if not vectors:
            raise ValueError(f"None of the words in the phrase: '{phrase}' were found in the vocabulary.")

        return np.mean(vectors, axis=0)


def tokenizer_downloaded():
    try:
        nltk.data.find('tokenizers/punkt')
        return True
    except LookupError:
        return False


def download_tokenizer():
    print("Downloading NLTK Punkt tokenizer...")
    nltk.download('punkt')
    print("Download complete.")


def create_custom_sentence_tokenizer():
    """Create a custom sentence tokenizer that doesn't split sentences at common legal abbreviations."""
    punkt_param = PunktParameters()
    legal_abbreviations = [
        "art.", "lit.", "cal.", "cit.", "civ.", "sec.", "cl.", "v.", "p.", "pp.", "no.", "n.", "ann.", "ord.", "par.",
        "para.", "fig.", "ex.", "exh.", "doc.", "vol.", "ed.", "ch.", "pt.", "inc.", "corp.", "plc.", "gov.", "min.",
        "dept.", "div.", "comm.", "ag.", "adj.", "adv.", "aff.", "agr.", "app.", "att.", "bldg.", "bk.", "bul.", "cir.",
        "co.", "com.", "conc.", "conf.", "const.", "def.", "dept.", "det.", "dev.", "dir.", "dist.", "ed.", "est.",
        "exp.", "ext.", "fig.", "gen.", "gov.", "hist.", "il.", "inc.", "ind.", "info.", "int.", "jr.", "jud.", "leg.",
        "llc.", "lt.", "maj.", "max.", "med.", "mem.", "misc.", "mtg.", "nat.", "nr.", "org.", "ph.", "pl.", "pub.",
        "reg.", "rep.", "rev.", "sci.", "sec.", "ser.", "st.", "sub.", "supp.", "techn.", "temp.", "treas.", "univ.",
        "vol.", "vs.", "yr.", "zoning.", "appx.", "supp.", "admr.", "dba.", "et al.", "et seq.", "etc.", "i.e.", "e.g.",
        "ca.", "id.", "infra.", "supra.", "viz.", "vs.", "am.", "pm.", "corp.", "ltd.", "inc.", "co.", "re.", "u.s.",
        "u.k.", "ca.", "fla.", "ny.", "tex.", "jr.", "sr.", "rev.", "dr.", "mr.", "mrs.", "prof.", "pres.", "gov.",
        "sen.", "rep.", "gen.", "col.", "maj.", "lt.", "sgt.", "cpt.", "det.", "st.", "m.d.", "ph.d.", "esq.", "l.c.",
        "l.l.p.", "l.p.", "s.p.", "2d", "3d", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "i.", "ii.", "iii.",
        "iv.", "v.", "vi.", "vii.", "viii.", "ix.", "x.", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.",
        "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "oct.", "nov.", "dec."  # Months
    ]
    punkt_param.abbrev_types.update(legal_abbreviations)
    return PunktSentenceTokenizer(punkt_param)


def create_dataset(tokenizer: PunktSentenceTokenizer) -> List[str]:
    """Create a dataset from the files given in the dataset folder."""
    dataset_file = os.path.join(DATA_DIR, 'dataset.txt')
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Created '{DATA_DIR}' folder. Please fill it with the required data.")
        sys.exit(0)

    if os.path.exists(dataset_file):
        overwrite = input("The 'dataset.txt' file already exists. Do you want to overwrite it? (yes/n) ")
        if overwrite.lower() != 'yes':
            return []
        os.remove(dataset_file)

    def extract_lines(file_path: str) -> list:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        html_tag_pattern = re.compile(r'<[^>]+>')
        return [line.strip() for line in lines if line.strip() and not html_tag_pattern.match(line.strip())]

    total_sentences = 0
    with open(dataset_file, 'w', encoding='utf-8') as dataset:
        for file in os.listdir(DATA_DIR):
            if file.endswith(('.txt', '.html')) and file != 'dataset.txt':
                lines = extract_lines(os.path.join(DATA_DIR, file))
                for line in lines:
                    for sentence in tokenizer.tokenize(line):
                        if sentence.strip():
                            dataset.write(sentence.strip() + '\n')
                            total_sentences += 1

    print(f"Dataset created successfully. {total_sentences} sentences written to 'dataset.txt'.")
    return []


def load_sentences(file_path):
    """Load sentences from a dataset file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def fasttext_phrase_similarity(manager: ModelManager, phrase1: str, phrase2: str) -> float:
    vec1 = manager.get_fasttext_phrase_embedding(phrase1)
    vec2 = manager.get_fasttext_phrase_embedding(phrase2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    # Handle division by zero, usually occurs when one of the input phrases is empty
    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    return similarity


def compare_annotations_fasttext(manager: ModelManager, file1: str, file2: str):
    """Compare annotations between two JSON files."""
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        annotations1 = json.load(f1)
        annotations2 = json.load(f2)

    for i, (ann1, ann2) in enumerate(zip(annotations1, annotations2)):
        if ann1['passage'] != ann2['passage']:
            print(f"Passage mismatch at index {i}:")
            print(f"File 1: {ann1['passage']}")
            print(f"File 2: {ann2['passage']}")
            continue

        if not ann1['annotations'] and not ann2['annotations']:
            print(f"No annotations from both LLMs for passage at index {i}.")
            continue

        shorter_list = ann1['annotations'] if len(ann1['annotations']) < len(ann2['annotations']) else ann2['annotations']
        total_similarity = 0
        for ann in shorter_list:
            similarity_scores = [fasttext_phrase_similarity(manager, ann['value'], other_ann['value']) for other_ann in ann2['annotations']]
            total_similarity += max(similarity_scores)
        similarity_score = total_similarity / len(shorter_list)
        print(f"Average similarity score for passage at index {i}: {similarity_score:.4f}")


def compare_phrases_fasttext(manager: ModelManager, phrase1: str, phrase2: str):
    """Compare the similarity between two phrases."""
    similarity_score = fasttext_phrase_similarity(manager, phrase1, phrase2)
    print(f"Similarity score between '{phrase1}' and '{phrase2}': {similarity_score:.4f}")


def evaluate_annotations_for_llm_model(manager: ModelManager, llm_name: str, package_names: List[str], annotated_dir: str, reference_dir: str, evaluation_dir: str,
                                       evaluate_reviewed: bool):
    for package_name in package_names:
        annotated_file = find_json_file(annotated_dir, package_name, llm_name)
        reference_file = find_json_file(reference_dir, package_name)

        if not annotated_file or not reference_file:
            logging.warning(f"Skipping {package_name}: annotated or reference file not found for LLM '{llm_name}'")
            continue

        print(f"Evaluating {'reviewed ' if evaluate_reviewed else ''}annotations for {package_name} by LLM '{llm_name}'...")
        with open(annotated_file, 'r') as f_annotated, open(reference_file, 'r') as f_reference:
            annotated_data = json.load(f_annotated)
            reference_data = json.load(f_reference)

        evaluated_data = evaluate_file(manager, llm_name, reference_data, annotated_data, evaluate_reviewed)

        output_file = os.path.join(evaluation_dir, f"{'reviewed_' if evaluate_reviewed else ''}{llm_name}.{package_name}_evaluated.json")
        with open(output_file, 'w') as f_out:
            json.dump(evaluated_data, f_out, indent=4)

        logging.info(f"Evaluation completed for {package_name}")


def evaluate_annotations(manager: ModelManager, run_id: str, evaluate_reviewed: bool):
    output_dir = os.path.join(POLICY_ANALYSIS_DIR, 'output', run_id)
    annotated_dir = os.path.join(output_dir, 'annotated')
    reviewed_dir = os.path.join(output_dir, 'reviewed')
    reference_dir = os.path.join(EVALUATION_DIR, 'reference_annotations')
    evaluated_dir = os.path.join(output_dir, 'evaluated')
    os.makedirs(evaluated_dir, exist_ok=True)

    id_file_path = os.path.join(output_dir, 'id_file.txt')
    with open(id_file_path, 'r') as id_file:
        package_names = [line.strip() for line in id_file]

    # if a model_file.txt file exists in the output folder, assume that the evaluation against the reference data is to be done for each LLM model in the file
    model_file_path = os.path.join(output_dir, 'models.txt')
    if os.path.exists(model_file_path):
        with open(model_file_path, 'r') as model_file:
            llm_names = [line.strip() for line in model_file]
        for llm_name in llm_names:
            if llm_name in api_wrapper.OLLAMA_MODELS.keys():
                llm_name = api_wrapper.OLLAMA_MODELS[llm_name].replace(':', '_')
            elif llm_name in api_wrapper.OPENAI_MODELS.keys():
                pass
            else:
                logging.warning(f"Model '{llm_name}' not found in the list of available models.")
                print(f"Model '{llm_name}' not found in the list of available models. Skipping...")
                continue
            evaluate_annotations_for_llm_model(manager, llm_name, package_names, annotated_dir if not evaluate_reviewed else reviewed_dir, reference_dir, evaluated_dir,
                                               evaluate_reviewed)
    else:
        # if no model_file.txt file exists, assume that the evaluation is to be done for a single model
        llm_names = [ollama_name.replace(':', '_') for ollama_name in api_wrapper.OLLAMA_MODELS.values()] + list(api_wrapper.OPENAI_MODELS.keys())
        first_file = next((file for file in os.listdir(annotated_dir) if file.endswith('.json')), None)
        if not first_file:
            logging.error(f"No annotated files found in the directory '{annotated_dir}'.")
            return
        llm_name = next((llm for llm in llm_names if llm in first_file), None)
        if not llm_name:
            logging.error(f"No valid model name found in the file name '{first_file}'.")
            print(f"No valid model name found in the file name '{first_file}'. Exiting...")
            return
        evaluate_annotations_for_llm_model(manager, llm_name, package_names, annotated_dir if not evaluate_reviewed else reviewed_dir, reference_dir, evaluated_dir,
                                           evaluate_reviewed)


def find_json_file(directory: str, package_name: str, llm_name: str = None) -> str:
    for filename in os.listdir(directory):
        if filename.endswith('.json') and package_name in filename and (llm_name is None or llm_name in filename):
            return os.path.join(directory, filename)
    return None


def evaluate_file(manager: ModelManager, llm_name: str, reference_data: List[Dict], annotated_data: List[Dict], evaluate_reviewed: bool) -> List[Dict]:
    evaluated_data = []

    for ref_entry in reference_data:
        ref_passage = ref_entry['passage'].lower().strip()
        try:
            if evaluate_reviewed:
                matching_entry = next(
                    (entry['revised'] for entry in annotated_data if entry.get('revised') and entry['revised'].get('passage', '').lower().strip() == ref_passage), None)
            else:
                matching_entry = next((entry for entry in annotated_data if entry.get('passage', '').lower().strip() == ref_passage), None)
        except KeyError as ke:
            logging.error(f"KeyError for passage: {ref_passage[:50]} in entry: {ref_entry}: {ke}")
            matching_entry = None
        except AttributeError as ae:
            logging.error(f"AttributeError for passage: {ref_passage[:50]} in entry: {ref_entry}: {ae}")
            matching_entry = None

        if not matching_entry:
            matching_entry = find_similar_passage(manager, ref_passage, annotated_data, evaluate_reviewed)

        if matching_entry:
            try:
                if 'annotations' not in ref_entry or 'annotations' not in matching_entry or (matching_entry.get('annotations') is None and ref_entry.get('annotations') is not None):
                    logging.warning(f"No annotations found for passage: {ref_passage[:50]} for LLM '{llm_name}'")
                    matching_entry['correct_requirements_ratio'] = 0
                    matching_entry['false_positives'] = 0
                    matching_entry['false_negatives'] = 0
                    matching_entry['reason'] = "No annotations found for this passage."
                else:
                    correct_requirements_ratio, false_positives, false_negatives = calculate_requirements_stats(ref_entry['annotations'], matching_entry['annotations'])
                    matching_entry['correct_requirements_ratio'] = float(correct_requirements_ratio)
                    matching_entry['false_positives'] = false_positives
                    matching_entry['false_negatives'] = false_negatives

                    for ref_ann in ref_entry['annotations']:
                        best_match, semantic_similarity, word_level_similarity = find_best_matching_annotation(manager, ref_ann, matching_entry['annotations'])
                        if best_match:
                            if 'semantic_similarity' in best_match and 'word_level_similarity' in best_match:
                                if semantic_similarity > best_match['semantic_similarity']:
                                    best_match['semantic_similarity'] = float(semantic_similarity)
                                    best_match['word_level_similarity'] = float(word_level_similarity)
                                else:
                                    continue
                            else:
                                best_match['semantic_similarity'] = float(semantic_similarity)
                                best_match['word_level_similarity'] = float(word_level_similarity)

                    # Handle annotations made by the contender model that were not found in the reference data
                    for ann in matching_entry['annotations']:
                        if 'semantic_similarity' not in ann and 'word_level_similarity' not in ann:
                            ann['semantic_similarity'] = 0.0
                            ann['word_level_similarity'] = 0.0
                            ann['reason'] = "This annotation was not made by the reference model"
            except Exception as e:
                logging.error(f"Error processing annotations for passage: {ref_passage[:50]}: {e}", exc_info=True)
                matching_entry['correct_requirements_ratio'] = 0
                matching_entry['false_positives'] = 0
                matching_entry['false_negatives'] = 0
                matching_entry['reason'] = f"Error processing annotations: {str(e)}"
        else:
            matching_entry = ref_entry.copy()
            matching_entry['correct_requirements_ratio'] = 0
            matching_entry['false_positives'] = 0
            matching_entry['false_negatives'] = 0
            matching_entry['reason'] = "This entry was not found in the annotated data and was therefore copied from the reference data."
            logging.info(f"No matching entry found for passage: {ref_passage[:50]}...")

        evaluated_data.append(matching_entry)

    return evaluated_data


def calculate_requirements_stats(ref_annotations: List[Dict], annotated_annotations: List[Dict]) -> (float, int, int):
    if not ref_annotations and not annotated_annotations:
        return 1.0, 0, 0

    ref_requirements = {ann['requirement'] for ann in ref_annotations}
    annotated_requirements = {ann['requirement'] for ann in annotated_annotations}

    correct_requirements = ref_requirements & annotated_requirements
    false_positives = len(annotated_requirements - ref_requirements)
    false_negatives = len(ref_requirements - annotated_requirements)

    correct_requirements_percentage = len(correct_requirements) / len(ref_requirements) if ref_requirements else 0

    return correct_requirements_percentage, false_positives, false_negatives


def find_best_matching_annotation(manager: ModelManager, ref_ann: Dict, annotated_annotations: List[Dict]) -> (Dict, float, float):
    best_match = None
    best_semantic_similarity = 0
    best_word_level_similarity = 0

    for ann in annotated_annotations:
        if ann['requirement'] == ref_ann['requirement']:
            try:
                if not isinstance(ref_ann['value'], str) or not isinstance(ann['value'], str):
                    raise ValueError("Invalid 'value' type. Expected a string.")

                semantic_similarity = fasttext_phrase_similarity(manager, ref_ann['value'], ann['value'])
                word_level_similarity = rapidfuzz.fuzz.ratio(ref_ann['value'], ann['value'], processor=rapidfuzz.utils.default_process) / 100.0

                if semantic_similarity > best_semantic_similarity:
                    best_match = ann
                    best_semantic_similarity = semantic_similarity
                    best_word_level_similarity = word_level_similarity
            except ValueError as e:
                logging.error(f"Error processing annotation values: {e}", exc_info=True)

    return best_match, best_semantic_similarity, best_word_level_similarity


def find_similar_passage(manager: ModelManager, ref_passage: str, annotated_data: List[Dict], evaluate_reviewed: bool) -> Dict:
    try:
        similarity_scores = [
            (entry.get('revised', entry) if evaluate_reviewed else entry,
             fasttext_phrase_similarity(manager, ref_passage, entry.get('revised', {}).get('passage', '') if evaluate_reviewed else entry.get('passage', '')))
            for entry in annotated_data if entry.get('revised') or entry.get('passage')
        ]
        similar_entries = [entry for entry, score in similarity_scores if score > 0.9]

        if len(similar_entries) == 1:
            return similar_entries[0]
        elif len(similar_entries) > 1:
            logging.info(f"Multiple similar passages found for: {ref_passage[:50]}...")
        else:
            logging.info(f"No similar passage found for: {ref_passage[:50]}...")
    except Exception as e:
        logging.error(f"Error finding similar passage for: {ref_passage[:50]}: {e}")

    return None


def calculate_annotation_similarity(manager: ModelManager, ref_annotations: List[Dict], annotated_annotations: List[Dict]) -> (List[float], float):
    if not ref_annotations and not annotated_annotations:
        return [], 1.0
    elif not ref_annotations or not annotated_annotations:
        return [], 0.0

    similarity_scores = []

    for ref_ann in ref_annotations:
        best_match_score = max([
            calculate_single_annotation_similarity(manager, ref_ann, ann)
            for ann in annotated_annotations
        ], default=0)
        similarity_scores.append(best_match_score)

    avg_similarity_score = sum(similarity_scores) / len(similarity_scores)
    return similarity_scores, avg_similarity_score


def calculate_single_annotation_similarity(manager: ModelManager, ref_ann: Dict, ann: Dict) -> float:
    requirement_match = ref_ann['requirement'] == ann['requirement']
    if not requirement_match:
        return 0

    if 'value' not in ref_ann or 'value' not in ann:
        logging.warning(f"Value not found in annotation: {ref_ann}, {ann}")
        return 0
    if not isinstance(ref_ann['value'], str) or not isinstance(ann['value'], str):
        logging.warning(f"Value not a string: {ref_ann['value']}, {ann['value']} with types {type(ref_ann['value'])}, {type(ann['value'])}")
        return 0

    value_similarity = fasttext_phrase_similarity(manager, ref_ann['value'], ann['value'])

    return value_similarity


def main():
    argparser = argparse.ArgumentParser(description="Embedding Evaluation Tool")
    argparser.add_argument('--create-dataset', action='store_true', help="Create a dataset from the files in the 'data' folder.")
    argparser.add_argument('--compare-files', nargs=2, metavar=('file1', 'file2'), help="Compare annotations in two files.")
    argparser.add_argument('--compare-phrases', nargs=2, metavar=('phrase1', 'phrase2'), help="Compare two phrases.")
    argparser.add_argument('--fine-tune', action='store_true', help="Fine-tune the selected model on the created dataset.")
    argparser.add_argument('--train', action='store_true', help="Train a new FastText model from scratch.")
    argparser.add_argument('--download-model', action='store_true', help="Download the selected model.")
    argparser.add_argument('--model-type', type=str, choices=['pretrained', 'custom', 'finetuned'], help="Specify the type of model to use for the evaluation.",
                           default='pretrained')
    argparser.add_argument('--evaluate', action='store_true', help="Evaluate annotations against reference annotations.")
    argparser.add_argument('--run-id', type=str, help="Run ID for evaluation.")
    argparser.add_argument('--evaluate-reviewed', action='store_true', help="Evaluate reviewed annotations instead of regular annotations", default=False)
    args = argparser.parse_args()

    # Create necessary directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FASTTEXT_DIR, exist_ok=True)

    if not tokenizer_downloaded():
        download_tokenizer()

    # Handle --create-dataset separately as it doesn't require a model
    if args.create_dataset:
        tokenizer = create_custom_sentence_tokenizer()
        create_dataset(tokenizer)
        return

    if args.download_model:
        manager = ModelManager()
        manager.download_pretrained_fasttext_model()
        return

    if args.train:
        dataset_path = os.path.join(DATA_DIR, 'dataset.txt')
        if os.path.exists(dataset_path):
            print('Checkpoint')
            dataset = load_sentences(dataset_path)
            manager = ModelManager(fasttext_model_name="custom-model")
            manager.train_fasttext_model(dataset, epochs=15)
        else:
            print("Dataset not found. Please create the dataset first using --create-dataset.")
        return

    if args.model_type == 'pretrained':
        manager = ModelManager()
        manager.load_pretrained_fasttext_model()
    elif args.model_type == 'custom':
        manager = ModelManager(fasttext_model_name="custom-model")
        manager.load_custom_fasttext_model()
    elif args.model_type == 'finetuned':
        manager = ModelManager()
        manager.load_finetuned_fasttext_model()
    else:
        print("Please provide a valid model type.")
        sys.exit(1)

    if args.evaluate:
        if not args.run_id:
            print("Please provide a --run-id when using --evaluate.")
            sys.exit(1)
        print(f"Evaluating {'reviewed ' if args.evaluate_reviewed else ''}annotations for {args.run_id}...\n\n")
        evaluate_annotations(manager, args.run_id, args.evaluate_reviewed)
    elif args.compare_files:
        compare_annotations_fasttext(manager, args.compare_files[0], args.compare_files[1])
    elif args.compare_phrases:
        compare_phrases_fasttext(manager, args.compare_phrases[0], args.compare_phrases[1])
    elif args.fine_tune:
        dataset_path = os.path.join(DATA_DIR, 'dataset.txt')
        if os.path.exists(dataset_path):
            manager.fine_tune_fasttext_model(dataset_path)
        else:
            print("Dataset not found. Please create the dataset first using --create-dataset.")
    else:
        print("No valid arguments provided. Please use one of the following options:")
        print(argparser.print_help())


if __name__ == '__main__':
    main()
