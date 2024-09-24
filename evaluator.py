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
import numpy as np
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from gensim.models.fasttext import FastText
from typing import List, Dict

from src import api_wrapper

# Define constants for file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'embedding_data')
MODELS_DIR = os.path.join(BASE_DIR, 'embedding_models')
FASTTEXT_DIR = os.path.join(MODELS_DIR, 'fasttext')
SBERT_DIR = os.path.join(MODELS_DIR, 'sbert')


class ModelNotFoundError(Exception):
    """Exception raised when a model is not found."""
    pass


class ModelManager:
    def __init__(self, fasttext_model_name="wiki-news-300d-1M-subword", sbert_model_name="sentence-transformers/all-mpnet-base-v2"):
        self.fasttext_model_name = fasttext_model_name
        self.sbert_model_name = sbert_model_name
        self.fasttext_model = None
        self.sbert_model = None
        self.current_model = None

        # Define model paths
        self.fasttext_pretrained_path = os.path.join(FASTTEXT_DIR, self.fasttext_model_name, f"{self.fasttext_model_name}.bin")
        self.fasttext_finetuned_path = os.path.join(FASTTEXT_DIR, self.fasttext_model_name, f"{self.fasttext_model_name}-finetuned.model")
        self.sbert_finetuned_path = os.path.join(SBERT_DIR, "fine_tuned_sbert")

    def download_fasttext_model(self):
        print("Downloading FastText model...")
        os.makedirs(os.path.join(FASTTEXT_DIR, self.fasttext_model_name), exist_ok=True)
        url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.bin.zip'
        zip_path = os.path.join(FASTTEXT_DIR, self.fasttext_model_name, 'wiki-news-300d-1M-subword.bin.zip')
        bin_path = os.path.join(FASTTEXT_DIR, self.fasttext_model_name, 'wiki-news-300d-1M-subword.bin')

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

    def load_fasttext_model(self):
        if self.fasttext_model is not None:
            print("FastText model already loaded.")
            return self.fasttext_model

        if os.path.exists(self.fasttext_finetuned_path):
            print("Loading fine-tuned FastText model...")
            self.fasttext_model = FastText.load(self.fasttext_finetuned_path)
            print("Fine-tuned FastText model loaded successfully.")
        else:
            if not os.path.exists(self.fasttext_pretrained_path):
                print("Pre-trained FastText model not found. Downloading...")
                self.download_fasttext_model()
            else:
                print("Loading pre-trained FastText model...")
            # Load the pre-trained FastText model
            self.fasttext_model = gensim.models.fasttext.load_facebook_model(self.fasttext_pretrained_path)
            print("Pre-trained FastText model loaded successfully.")

        self.current_model = 'fasttext'
        return self.fasttext_model

    def load_sbert_model(self):
        if self.sbert_model is not None:
            print("SBERT model already loaded.")
            return self.sbert_model

        self.sbert_model = SentenceTransformer(self.sbert_model_name)
        print(f"SBERT model '{self.sbert_model_name}' loaded successfully.")
        self.current_model = 'sbert'

    def load_finetuned_sbert_model(self, model_path: str):
        if self.sbert_model is not None:
            print("Fine-tuned SBERT model already loaded.")
            return self.sbert_model

        if not os.path.exists(model_path):
            raise ModelNotFoundError(f"Model not found at {model_path}")

        print("Loading fine-tuned SBERT model...")
        self.sbert_model = SentenceTransformer(model_path)
        print(f"Fine-tuned SBERT model loaded successfully from '{model_path}'.")
        self.current_model = 'sbert'

    def fine_tune_fasttext_model(self, dataset_file: str):
        # Ensure directories exist
        os.makedirs(os.path.join(FASTTEXT_DIR, self.fasttext_model_name), exist_ok=True)

        # Load and tokenize sentences
        sentences = load_sentences(dataset_file)
        tokenized_data = [sentence.strip().split() for sentence in sentences if sentence.strip()]
        print(f"Fine-tuning on: {tokenized_data[:5]}")

        # Load or download pre-trained model
        self.load_fasttext_model()

        # Build the vocabulary with your dataset
        print("Building vocabulary with custom dataset...")
        self.fasttext_model.build_vocab(corpus_iterable=tokenized_data, update=True)
        print(f"Vocabulary updated. Number of unique tokens: {len(self.fasttext_model.wv)}")

        # Set training parameters
        self.fasttext_model.min_count = 1
        self.fasttext_model.workers = 4
        self.fasttext_model.window = 10

        # Fine-tune the model
        print("Training FastText model...")
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

        self.current_model = 'fasttext'

    def fine_tune_sbert_unsupervised(self, file_path, output_dir=SBERT_DIR):
        sentences = load_sentences(file_path)
        train_examples = create_input_examples(sentences)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.MultipleNegativesRankingLoss(self.sbert_model)
        self.sbert_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=3,
            warmup_steps=100,
            show_progress_bar=True
        )
        self.sbert_model.save(output_dir)
        print(f"Model fine-tuned and saved successfully at {output_dir}")

    def get_fasttext_phrase_embedding(self, phrase: str) -> np.ndarray:
        words = phrase.strip().split()
        if not words:
            logging.warning(f"Empty phrase: '{phrase}'. Continuing with an empty vector.")
            return np.zeros(self.fasttext_model.vector_size)

        vectors = [self.fasttext_model.wv[word] for word in words]

        if not vectors:
            raise ValueError(f"None of the words in the phrase: '{phrase}' were found in the vocabulary.")

        return np.mean(vectors, axis=0)

    def get_sbert_phrase_embedding(self, phrase: str) -> np.ndarray:
        if self.current_model != 'sbert':
            self.load_sbert_model()
        return self.sbert_model.encode(phrase, convert_to_tensor=True).cpu().numpy()


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


def create_input_examples(sentences):
    """Create InputExamples for unsupervised fine-tuning by duplicating sentences."""
    examples = [InputExample(texts=[sentence, sentence]) for sentence in sentences]
    return examples


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


def sbert_phrase_similarity(manager: ModelManager, phrase1: str, phrase2: str) -> float:
    vec1 = manager.get_sbert_phrase_embedding(phrase1)
    vec2 = manager.get_sbert_phrase_embedding(phrase2)
    return util.cos_sim(vec1, vec2).item()


def sbert_annotation_similarity(manager: ModelManager, annotation1: dict, annotation2: dict) -> float:
    """Compare two annotations and return a similarity score."""
    if 'requirement' not in annotation1 or 'requirement' not in annotation2:
        logging.error(f"One of the annotations does not contain a 'requirement' key: {annotation1}, {annotation2}")
        return 0
    if annotation1['requirement'].lower() != annotation2['requirement'].lower():
        return 0

    return sbert_phrase_similarity(manager, annotation1['value'], annotation2['value'])


def compare_annotations_sbert(manager: ModelManager, file1: str, file2: str):
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
            similarity_scores = [sbert_annotation_similarity(manager, ann, other_ann) for other_ann in ann2['annotations']]
            total_similarity += max(similarity_scores)
        similarity_score = total_similarity / len(shorter_list)
        print(f"Average similarity score for passage at index {i}: {similarity_score:.4f}")


def compare_phrases_sbert(manager: ModelManager, phrase1: str, phrase2: str):
    """Compare the similarity between two phrases."""
    similarity_score = sbert_phrase_similarity(manager, phrase1, phrase2)
    print(f"Similarity score between '{phrase1}' and '{phrase2}': {similarity_score:.4f}")


def evaluate_annotations_for_llm_model(manager: ModelManager, llm_name: str, package_names: List[str], annotated_dir: str, reference_dir: str, evaluation_dir: str):
    for package_name in package_names:
        annotated_file = find_json_file(annotated_dir, package_name, llm_name)
        reference_file = find_json_file(reference_dir, package_name)

        if not annotated_file or not reference_file:
            logging.warning(f"Skipping {package_name}: annotated or reference file not found for LLM '{llm_name}'")
            continue

        print(f"Evaluating annotations for {package_name} by LLM '{llm_name}'...")
        with open(annotated_file, 'r') as f_annotated, open(reference_file, 'r') as f_reference:
            annotated_data = json.load(f_annotated)
            reference_data = json.load(f_reference)

        evaluated_data = evaluate_file(manager, llm_name, reference_data, annotated_data)

        output_file = os.path.join(evaluation_dir, f"{llm_name}.{package_name}_evaluated.json")
        with open(output_file, 'w') as f_out:
            json.dump(evaluated_data, f_out, indent=4)

        logging.info(f"Evaluation completed for {package_name}")


def evaluate_annotations(manager: ModelManager, run_id: str):
    output_dir = os.path.join(BASE_DIR, 'output', run_id)
    annotated_dir = os.path.join(output_dir, 'annotated')
    reference_dir = os.path.join(BASE_DIR, 'reference_annotations')
    evaluation_dir = os.path.join(output_dir, 'evaluation')
    os.makedirs(evaluation_dir, exist_ok=True)

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
            evaluate_annotations_for_llm_model(manager, llm_name, package_names, annotated_dir, reference_dir, evaluation_dir)
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
        evaluate_annotations_for_llm_model(manager, llm_name, package_names, annotated_dir, reference_dir, evaluation_dir)


def find_json_file(directory: str, package_name: str, llm_name: str = None) -> str:
    for filename in os.listdir(directory):
        if filename.endswith('.json') and package_name in filename and (llm_name is None or llm_name in filename):
            return os.path.join(directory, filename)
    return None


def evaluate_file(manager: ModelManager, llm_name: str, reference_data: List[Dict], annotated_data: List[Dict]) -> List[Dict]:
    evaluated_data = []

    for ref_entry in reference_data:
        ref_passage = ref_entry['passage'].lower().strip()
        try:
            matching_entry = next((entry for entry in annotated_data if entry['passage'].lower().strip() == ref_passage), None)
        except KeyError as ke:
            logging.error(f"KeyError for passage: {ref_passage[:50]} in entry: {ref_entry}: {ke}")
            continue
        except AttributeError as ae:
            logging.error(f"AttributeError for passage: {ref_passage[:50]} in entry: {ref_entry}: {ae}")
            continue

        if not matching_entry:
            matching_entry = find_similar_passage(manager, ref_passage, annotated_data)

        if matching_entry:
            if 'annotations' not in ref_entry or 'annotations' not in matching_entry or not matching_entry['annotations']:
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
                word_level_similarity = rapidfuzz.fuzz.ratio(ref_ann['value'], ann['value']) / 100.0

                if semantic_similarity > best_semantic_similarity:
                    best_match = ann
                    best_semantic_similarity = semantic_similarity
                    best_word_level_similarity = word_level_similarity
            except ValueError as e:
                logging.error(f"Error processing annotation values: {e}", exc_info=True)

    return best_match, best_semantic_similarity, best_word_level_similarity

def find_similar_passage(manager: ModelManager, ref_passage: str, annotated_data: List[Dict]) -> Dict:
    if manager.current_model == 'fasttext':
        similarity_scores = [
            (entry, fasttext_phrase_similarity(manager, ref_passage, entry['passage']))
            for entry in annotated_data
        ]
    else:  # SBERT
        similarity_scores = [
            (entry, sbert_phrase_similarity(manager, ref_passage, entry['passage']))
            for entry in annotated_data
        ]
    similar_entries = [entry for entry, score in similarity_scores if score > 0.9]

    if len(similar_entries) == 1:
        return similar_entries[0]
    elif len(similar_entries) > 1:
        logging.info(f"Multiple similar passages found for: {ref_passage[:50]}...")
    else:
        logging.info(f"No similar passage found for: {ref_passage[:50]}...")

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

    if manager.current_model == 'fasttext':
        if 'value' not in ref_ann or 'value' not in ann:
            logging.warning(f"Value not found in annotation: {ref_ann}, {ann}")
            return 0
        if not isinstance(ref_ann['value'], str) or not isinstance(ann['value'], str):
            logging.warning(f"Value not a string: {ref_ann['value']}, {ann['value']} with types {type(ref_ann['value'])}, {type(ann['value'])}")
            return 0

        value_similarity = fasttext_phrase_similarity(manager, ref_ann['value'], ann['value'])
    else:  # SBERT
        value_similarity = sbert_phrase_similarity(manager, ref_ann['value'], ann['value'])

    return value_similarity


def main():
    argparser = argparse.ArgumentParser(description="Embedding Evaluation Tool")
    argparser.add_argument('--model', default='fasttext', choices=['fasttext', 'sbert'], help="Select the model to use.")
    argparser.add_argument('--create-dataset', action='store_true', help="Create a dataset from the files in the 'data' folder.")
    argparser.add_argument('--compare-files', nargs=2, metavar=('file1', 'file2'), help="Compare annotations in two files.")
    argparser.add_argument('--compare-phrases', nargs=2, metavar=('phrase1', 'phrase2'), help="Compare two phrases.")
    argparser.add_argument('--fine-tune', action='store_true', help="Fine-tune the selected model on the created dataset.")
    argparser.add_argument('--download-model', action='store_true', help="Download the selected model.")
    argparser.add_argument('--evaluate', action='store_true', help="Evaluate annotations against reference annotations.")
    argparser.add_argument('--run-id', type=str, help="Run ID for evaluation.")
    args = argparser.parse_args()

    # Create necessary directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FASTTEXT_DIR, exist_ok=True)
    os.makedirs(SBERT_DIR, exist_ok=True)

    if not tokenizer_downloaded():
        download_tokenizer()

    # Handle --create-dataset separately as it doesn't require a model
    if args.create_dataset:
        tokenizer = create_custom_sentence_tokenizer()
        create_dataset(tokenizer)
        return

    # Initialize the model manager
    manager = ModelManager()

    if args.download_model:
        if args.model == 'fasttext':
            manager.download_fasttext_model()
        elif args.model == 'sbert':
            print("SBERT model will be downloaded automatically when needed.")
        return

    if args.model == 'fasttext':
        manager.load_fasttext_model()
    elif args.model == 'sbert':
        if os.path.exists(manager.sbert_finetuned_path):
            manager.load_finetuned_sbert_model(manager.sbert_finetuned_path)
        else:
            manager.load_sbert_model()
    else:
        print("Invalid model selected. Please choose either 'fasttext' or 'sbert'.")
        sys.exit(1)

    if args.evaluate:
        if not args.run_id:
            print("Please provide a --run-id when using --evaluate.")
            sys.exit(1)
        evaluate_annotations(manager, args.run_id)
    elif args.compare_files:
        if args.model == 'fasttext':
            compare_annotations_fasttext(manager, args.compare_files[0], args.compare_files[1])
        elif args.model == 'sbert':
            compare_annotations_sbert(manager, args.compare_files[0], args.compare_files[1])
    elif args.compare_phrases:
        if args.model == 'fasttext':
            compare_phrases_fasttext(manager, args.compare_phrases[0], args.compare_phrases[1])
        elif args.model == 'sbert':
            compare_phrases_sbert(manager, args.compare_phrases[0], args.compare_phrases[1])
    elif args.fine_tune:
        if args.model == 'fasttext':
            dataset_path = os.path.join(DATA_DIR, 'dataset.txt')
            if os.path.exists(dataset_path):
                manager.fine_tune_fasttext_model(dataset_path)
            else:
                print("Dataset not found. Please create the dataset first using --create-dataset.")
        elif args.model == 'sbert':
            dataset_path = os.path.join(DATA_DIR, 'dataset.txt')
            if os.path.exists(dataset_path):
                manager.load_sbert_model()
                manager.fine_tune_sbert_unsupervised(dataset_path)
            else:
                print("Dataset not found. Please create the dataset first using --create-dataset.")
    else:
        print("No valid arguments provided. Please use one of the following options:")
        print("  --create-dataset: Create a dataset from the files in the 'data' folder.")
        print("  --compare-files <file1> <file2>: Compare annotations in two files.")
        print("  --compare-phrases <phrase1> <phrase2>: Compare two phrases.")
        print("  --fine-tune: Fine-tune the selected model on the created dataset.")
        print("  --evaluate --run-id <run_id>: Evaluate annotations against reference annotations.")


if __name__ == '__main__':
    main()
