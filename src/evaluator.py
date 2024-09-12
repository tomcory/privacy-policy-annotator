import os
import re
import sys
import nltk
import json
import logging
import argparse
import numpy as np
import gensim.downloader as api
from datasets import Dataset
from bs4 import BeautifulSoup
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from gensim.models import FastText, KeyedVectors

# Global variables for the FastText model
fasttext_model_name = "fasttext-wiki-news-subwords-300"  # FastText model for sentence embeddings
# fasttext_model_name = "conceptnet-numberbatch-17-06-300"
fasttext_model = None
# Global variables for the SBERT model
sbert_model_name = "sentence-transformers/all-mpnet-base-v2"  # SBERT model for phrase embeddings
sbert_model = None


class ModelManager:
    def __init__(self, fasttext_model_name="fasttext-wiki-news-subwords-300", sbert_model_name="sentence-transformers/all-mpnet-base-v2"):
        self.fasttext_model_name = fasttext_model_name
        self.sbert_model_name = sbert_model_name
        self.fasttext_model = None
        self.sbert_model = None

    def download_fasttext_model(self):
        print("Downloading FastText model...")
        fasttext_model_path = api.load(self.fasttext_model_name, return_path=True)
        print("Download complete.")
        return fasttext_model_path

    def load_pretrained_fasttext_model(self, model_path: str):
        print("Loading pre-trained FastText model...")
        self.fasttext_model = KeyedVectors.load_word2vec_format(model_path, binary=False)
        print("Pre-trained FastText model loaded successfully.")
        return self.fasttext_model

    def load_finetuned_fasttext_model(self, model_path: str):
        print("Loading fine-tuned FastText model...")
        self.fasttext_model = FastText.load(model_path)
        print("Fine-tuned FastText model loaded successfully.")
        return self.fasttext_model

    def load_sbert_model(self):
        self.sbert_model = SentenceTransformer(self.sbert_model_name).cuda()
        print(f"SBERT model '{self.sbert_model_name}' loaded successfully.")

    def load_finetuned_sbert_model(self, model_path: str):
        self.sbert_model = SentenceTransformer(model_path).cuda()
        print(f"Fine-tuned SBERT model loaded successfully from '{model_path}'.")

    def fine_tune_fasttext_model(self, dataset_file: str):
        sentences = load_sentences(dataset_file)
        tokenized_data = [sentence.split() for sentence in sentences]
        self.fasttext_model = FastText(vector_size=300, window=5, min_count=1)
        self.fasttext_model.build_vocab(corpus_iterable=tokenized_data)
        print(f"Vocabulary built. Number of unique tokens: {len(self.fasttext_model.wv.index_to_key)}")
        print("Training FastText model...")
        self.fasttext_model.train(corpus_iterable=tokenized_data, total_examples=len(tokenized_data), epochs=10)
        print("FastText model fine-tuned successfully.")
        self.fasttext_model.save(f'../fasttext_dataset/{self.fasttext_model_name}')
        print("FastText model saved successfully.")

    def fine_tune_sbert_unsupervised(self, file_path, output_dir='../fine_tuned_sbert'):
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
        words = phrase.split()
        vectors = [self.fasttext_model.wv[word] for word in words if word in self.fasttext_model.wv]
        if not vectors:
            raise ValueError(f"None of the words in '{phrase}' were found in the vocabulary.")
        return np.mean(vectors, axis=0)

    def get_sbert_phrase_embedding(self, phrase: str) -> np.ndarray:
        return self.sbert_model.encode(phrase, convert_to_tensor=True).cpu().numpy()


def tokenizer_downloaded():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        return True
    except LookupError:
        return False


def download_tokenizer():
    print("Downloading NLTK Punkt tokenizer...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    print("Download complete.")


def download_fasttext_model():
    """Download the FastText model for sentence embeddings."""
    print("Downloading FastText model...")
    fasttext_model_path = api.load(fasttext_model_name, return_path=True)
    print("Download complete.")
    return fasttext_model_path


def load_pretrained_fasttext_model(model_path: str):
    """Load the pre-trained FastText model from a given path."""
    global fasttext_model
    print("Loading pre-trained FastText model...")
    fasttext_model = KeyedVectors.load_word2vec_format(model_path, binary=False)
    print("Pre-trained FastText model loaded successfully.")
    return fasttext_model


def load_finetuned_fasttext_model(model_path: str):
    """Load a fine-tuned FastText model from a given path."""
    global fasttext_model
    print("Loading fine-tuned FastText model...")
    fasttext_model = FastText.load(model_path)
    print("Fine-tuned FastText model loaded successfully.")
    return fasttext_model


def load_sbert_model():
    """Load the SBERT model."""
    global sbert_model
    sbert_model = SentenceTransformer(sbert_model_name).cuda()  # Send model to GPU
    print(f"SBERT model '{sbert_model_name}' loaded successfully.")


def load_finetuned_sbert_model(model_path: str):
    """Load a fine-tuned SBERT model from a given path."""
    global sbert_model
    sbert_model = SentenceTransformer(model_path).cuda()
    print(f"Fine-tuned SBERT model loaded successfully from '{model_path}'.")


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


def create_dataset(tokenizer: PunktSentenceTokenizer):
    """Create a dataset from the files given in the 'fasttext_dataset' folder."""
    if not os.path.exists('../fasttext_dataset'):
        os.mkdir('../fasttext_dataset')
        print("Created 'fasttext_dataset' folder. Please fill it with the required data.")
        sys.exit(0)

    if os.path.exists('../fasttext_dataset/dataset.txt'):
        overwrite = input("The 'dataset.txt' file already exists. Do you want to overwrite it? (yes/n) ")
        if overwrite.lower() != 'yes':
            return
        os.remove('../fasttext_dataset/dataset.txt')

    def extract_lines_uppp_html(file_path: str) -> list:
        with open(file_path, 'r') as f:
            soup = BeautifulSoup(f, 'html.parser')
            text = soup.get_text(separator='|||', strip=True)
            return [line for line in text.split('|||') if line.strip()]

    def extract_lines_uppp_txt(file_path: str) -> list:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def extract_lines_pa_txt(file_path: str) -> list:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('<')]

    with open('../fasttext_dataset/dataset.txt', 'w', encoding='utf-8') as dataset:
        total_sentences = 0
        for file in os.listdir('../fasttext_dataset'):
            if re.match(r'^\d+.*\.html$', file):
                lines = extract_lines_uppp_html(f'../fasttext_dataset/{file}')
            elif re.match(r'^\d+.*\.txt$', file) and file != 'dataset.txt':
                lines = extract_lines_uppp_txt(f'../fasttext_dataset/{file}')
            elif re.match(r'^\D.*\.html$', file):
                lines = extract_lines_pa_txt(f'../fasttext_dataset/{file}')
            else:
                continue

            for line in lines:
                for sentence in tokenizer.tokenize(line):
                    if sentence.strip():
                        dataset.write(sentence.strip() + '\n')
                        total_sentences += 1

    print(f"Dataset created successfully. {total_sentences} sentences written to 'dataset.txt'.")


def load_sentences(file_path):
    """Load sentences from a dataset file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def fine_tune_fasttext_model(dataset_file: str):
    """Fine-tune the FastText model using our own dataset of sentences."""
    global fasttext_model

    # Load the dataset
    sentences = load_sentences(dataset_file)
    tokenized_data = [sentence.split() for sentence in sentences]

    # Initialize FastText model and build the vocabulary from the dataset
    fasttext_model = FastText(vector_size=300, window=5, min_count=1)
    fasttext_model.build_vocab(corpus_iterable=tokenized_data)
    print(f"Vocabulary built. Number of unique tokens: {len(fasttext_model.wv.index_to_key)}")

    # Fine-tune the model on the dataset
    print("Training FastText model...")
    fasttext_model.train(corpus_iterable=tokenized_data, total_examples=len(tokenized_data), epochs=10)
    print("FastText model fine-tuned successfully.")

    # Save the fine-tuned model
    fasttext_model.save(f'../fasttext_dataset/{fasttext_model_name}')
    print("FastText model saved successfully.")


def create_input_examples(sentences):
    """Create InputExamples for unsupervised fine-tuning by duplicating sentences."""
    # We treat each sentence as its own positive pair
    return [InputExample(texts=[sentence, sentence]) for sentence in sentences]


def load_dataset_for_finetuning(file_path: str):
    """Load the dataset from a file and prepare it for fine-tuning with SBERT."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Create a Hugging Face Dataset object from the list of sentences
    dataset = Dataset.from_dict({"text": lines})

    # Return list of InputExamples for SentenceTransformer fine-tuning
    return [InputExample(texts=[line]) for line in lines]


def fasttext_phrase_similarity(manager: ModelManager, phrase1: str, phrase2: str) -> float:
    vec1 = manager.get_fasttext_phrase_embedding(phrase1)
    vec2 = manager.get_fasttext_phrase_embedding(phrase2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


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


def main():
    argparser = argparse.ArgumentParser(description="SBERT Embedding Evaluation")
    argparser.add_argument('--create-dataset', action='store_true', help="Create a dataset from the files in the 'fasttext_dataset' folder.")
    argparser.add_argument('--model', default='fasttext', choices=['fasttext', 'sbert'], help="Select the model to use.")
    argparser.add_argument('--compare-files', nargs=2, metavar=('file1', 'file2'), help="Compare annotations in two files.")
    argparser.add_argument('--compare-phrases', nargs=2, metavar=('phrase1', 'phrase2'), help="Compare two phrases.")
    argparser.add_argument('--fine-tune', action='store_true', help="Fine-tune the pre-trained SBERT model on the created dataset.")
    args = argparser.parse_args()

    if not tokenizer_downloaded():
        download_tokenizer()

    manager = ModelManager()

    if args.model == 'fasttext':
        if os.path.exists(f'../fasttext_dataset/{manager.fasttext_model_name}'):
            manager.load_finetuned_fasttext_model(f'../fasttext_dataset/{manager.fasttext_model_name}')
        else:
            fasttext_model_path = manager.download_fasttext_model()
            manager.load_pretrained_fasttext_model(fasttext_model_path)
    elif args.model == 'sbert':
        if os.path.exists('../fine_tuned_sbert'):
            manager.load_finetuned_sbert_model('../fine_tuned_sbert')
        else:
            manager.load_sbert_model()
    else:
        print("Invalid model selected. Please choose either 'fasttext' or 'sbert'.")
        sys.exit(1)

    if args.create_dataset:
        tokenizer = create_custom_sentence_tokenizer()
        create_dataset(tokenizer)
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
            if os.path.exists('../fasttext_dataset/dataset.txt'):
                manager.fine_tune_fasttext_model('../fasttext_dataset/dataset.txt')
            else:
                print("Dataset not found. Please create the dataset first using --create-dataset.")
        elif args.model == 'sbert':
            if os.path.exists('../fasttext_dataset/dataset.txt'):
                manager.fine_tune_sbert_unsupervised('../fasttext_dataset/dataset.txt')
            else:
                print("Dataset not found. Please create the dataset first using --create-dataset.")
    else:
        print("No valid arguments provided. Please use one of the following options:")
        print("  --create-dataset: Create a dataset from the files in the 'fasttext_dataset' folder.")
        print("  --compare-files <file1> <file2>: Compare annotations in two files.")
        print("  --compare-phrases <phrase1> <phrase2>: Compare two phrases.")
        print("  --fine-tune: Fine-tune the SBERT model on the created dataset.")


if __name__ == '__main__':
    main()
