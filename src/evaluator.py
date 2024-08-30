import os
import sys
import nltk
import numpy
import timeit
import fasttext.util
from typing import Union
from datetime import timedelta
from bs4 import BeautifulSoup
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from gensim.models import FastText
from gensim.models import fasttext as ft
from gensim.utils import simple_preprocess
from scipy.spatial.distance import cosine


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

def fasttest_model_downloaded():
    return os.path.exists('../fasttext_dataset/cc.en.300.bin')

def download_fasttext_model():
    print("Downloading FastText model 'cc.en.300.bin'...")
    fasttext.util.download_model('en', if_exists='ignore')
    print("Download complete.")
    # move the downloaded files to the "fasttext_dataset" folder
    os.rename('cc.en.300.bin', '../fasttext_dataset/cc.en.300.bin')
    os.rename('cc.en.300.bin.gz', '../fasttext_dataset/cc.en.300.bin.gz')
    print("Model moved to 'fasttext_dataset' folder.")

def create_custom_sentence_tokenizer():
    """
    Create a custom sentence tokenizer that doesn't split sentences at common legal abbreviations.

    :return: PunktSentenceTokenizer: A sentence tokenizer customized with the pre-defined abbreviations.
    """
    punkt_param = PunktParameters()
    # add common abbreviations to the tokenizer so that it doesn't split sentences at these abbreviations
    legal_abbreviations = [
        "art.", "lit.", "cal.", "cit.", "civ.",
        "sec.", "cl.", "v.", "p.", "pp.", "no.", "n.", "ann.", "ord.", "par.", "para.", "fig.",
        "ex.", "exh.", "doc.", "vol.", "ed.", "ch.", "pt.", "inc.", "corp.", "plc.",
        "gov.", "min.", "dept.", "div.", "comm.", "ag.", "adj.", "adv.", "aff.", "agr.",
        "app.", "att.", "bldg.", "bk.", "bul.", "cir.", "co.", "com.", "conc.", "conf.",
        "const.", "def.", "dept.", "det.", "dev.", "dir.", "dist.", "ed.", "est.", "exp.",
        "ext.", "fig.", "gen.", "gov.", "hist.", "il.", "inc.", "ind.", "info.", "int.",
        "jr.", "jud.", "leg.", "llc.", "lt.", "maj.", "max.", "med.", "mem.", "misc.",
        "mtg.", "nat.", "nr.", "org.", "ph.", "pl.", "pub.", "reg.", "rep.", "rev.",
        "sci.", "sec.", "ser.", "st.", "sub.", "supp.", "techn.", "temp.", "treas.",
        "univ.", "vol.", "vs.", "yr.", "zoning.", "appx.", "supp.", "admr.", "dba.",
        "et al.", "et seq.", "etc.", "i.e.", "e.g.", "ca.", "id.", "infra.", "supra.",
        "viz.", "vs.", "am.", "pm.", "corp.", "ltd.", "inc.", "co.", "re.", "u.s.",
        "u.k.", "ca.", "fla.", "ny.", "tex.", "jr.", "sr.", "rev.", "dr.", "mr.", "mrs.",
        "prof.", "pres.", "gov.", "sen.", "rep.", "gen.", "col.", "maj.", "lt.", "sgt.",
        "cpt.", "det.", "st.", "m.d.", "ph.d.", "esq.", "l.c.", "l.l.p.", "l.p.", "s.p.",
        "2d", "3d", "4th", "5th", "6th", "7th", "8th", "9th", "10th",  # Legal citation editions
        "i.", "ii.", "iii.", "iv.", "v.", "vi.", "vii.", "viii.", "ix.", "x.",  # Roman numerals
        "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.",  # Numberings
        "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "oct.", "nov.", "dec."  # Months
    ]
    punkt_param.abbrev_types.update(legal_abbreviations)
    return PunktSentenceTokenizer(punkt_param)

def create_dataset(tokenizer: PunktSentenceTokenizer):
    """
    Create a dataset from the files given in the "fasttext_dataset" folder.

    Possible file types are .html files and .txt files. The .html files are parsed using BeautifulSoup and the text is extracted,
    assuming that the contained policy is cleaned already and formatted as per the Usable Privacy Policy Project (i.e., each sentence
    is separated by '|||').
    Sentences contained herein are then split using the custom sentence tokenizer.

    The .txt files are assumed to already be cleaned and are simply parsed line by line, without any further processing.

    :param tokenizer: PunktSentenceTokenizer: Tokenizer to split the text into sentences.
    :return: None
    """

    # verify that the "fasttext_dataset" folder exists
    if not os.path.exists('../fasttext_dataset'):
        os.mkdir('../fasttext_dataset')
        print("Created 'fasttext_dataset' folder. Please fill it with the required data.")
        sys.exit(0)

    # if the "dataset.txt" file already exists, ask the user if they want to overwrite it
    if os.path.exists('../fasttext_dataset/dataset.txt'):
        overwrite = input("The 'dataset.txt' file already exists. Do you want to overwrite it? (yes/n) ")
        if overwrite.lower() != 'yes':
            return
        else:
            overwrite = input("Are you sure you want to overwrite the 'dataset.txt' file? (yes/n) ")
            if overwrite.lower() != 'yes':
                return
            else:
                print("Overwriting 'dataset.txt' file...")
                os.remove('../fasttext_dataset/dataset.txt')

    with open('../fasttext_dataset/dataset.txt', 'w', encoding='utf-8') as dataset:
        total_sentences = 0
        # for each .html file in the directory, read it and extract the text
        # then, split the string into sentences and append each sentence to the "dataset.txt" file
        for file in os.listdir('../fasttext_dataset'):
            if file.endswith('.html'):
                with open(f'../fasttext_dataset/{file}', 'r') as f:
                    soup = BeautifulSoup(f, 'html.parser')
                    text = soup.get_text(separator='|||', strip=True)
                    lines = text.split('|||')
                    # remove empty lines
                    lines = [line for line in lines if line.strip()]
                    for line in lines:
                        for sentence in tokenizer.tokenize(line):
                            if sentence.strip():  # Check if the sentence is not empty
                                dataset.write(sentence + '\n')
                                total_sentences += 1

            # for each .txt file in the directory, if it is not "dataset.txt", read it line by line and append each line to the "dataset.txt" file
            elif file.endswith('.txt') and file != 'dataset.txt':
                with open(f'../fasttext_dataset/{file}', 'r', encoding='utf-8') as f:
                    for line in f:
                        # if the line is empty, skip it
                        if not line.strip():
                            continue

                        dataset.write(line.strip() + '\n')  # Ensure no empty lines are written
                        total_sentences += 1

    print(f"Dataset created successfully. {total_sentences} sentences written to 'dataset.txt'.")

def preprocess_corpus(file_path: str):
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield simple_preprocess(line)

def fine_tune_embedding_model():
    corpus = list(preprocess_corpus('../fasttext_dataset/dataset.txt'))
    print("Corpus loaded successfully.")

    model = ft.load_facebook_model('../fasttext_dataset/cc.en.300.bin')
    print("Model loaded successfully.")

    print("Fine-tuning model...")
    start_time = timeit.default_timer()
    model.build_vocab(corpus, update=True)
    model.train(corpus, total_examples=len(corpus), epochs=10)
    end_time = timeit.default_timer()
    print(f"Vector embedding model training completed in {timedelta(seconds=end_time - start_time)}.")

    model.save('../fasttext_dataset/fasttext_finetuned.model')
    print("Model saved successfully.")
    return model

def train_embedding_model():
    corpus = list(preprocess_corpus('../fasttext_dataset/dataset.txt'))
    print("Corpus loaded successfully.")

    model = FastText(
        vector_size=100,
        window=5,
        min_count=2,
        workers=8,
        sg=1
    )

    print("Training model...")
    start_time = timeit.default_timer()
    model.build_vocab(corpus)
    model.train(corpus, total_examples=len(corpus), epochs=10)
    end_time = timeit.default_timer()
    print(f"Vector embedding model training completed in {timedelta(seconds=end_time - start_time)}.")

    model.save('../fasttext_dataset/fasttext_custom.model')
    print("Model saved successfully.")
    return model

def load_embedding_model(model_path: str) -> Union[FastText, None]:
    if not os.path.exists(model_path):
        print("Model file not found. Please train/finetune a model first or provide the correct path.")
        return
    else:
        print(f"Loading model from '{model_path}'...")
        return FastText.load(model_path)

def get_word_vector(model: FastText, word: str) -> numpy.ndarray:
    try:
        return model.wv[word]
    except KeyError:
        print(f"Word '{word}' not found in the model's vocabulary.")
        return numpy.zeros(model.vector_size)

def get_phrase_vector(model: FastText, phrase: str) -> numpy.ndarray:
    words = simple_preprocess(phrase)
    word_vectors = [get_word_vector(model, word) for word in words if word in model.wv]
    if not word_vectors:
        print(f"No words found in the model's vocabulary for the phrase '{phrase}'.")
        return numpy.zeros(model.vector_size)
    return numpy.mean(word_vectors, axis=0)

def word_similarity(model: FastText, word1: str, word2: str) -> float:
    return 1 - cosine(get_word_vector(model, word1), get_word_vector(model, word2))

def phrase_similarity(model: FastText, phrase1: str, phrase2: str) -> float:
    return 1 - cosine(get_phrase_vector(model, phrase1), get_phrase_vector(model, phrase2))

def main():
    if not tokenizer_downloaded():
        download_tokenizer()

    if not fasttest_model_downloaded():
        download_fasttext_model()

    embedding_model = load_embedding_model('../fasttext_dataset/fasttext_finetuned.model')

    phrase1 = "datenschutz"
    phrase2 = "data protection policy"
    similarity = phrase_similarity(embedding_model, phrase1, phrase2)
    print(f"Similarity between '{phrase1}' and '{phrase2}': {similarity:.4f}")

    # find the most similar words to "privacy" in the model's vocabulary
    word = "privacy"
    similar_words = embedding_model.wv.most_similar(word)
    print(f"Words most similar to '{word}':")
    for similar_word, score in similar_words:
        print(f"{similar_word}: {score:.4f}")


if __name__ == '__main__':
    main()
