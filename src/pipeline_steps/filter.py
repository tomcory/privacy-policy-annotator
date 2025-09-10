import json
import os

from bs4 import BeautifulSoup
from py3langid.langid import LanguageIdentifier, MODEL_FILE

from src import util

accepted_languages = ['en']

def filter_language(run_id: str, pkg: str, html: str):
    text = BeautifulSoup(html, 'html.parser').get_text()
    language, confidence = identify_language(pkg, text)

    # log the language detection result
    util.append_to_file(
        f"../output/{run_id}/log/language.jsonl",
        json.dumps({'pkg': pkg, 'language': language, 'confidence': float(confidence)})
    )
    if confidence < 0.9:
        path = f"../output/{run_id}/policies/langid/unsure"
        if not os.path.exists(path):
            os.mkdir(path)
        file_name = f"{path}/{pkg}.html"
        util.write_to_file(file_name, html)
    else:
        # copy the file to the other_languages folder
        # check whether the folder exists, if not, create it
        path = f"../output/{run_id}/policies/langid/{language}"
        if not os.path.exists(path):
            os.mkdir(path)
        file_name = f"{path}/{pkg}.html"
        util.write_to_file(file_name, html)
    # if the html is empty or contains the string "privacy policy" (case-insensitive), return None
    if language in accepted_languages:
        # if 'privacy policy' in text.lower():
        #     # copy the file to the accepted folder
        #     file_name = f"../output/{run_id}/policies/accepted/{pkg}.html"
        #     util.write_to_file(file_name, html)
        #     return None
        pass # TODO: implement


def identify_language(self, text: str):
    identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
    language, confidence = identifier.classify(text)
    return language, confidence