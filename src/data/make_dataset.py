# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import numpy as np
from data.contractions import contraction_mapping

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def text_cleaner(text):
    new_string = text.lower()
    new_string = BeautifulSoup(new_string, "lxml").text
    new_string = re.sub(r'\([^)]*\)', '', new_string)
    new_string = re.sub('"', '', new_string)
    new_string = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in new_string.split(" ")])
    new_string = re.sub(r"'s\b", "", new_string)
    new_string = re.sub("[^a-zA-Z]", " ", new_string)
    tokens = [w for w in new_string.split() if not w in stop_words]
    long_words = []
    for i in tokens:
        if len(i) >= 3:
            long_words.append(i)
    return (" ".join(long_words)).strip()


def summary_cleaner(text):
    new_string = re.sub('"', '', text)
    new_string = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in new_string.split(" ")])
    new_string = re.sub(r"'s\b", "", new_string)
    new_string = re.sub("[^a-zA-Z]", " ", new_string)
    new_string = new_string.lower()
    tokens = new_string.split()
    new_string = ''
    for i in tokens:
        if len(i) > 1:
            new_string = new_string + i + ' '
    return new_string


def make_data_set(input_filepath, output_filepath):
    data = pd.read_csv(input_filepath, quoting=3, error_bad_lines=False, warn_bad_lines=False)
    data = data.drop_duplicates(subset=['Text'], inplace=False)
    data = data.dropna(axis=0, inplace=False)
    cleaned_text = []
    for t in data['Text']:
        cleaned_text.append(text_cleaner(t))
    cleaned_summary = []
    for t in data['Summary']:
        cleaned_summary.append(summary_cleaner(t))

    cleaned_data = pd.DataFrame(
        {'cleaned_text': cleaned_text,
         'cleaned_summary': cleaned_summary})
    cleaned_data = cleaned_data.replace('', np.nan, inplace=False)
    cleaned_data = cleaned_data.dropna(axis=0, inplace=False)

    cleaned_data['cleaned_summary'] = cleaned_data['cleaned_summary'].apply(lambda x: '_START_ ' + x + ' _END_')
    cleaned_data.to_csv(output_filepath)
    print("cleaned the data and moved the cleaned data to the root/data/processed folder")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    make_data_set()
