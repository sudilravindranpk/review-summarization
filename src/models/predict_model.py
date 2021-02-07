import nltk
from nltk.corpus import stopwords
from models.attention_layer import AttentionLayer
from data.contractions import contraction_mapping
from bs4 import BeautifulSoup
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import joblib
import logging

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
x_tokenizer = joblib.load('features/x_tokenizer.pkl')
y_tokenizer = joblib.load('features/y_tokenizer.pkl')
encoder = load_model('models/enc.h5')
decoder = load_model('models/dec.h5', custom_objects={'AttentionLayer': AttentionLayer})
reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index
max_len_text = 80
max_len_summary = 10


def text_cleaner(text):
    new_string = text.lower()
    new_string = BeautifulSoup(new_string, "lxml").text
    new_string = re.sub(r'\([^)]*\)', '', new_string)
    new_string = re.sub('"', '', new_string)
    new_string = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in new_string.split(" ")])
    new_string = re.sub(r"'s\b", "", new_string)
    new_string = re.sub("[^a-zA-Z]", " ", new_string)
    tokens = [w for w in new_string.split() if w not in stop_words]
    long_words = []
    for i in tokens:
        if len(i) >= 3:
            long_words.append(i)
    return (" ".join(long_words)).strip()


def tokenize_pad_sequence(text):
    x_tr = x_tokenizer.texts_to_sequences(text)
    x_tr = pad_sequences(x_tr, maxlen=max_len_text, padding='post')
    return x_tr


def seq2summary(input_seq):
    new_string = ''
    for i in input_seq:
        if (i != 0 and i != target_word_index['start']) and i != target_word_index['end']:
            new_string = new_string + reverse_target_word_index[i] + ' '
    return new_string


def seq2text(input_seq):
    new_string = ''
    for i in input_seq:
        if i != 0:
            new_string = new_string + reverse_source_word_index[i] + ' '
    return new_string


def decode_sequence(input_seq):
    e_out, e_state_h, e_state_c = encoder.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_word_index['start']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, decoder_state_h, decoder_state_c = decoder.predict([target_seq] + [e_out, e_state_h, e_state_c])
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        if sampled_token != 'end':
            decoded_sentence += ' ' + sampled_token

        if sampled_token == 'end' or len(decoded_sentence.split()) >= (max_len_summary - 1):
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        e_state_h = decoder_state_h
        e_state_c = decoder_state_c

    return decoded_sentence


def predict(review_text):
    review_text = review_text if review_text is not None else '"I was always a white truffle fan'
    cleaned_text = text_cleaner(review_text)
    tokenized_padded_text = tokenize_pad_sequence([cleaned_text])
    return decode_sequence(tokenized_padded_text.reshape(1, max_len_text))


if __name__ == '__main__':
    predict()