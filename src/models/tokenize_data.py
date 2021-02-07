from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import joblib


def tokenize_review(x_tr, x_val, max_len_text):
    x_tokenizer = Tokenizer()
    x_tokenizer.fit_on_texts(list(x_tr))

    # convert text sequences into integer sequences
    x_tr = x_tokenizer.texts_to_sequences(x_tr)
    x_val = x_tokenizer.texts_to_sequences(x_val)

    # padding zero upto maximum length
    x_tr = pad_sequences(x_tr, maxlen=max_len_text, padding='post')
    x_val = pad_sequences(x_val, maxlen=max_len_text, padding='post')

    x_voc_size = len(x_tokenizer.word_index) + 1
    joblib.dump(x_tokenizer, 'models/x_tokenizer.pkl')
    return x_tr, x_val, x_voc_size


def tokenize_comment(y_tr, y_val, max_len_summary):
    y_tokenizer = Tokenizer()
    y_tokenizer.fit_on_texts(list(y_tr))

    # convert summary sequences into integer sequences
    y_tr = y_tokenizer.texts_to_sequences(y_tr)
    y_val = y_tokenizer.texts_to_sequences(y_val)

    # padding zero upto maximum length
    y_tr = pad_sequences(y_tr, maxlen=max_len_summary, padding='post')
    y_val = pad_sequences(y_val, maxlen=max_len_summary, padding='post')

    y_voc_size = len(y_tokenizer.word_index) + 1
    joblib.dump(y_tokenizer, 'models/y_tokenizer.pkl')
    return y_tr, y_val, y_voc_size


def main():
    print("This only executes when %s is imported rather than executed" % __file__)


if __name__ == "__main__":
    main()
