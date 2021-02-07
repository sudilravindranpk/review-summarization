from sklearn.model_selection import train_test_split
import pandas as pd
from keras import backend as K
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional, Dropout

from models.attention_layer import AttentionLayer
from features.tokenize_data import tokenize_review, tokenize_comment
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot

max_len_text = 80
max_len_summary = 10
latent_dim = 500


def create_trained_model(processed_file_path):
    data = pd.read_csv(processed_file_path)
    x_tr, x_val, y_tr, y_val = train_test_split(data['cleaned_text'], data['cleaned_summary'], test_size=0.1,
                                                random_state=0, shuffle=True)
    x_tr, x_val, x_voc_size = tokenize_review(x_tr, x_val, max_len_text)
    y_tr, y_val, y_voc_size = tokenize_comment(y_tr, y_val, max_len_summary)
    encoder_model_inference, decoder_model_inference = train_model(x_voc_size, y_voc_size, x_tr, y_tr, x_val, y_val)
    encoder_model_inference.save('models/enc.h5')
    decoder_model_inference.save('models/dec.h5')


def train_model(x_voc_size, y_voc_size, x_tr, y_tr, x_val, y_val):
    K.clear_session()
    # Encoder
    encoder_inputs = Input(shape=(max_len_text,), name="encoder_inputs")
    enc_emb = Embedding(x_voc_size, latent_dim, trainable=True)(encoder_inputs)

    # bidirectional LSTM 1
    encoder_lstm1 = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True),
                                  name="bidirectional_LSTM_1")
    encoder_output1, state_h1, state_c1, backward_state_h1, backward_state_c1 = encoder_lstm1(enc_emb)
    encoder_output_dropout_01 = Dropout(0.3)(encoder_output1)

    # bidirectional LSTM 2
    encoder_lstm2 = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True),
                                  name="bidirectional_LSTM_2")
    encoder_output2, state_h2, state_c2, backward_state_h2, backward_state_c2 = encoder_lstm2(encoder_output_dropout_01)
    encoder_output_dropout_02 = Dropout(0.3)(encoder_output2)

    # bidirectional  LSTM 3
    encoder_lstm3 = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True),
                                  name="bidirectional_LSTM_3")
    encoder_output, state_h3, state_c3, backward_state_h3, backward_state_c3 = encoder_lstm3(encoder_output_dropout_02)
    encoder_state_h = Concatenate()([state_h3, backward_state_h3])
    encoder_state_c = Concatenate()([state_c3, backward_state_c3])

    decoder_inputs = Input(shape=(None,), name="decoder_inputs")
    dec_emb_layer = Embedding(y_voc_size, latent_dim, trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True, name="decoder_LSTM")
    decoder_output, decoder_state_h, decoder_state_c = decoder_lstm(dec_emb,
                                                                    initial_state=[encoder_state_h, encoder_state_c])

    attn_layer = AttentionLayer(name='attention_layer')
    attn_out, attn_states = attn_layer([encoder_output, decoder_output])

    # Concat attention output and decoder LSTM output
    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_output, attn_out])

    # Dense layer
    decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax', name='dense_layer_with_soft_max'),
                                    name='time_distributed_layer')
    decoder_dense_output = decoder_dense(decoder_concat_input)

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_dense_output)
    with open('models/model_summary.txt', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    train(model, x_tr, y_tr, x_val, y_val)

    # Encoder Inference Model
    encoder_model_inference = Model(encoder_inputs, [encoder_output, encoder_state_h, encoder_state_c])

    # Decoder Inference
    # Below tensors will hold the states of the previous time step
    decoder_state_h = Input(shape=(latent_dim * 2,))
    decoder_state_c = Input(shape=(latent_dim * 2,))
    decoder_intermittent_state_input = Input(shape=(max_len_text, latent_dim * 2))

    # Get Embeddings of Decoder Sequence
    decoder_embedding_inference = dec_emb_layer(decoder_inputs)

    # Predict Next Word in Sequence, Set Initial State to State from Previous Time Step
    decoder_output_inference, decoder_state_inference_h, decoder_state_inference_c = decoder_lstm(
        decoder_embedding_inference, initial_state=[decoder_state_h, decoder_state_c])

    # Attention Inference
    attention_layer = AttentionLayer()
    attention_out_inference, attention_state_inference = attention_layer(
        [decoder_intermittent_state_input, decoder_output_inference])
    decoder_inference_concat = Concatenate(axis=-1)([decoder_output_inference, attention_out_inference])

    # Dense Softmax Layer to Generate Prob. Dist. Over Target Vocabulary
    decoder_output_inference = decoder_dense(decoder_inference_concat)

    # Final Decoder Model
    decoder_model_inference = Model(
        [decoder_inputs, decoder_intermittent_state_input, decoder_state_h, decoder_state_c],
        [decoder_output_inference, decoder_state_inference_h, decoder_state_inference_c])
    return encoder_model_inference, decoder_model_inference


def train(model, x_tr, y_tr, x_val, y_val):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    history = model.fit([x_tr, y_tr[:, :-1]], y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:, 1:], epochs=200,
                        callbacks=[es], batch_size=512, validation_data=(
            [x_val, y_val[:, :-1]], y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:]))
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.savefig('models/training_history.png')


if __name__ == "__main__":
    create_trained_model()
