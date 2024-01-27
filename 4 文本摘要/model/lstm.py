import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
from transformers import BertTokenizer, TFBertModel


def LSTM(max_input_len, max_output_len, input_vocab_size, output_vocab_size, latent_dim):
    encoder_inputs = tf.keras.layers.Input(shape=(max_input_len,))
    encoder_embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size, output_dim=latent_dim )(encoder_inputs)
    encoder, state_h, state_c = tf.keras.layers.LSTM(latent_dim , return_state=True)(encoder_embedding)

    decoder_inputs = tf.keras.layers.Input(shape=(max_output_len,))
    decoder_embedding = tf.keras.layers.Embedding(input_dim=output_vocab_size, output_dim=latent_dim)(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
    decoder_dense = tf.keras.layers.Dense(output_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model