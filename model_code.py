import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import pandas as pd

class Emotion:
    def __init__(self):
        self.vocab_size = 10000
        self.embedding_dim = 16
        self.max_length = 120
        self.trunc_type = 'post'
        self.oov_tok = "<OOV>"

        self.mapping = {'sadness': 0, 'anger': 1, 'love': 2, 'surprise': 3, 'fear': 4, 'joy': 5}
        self.df_train = pd.read_csv('data/train.txt', names=['Text', 'Emotion'], sep=';')
        self.train_sentences = self.df_train["Text"]
        self.tokenizer = Tokenizer(self.vocab_size, oov_token=self.oov_tok)
        self.tokenizer.fit_on_texts(self.train_sentences)

        self.mapping = {'sadness': 0, 'anger': 1, 'love': 2, 'surprise': 3, 'fear': 4, 'joy': 5}
        self.model = tf.keras.models.load_model('emotion_recognizer_model')

        self.emotion_mapping = {}
        for key in self.mapping:
            val = self.mapping[key]
            self.emotion_mapping[val] = key

    def sort_dict(self, myDict):
        myKeys = list(myDict.keys())
        myKeys.sort(reverse = True)
        sorted_dict = {i: myDict[i] for i in myKeys}
        return sorted_dict

    def predict_emotion(self, sentence):
        sequences = self.tokenizer.texts_to_sequences(sentence)
        padded = pad_sequences(sequences, padding=self.trunc_type,
                               maxlen=self.max_length)
        ans = list(self.model.predict(padded)[0])
        emotion = {}
        for index, val in enumerate(ans):
            emotion[ans[index]] = self.emotion_mapping[index]

        emotion = self.sort_dict(emotion)
        print(emotion)
        return emotion



