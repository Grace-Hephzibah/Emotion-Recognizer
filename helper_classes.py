import re
import nltk
import operator
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class textPreprocessPipeline:
    def install_nltk(self):
        nltk.download('averaged_perceptron_tagger')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download("stopwords")

    def lemmatization(self, sentence):
        lemmatizer= WordNetLemmatizer()        
        lemma = []
        pos_tag(word_tokenize(sentence))
        for word, tag in pos_tag(word_tokenize(sentence)):
            wntag = tag[0].lower()
            wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
            if not wntag:
                lemma.append(word)
            else:
                lemma.append(lemmatizer.lemmatize(word, wntag))
        return " ".join(lemma)
    
    def remove_stop_words(self, text):
        self.stop_words = set(stopwords.words("english"))
        Text=[i for i in str(text).split() if i not in self.stop_words]
        return " ".join(Text)

    def Removing_numbers(self, text):
        text=''.join([i for i in text if not i.isdigit()])
        return text

    def lower_case(self, text):
        text = text.split()
        text=[y.lower() for y in text]
        return " " .join(text)

    def Removing_punctuations(self, text):
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛',"", )

        ## remove extra whitespace
        text = re.sub('\s+', ' ', text)
        text =  " ".join(text.split())
        return text.strip()

    def Removing_urls(self, text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)


    def normalized_sentence(self, sentence):
        sentence= self.lower_case(sentence)
#         print("Lower case : ", sentence)

        sentence= self.remove_stop_words(sentence)
#         print("Stop Word : ", sentence)

        sentence= self.Removing_numbers(sentence)
#         print("Numbers : ", sentence)

        sentence= self.Removing_punctuations(sentence)
#         print("Punctuation : ", sentence)

        sentence= self.Removing_urls(sentence)
#         print("URL : ", sentence)

        sentence= self.lemmatization(sentence)
#         print("Lemma : ", sentence)
        return sentence
    

class Emotion(textPreprocessPipeline):
    def __init__(self):
        # Install needed nltk packages 
        self.install_nltk()

        # Importing datasets
        self.df_train = pd.read_csv('data/train.txt', names=['Text', 'Emotion'], sep=';')
        self.train_sentences = self.df_train["Text"]
        self.df_test = pd.read_csv('data/test.txt', names = ['Text', 'Emotion'], sep=';')
        self.test_sentences = self.df_test["Text"]

        # Label Encoder 
        self.le = LabelEncoder()
        self.le.fit_transform(self.df_train['Emotion'])
        
        # Creating the tokenizer
        self.tokenizer = Tokenizer(oov_token='UNK')
        self.tokenizer.fit_on_texts(pd.concat([self.train_sentences, self.test_sentences], axis=0))

        # Loading model
        adam = Adam(learning_rate=0.005)
        self.model = tf.keras.models.load_model('new_model_tf_10', compile = False)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    
    def predict_emotion(self, sentence):
        sentence = self.normalized_sentence(sentence)
        sentence = self.tokenizer.texts_to_sequences([sentence])
        sentence = pad_sequences(sentence, maxlen=229, truncating='pre')

        proba =  self.model.predict(sentence)[0]
        ans = {}
        for i in range(0, 6):
            ans[self.le.inverse_transform([i])[0]] = proba[i]

        return self.sort_dict(ans)
        
    def sort_dict(self, dic):
        keys = list(dic.keys())
        values = list(dic.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index[::-1]}
        return sorted_dict
            