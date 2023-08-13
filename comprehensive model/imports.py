import re
import nltk

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer


def install_nltk():
    nltk.download("stopwords")
    nltk.download("SnowballStemmer")
    nltk.download("WordNetLemmatizer")
    nltk.download('wordnet')
    