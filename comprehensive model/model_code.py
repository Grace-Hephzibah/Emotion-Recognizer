from imports import *
from pipeline import textPreprocessPipeline

class Emotion(textPreprocessPipeline):
    def __init__(self):
        # Preparing nltk 
        # install_nltk()

        # Label Encoding for emotions
        self.mapping = {'anger' : 0, 'fear': 1, 'joy' : 2, 'love' : 3, 'sadness' : 4, 'surprise' : 5}

        # Importing datasets
        self.df_train = pd.read_csv('data/train.txt', names=['Text', 'Emotion'], sep=';')
        self.train_sentences = self.df_train["Text"]
        self.df_test = pd.read_csv('data/test.txt', names = ['Text', 'Emotion'], sep=';')
        self.test_sentences = self.df_test["Text"]

        # Creating the tokenizer
        self.tokenizer = Tokenizer(oov_token='UNK')
        self.tokenizer.fit_on_texts(pd.concat([self.train_sentences, self.test_sentences], axis=0))

        # Loading model
        self.model = tf.keras.models.load_model('lstm_glove')

        # Inverse mapping emotions 
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
        sentence = self.normalized_sentence(sentence)
        sentence = self.tokenizer.texts_to_sequences([sentence])
        sentence = pad_sequences(sentence, maxlen=229, truncating='pre')

        ans = list(self.model.predict(sentence)[0])
        return ans
        emotion = {}
        for index, val in enumerate(ans):
            emotion[ans[index]] = self.emotion_mapping[index]

        emotion = self.sort_dict(emotion)
        #print("Emotion : ", emotion)
        return emotion

# e = Emotion()
# print(e.predict_emotion("Happy!"))
