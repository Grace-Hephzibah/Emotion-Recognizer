from imports import *

class textPreprocessPipeline:
    def lemmatization(self, text):
        lemmatizer= WordNetLemmatizer()
        text = text.split()
        text=[lemmatizer.lemmatize(y) for y in text]
        return " " .join(text)
    
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
        sentence= self.remove_stop_words(sentence)
        sentence= self.Removing_numbers(sentence)
        sentence= self.Removing_punctuations(sentence)
        sentence= self.Removing_urls(sentence)
        sentence= self.lemmatization(sentence)
        return sentence
    
    