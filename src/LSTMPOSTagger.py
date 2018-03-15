from Project import AbstractLSTMPOSTagger
from random import randint
import numpy as np
import pickle
import string


class LSTMPOSTagger(AbstractLSTMPOSTagger):

    def __init__(self, model):
        self._model = model

    def predict(self, sentence):

        #Load_word_dictionary
        word_to_index = pickle.load(open('dict/word_to_index', 'rb'))

        #Load_tag_dictionary
        index_to_tags = pickle.load(open('dict/index_to_tags', 'rb'))

        #Input Preparation
        context_len = 5

        context_word=[]

        padding = 'padding_term'

        sent = [padding]*2 + sentence + [padding]*2

        for i in range(len(sent) - context_len + 1):
            context_word.append(sent[i: i + context_len])
        
        #Vectorization 
        X_test = np.zeros((len(context_word),context_len))

        for i, feature in enumerate(context_word):
            for t, word in enumerate(feature):
                if word in word_to_index:
                    X_test[i,t] = word_to_index[word]
                else:
                    continue

        #Prediction
        y_pred = []
        for sequence in X_test:
            context = np.array([sequence])
            prediction = self._model.predict(context)
            y_pred.append(index_to_tags[np.argmax(prediction)])        
        return y_pred