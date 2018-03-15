from keras.layers import Dense, Activation, Embedding, LSTM, Bidirectional
from Project import AbstractPOSTaggerTrainer
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras import metrics
from conllu import parse
import numpy as np
import pickle
import re

class POSTaggerTrainer(AbstractPOSTaggerTrainer):

    def train(self, training_path):

        #Open_file_training
        data = open(training_path, 'r')

        data_parsed = parse(data.read())

        #extract words, sentences and tags
        words, temp_words, sentences, tags = [], [], [], []

        for i in range(len(data_parsed)):
            for j in range(len(data_parsed[i])):
                words.append(list(data_parsed[i][j].values())[1].lower())
                temp_words.append(list(data_parsed[i][j].values())[1].lower())
                tags.append(list(data_parsed[i][j].values())[3])
            sentences.append(temp_words)
            temp_words = []

        padding = 'padding_term'
        
        all_words = [padding] + words
        
        list_words = list(set(all_words))

        word_to_index = dict((word, idx) for idx, word in enumerate(list_words))

        tag_set = sorted(list(set(tags)))

        tags_to_index = dict((tag, idx) for idx, tag in enumerate(tag_set))

        #Save_word_dictionary
        pickle.dump(word_to_index, open('dict/word_to_index', 'wb'))

        #Input Preparation
        context_len = 5

        context_word_train = []

        for sentence in sentences:
            sent = [padding]*2 + sentence + [padding]*2
            for j in range(len(sent) - context_len + 1):
                context_word_train.append(sent[j: j + context_len])

        #Vectorization
        X_train = np.zeros((len(context_word_train),context_len))
        y_train = np.zeros((len(context_word_train),len(tag_set)))

        for i, feature in enumerate(context_word_train):
            for t, word in enumerate(feature):
                X_train[i,t] = word_to_index[word]
            y_train[i,tags_to_index[tags[i]]] = 1

        #Build model
        model = Sequential()
        model.add(Embedding(len(word_to_index), 300, input_length=context_len))
        model.add(Bidirectional(LSTM(32)))
        model.add(Dense(len(tag_set)))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train, y_train, batch_size=128, epochs=5)

        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig('output/history_accuracy.jpg')
        plt.clf()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('output/history_loss.jpg')
        plt.clf()

        return model