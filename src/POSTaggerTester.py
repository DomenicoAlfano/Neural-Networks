from Project import AbstractPOSTaggerTester
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from conllu import parse
import numpy as np
import itertools
import pickle
import re

class POSTaggerTester(AbstractPOSTaggerTester):

    def test(self, postagger, test_file_path):

        #Open_file
        data = open(test_file_path, 'r')

        data_parsed = parse(data.read())

        #extract_word, sentences and tags
        temp_words, sentences, tags = [], [], []

        for i in range(len(data_parsed)):
            for j in range(len(data_parsed[i])):
                temp_words.append(list(data_parsed[i][j].values())[1].lower())
                tags.append(list(data_parsed[i][j].values())[3])
            sentences.append(temp_words)
            temp_words = []

        tag_set = sorted(list(set(tags)))
        index_to_tags = dict((idx, tag) for idx, tag in enumerate(tag_set))

        #Save_tag_dictionary
        pickle.dump(index_to_tags, open('dict/index_to_tags', 'wb'))

        y_test=tags

        #Prediction
        y_pred = []
        for sentence in sentences:
            y_pred = y_pred + postagger.predict(sentence)

        #Save pos_tagged_sentences
        output_sentence = open('output/pos_tagged_sentences.txt', 'w')
        i = 0
        for s, sentence in enumerate(sentences):
            output_sentence.write("%s\n" % ' '.join(sentence))
            output_sentence.write("%s\n" % ' '.join(y_pred[i:i + len(sentence)]))
            i = i + len(sentence)

        #Compute Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=tag_set)

        # Normalization
        cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  
        #Plot Confusion matrix
        plt.figure(figsize = (13,9))
        plt.imshow(cm_n, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(tag_set))
        plt.xticks(tick_marks, tag_set, rotation=45)
        plt.yticks(tick_marks, tag_set)

        fmt = '.2f'
        thresh = cm_n.max() / 2.
        
        for i, j in itertools.product(range(cm_n.shape[0]), range(cm_n.shape[1])):
            plt.text(j, i, format(cm_n[i, j], fmt),horizontalalignment="center",color="white" if cm_n[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('output/confusion_matrix.jpg')
        plt.clf()

        #Score
        sum_precision = sum(np.sum(cm, axis=0))
        sum_recall = sum(np.sum(cm, axis=1))

        precision = np.trace(cm) / sum_precision
        recall = np.trace(cm) / sum_recall
        f1 = (2 * precision * recall) / (precision + recall)
        score = {'f1': f1}

        return score