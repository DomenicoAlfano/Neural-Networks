from abc import ABCMeta, abstractmethod
from keras.models import Sequential
from keras.models import Model
import traceback
import os
import sys


class AbstractPOSTaggerTrainer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def load_resources(self):
        pass

    @abstractmethod
    def train(self, training_path):
        """
        Train the keras model from the training data.

        :param training_path: the path to training file
        :return: the keras model
        """
        pass


class ModelIO:
    @staticmethod
    def save(model, output_path):
        """
        Save the model to in the file pointed by the output_path variable

        :param model: the trained model
        :param output_path: the path to the file on which the model have to be saved
        :return: no return value is required
        """
        model.save(output_path)

    @staticmethod
    def load(model_file_path):
        """
        Load a sequential model saved in the file pointed by model_file_path

        :parah model_file_path: the path to the file that has to be loaded
        :return: a model loaded from the file
        """
        import keras
        return keras.models.load_model(model_file_path)


class AbstractPOSTaggerTester:
    __metaclass__ = ABCMeta

    @abstractmethod
    def load_resources(self):
        pass

    @abstractmethod
    def test(self, lstm_pos_tagger, test_file_path):
        """
        Test the lstm_pos_tagger against the gold standard.

        :param lst_pos_tagger: an istance of AbstractLSTMPOSTagger that has to be tested.
        :param test_file_path: a path to the gold standard file.

        :return: 'f1' score.

        Additional info:
        - Precision has to be computed as the number of correctly predicted 
          pos tag over the number of predicted pos tags.
        - Recall has to be computed as the number of correctly predicted 
          pos tag over the number of items in the gold standard
        - F1 has to be computed as the armonic mean between precision 
          and recall (2* P * R / (P + R)) 
        """
        pass


class AbstractLSTMPOSTagger:
    __metaclass__ = ABCMeta

    def __init__(self, model):
        self._model = model

    @abstractmethod
    def load_resources(self):
        pass

    def get_model(self):
        return self._model

    @abstractmethod
    def predict(self, sentence):
        """
        predict the pos tags for each token in the sentence.
        :param sentence: a list of tokens.
        :return: a list of pos tags (one for each input token).
        """
        pass


class Test:
    def __init__(self, training_path, model_path, gold_stanrdar_path):
        self._training_path = training_path
        self._model_path = model_path
        self._gold_standard_path = gold_stanrdar_path

    def test(self, lstm_trainer_implementation, lstm_tester_implementation, no_train=False):

        if no_train:
            print('No Train')
            model = ModelIO.load(self._model_path)
        else:
            print('Training..')
            lstm_trainer_implementation.load_resources()
            model = lstm_trainer_implementation.train(self._training_path)
            assert isinstance(model, Model)
            ModelIO.save(model, self._model_path)
            model = ModelIO.load(self._model_path)

        postagger = LSTMPOSTagger(model)
        postagger.load_resources()
        lstm_tester_implementation.load_resources()
        results = lstm_tester_implementation.test(postagger, self._gold_standard_path)

        return results


if __name__ == '__main__':
    """
    Main to run the test of the project.
    Use the parameter --no-train in order to skip the training of the model
    the program will not check if model_path already exist and will overwrite it if it is the case
    """
    if len(sys.argv) < 3:
        sys.exit(-1)
    model_index = 1
    project_dir_index = 2
    no_train = False
    if '--no-train' in sys.argv:
        model_index = 2
        project_dir_index = 3
        no_train = True

    # Check directory
    model_output_path = sys.argv[model_index]
    project_dir = sys.argv[project_dir_index]
    src_dir = project_dir + 'src/'
    if not os.path.exists(src_dir):
        raise IOError('src/ folder not found in ' + project_dir)
    data_dir = project_dir + 'data/'
    if not os.path.exists(data_dir):
        raise IOError('data/ folder not found in ' + project_dir)

    # dynamic import of modules
    sys.path.append(src_dir)
    from POSTaggerTester import POSTaggerTester
    from LSTMPOSTagger import LSTMPOSTagger
    from POSTaggerTrainer import POSTaggerTrainer

    # get files
    training_data = data_dir + 'en-train.conllu'
    test_data = data_dir + 'en-test.conllu'
    if not os.path.exists(training_data):
        raise IOError('en-train.conllu not found in ' + data_dir)
    if not os.path.exists(test_data):
        raise IOError('en-test.conllu not found in ' + data_dir)

    test = Test(training_data, model_output_path, test_data)
    trainer = POSTaggerTrainer()
    tester = POSTaggerTester()

    try:
        results = test.test(trainer, tester, no_train=no_train)
        print(results)
    except Exception as e:
        print("FAILED")
        raise traceback.print_exc(e)