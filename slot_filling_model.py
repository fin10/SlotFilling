import random
import re

import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.dynamic_rnn_estimator import multi_value_rnn_classifier, RNNKeys
from tensorflow.contrib.learn.python.learn.preprocessing import VocabularyProcessor

tf.logging.set_verbosity(tf.logging.INFO)


def dummy_tokenizer(iterator):
    yield iterator


class SlotFillingModel:
    BATCH_SIZE = 64
    STEP_SIZE = 1000
    MAX_SENTENCE_LENGTH = 100

    DOMAIN_REGEX = re.compile('\[([^\]]+)\] (.+)')
    IOB_REGEX = re.compile('\(([^)]+)\)\[([\w]+)\]')

    def __init__(self, slots: list):
        self.__slot_processor = VocabularyProcessor(SlotFillingModel.MAX_SENTENCE_LENGTH,
                                                    tokenizer_fn=dummy_tokenizer)
        self.__slot_processor.fit(slots)

        if os.path.exists('./word_processor.saved'):
            print('Word processor is restored.')
            self.__word_processor = VocabularyProcessor.restore('./word_processor.saved')
        else:
            self.__word_processor = VocabularyProcessor(SlotFillingModel.MAX_SENTENCE_LENGTH,
                                                        tokenizer_fn=dummy_tokenizer)
            with open('./word2vec.embeddings', 'r') as file:
                vocab = eval(file.read())
                self.__word_processor.fit(list(vocab.keys()))
                self.__word_processor.save('./word_processor.saved')

        seq_feature_columns = [
            tf.contrib.layers.real_valued_column('inputs', dimension=SlotFillingModel.MAX_SENTENCE_LENGTH)]
        self.__classifier = multi_value_rnn_classifier(num_classes=len(self.__slot_processor.vocabulary_),
                                                       num_units=SlotFillingModel.MAX_SENTENCE_LENGTH,
                                                       sequence_feature_columns=seq_feature_columns,
                                                       cell_type='gru',
                                                       model_dir='./model')

    def load_data(self, items: list):
        raw = []
        data = []
        target = []
        length = []
        for item in items:
            match = SlotFillingModel.DOMAIN_REGEX.findall(item)[0]
            text = SlotFillingModel.convert(match[0].strip(), match[1].strip())

            utterance = []
            iob = []
            tokens = text.split(' ')
            for token in tokens:
                if '/' in token:
                    part = token.partition('/')
                    iob.append(part[2])
                    utterance.append(part[0].lower())
                else:
                    iob.append('o')
                    utterance.append(token.lower())

            raw.append(text)
            data += list(self.__word_processor.transform(utterance))
            target += list(self.__slot_processor.transform(iob))
            length.append(len(iob))

        data = np.array(data)
        target = np.array(target)

        return {
            'raw': raw, 'data': data, 'target': target, 'length': length, 'size': len(data)
        }

    @staticmethod
    def convert(domain: str, utterance: str):
        for match in SlotFillingModel.IOB_REGEX.finditer(utterance):
            tokens = match.group(1).split(' ')
            iob = ' '.join(['{}/{}-{}.{}'.format(tokens[i], (i == 0 and 'b' or 'i'), domain, match.group(2))
                            for i in range(len(tokens))]).strip()
            utterance = utterance.replace(match.group(0), iob)
        return utterance

    def input_fn(self, dataset: dict, size: int):
        input_dict = {
            # '{}_{}'.format(RNNKeys.STATE_PREFIX, i): tf.random_uniform([len(dataset[0]), MAX_SENTENCE_LENGTH])
            # for i in range(MAX_SENTENCE_LENGTH)
        }

        indices = random.sample([x for x in range(dataset['size'])], size)
        input_dict['inputs'] = tf.contrib.layers.embed_sequence(
            ids=np.array([dataset['data'][i] for i in indices]),
            vocab_size=len(self.__word_processor.vocabulary_),
            embed_dim=128)
        input_dict[RNNKeys.SEQUENCE_LENGTH_KEY] = tf.constant(np.array([dataset['length'][i] for i in indices]))
        labels = tf.constant(np.array([dataset['target'][i] for i in indices]))

        return input_dict, labels

    def train(self, train_data, dev_data):
        self.__classifier.fit(
            input_fn=lambda: self.input_fn(self.load_data(train_data), SlotFillingModel.BATCH_SIZE),
            steps=SlotFillingModel.STEP_SIZE)

        accuracy_score = self.__classifier.evaluate(
            input_fn=lambda: self.input_fn(self.load_data(dev_data), len(dev_data)),
            steps=1)['accuracy']
        print("Accuracy: {0:f}".format(accuracy_score))

    def test(self, test_data):
        dataset = self.load_data(test_data)
        result = self.__classifier.predict(
            input_fn=lambda: self.input_fn(dataset, dataset['size']))
        for i, p in enumerate(result):
            length = dataset['length'][i]
            print("Expected : %s" % dataset['raw'][i])
            print("Actual   : %s" % self.compile(dataset['data'][i][:length], p["predictions"][:length].tolist()))

    def compile(self, data: list, target: list):
        words = list(self.__word_processor.reverse([data]))[0].split(' ')
        slots = list(self.__slot_processor.reverse([target]))[0].split(' ')

        result = []
        for i in range(len(words)):
            if slots[i] == 'o':
                result.append(words[i])
            else:
                result.append(words[i] + '/' + slots[i])

        return ' '.join(result)


if __name__ == '__main__':
    pass
