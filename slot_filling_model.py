import random
import re

import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.dynamic_rnn_estimator import RNNKeys
from tensorflow.contrib.learn.python.learn.preprocessing import VocabularyProcessor

tf.logging.set_verbosity(tf.logging.INFO)
MODEL_DIR = './model'


def dummy_tokenizer(iterator):
    yield iterator


class SlotFillingModel:
    BATCH_SIZE = 64
    STEP_SIZE = 1000
    CELL_SIZE = 128
    MAX_SENTENCE_LENGTH = 100

    DOMAIN_REGEX = re.compile('\[([^\]]+)\] (.+)')
    IOB_REGEX = re.compile('\(([^)]+)\)\[([\w]+)\]')

    def __init__(self, slots: list):
        if not os.path.exists(MODEL_DIR):
            os.mkdir(MODEL_DIR)

        sp = os.path.join(MODEL_DIR, './slot_processor.saved')
        if os.path.exists(sp):
            print('Slot processor is restored.')
            self.__slot_processor = VocabularyProcessor.restore(sp)
        else:
            self.__slot_processor = VocabularyProcessor(SlotFillingModel.MAX_SENTENCE_LENGTH,
                                                        tokenizer_fn=dummy_tokenizer)
            self.__slot_processor.fit(slots)
            self.__slot_processor.save(sp)

        wp = os.path.join(MODEL_DIR, './word_processor.saved')
        if os.path.exists(wp):
            print('Word processor is restored.')
            self.__word_processor = VocabularyProcessor.restore(wp)
        else:
            self.__word_processor = VocabularyProcessor(SlotFillingModel.MAX_SENTENCE_LENGTH,
                                                        tokenizer_fn=dummy_tokenizer)
            with open('./word2vec.embeddings', 'r') as file:
                vocab = eval(file.read())
                self.__word_processor.fit(list(vocab.keys()))
                self.__word_processor.save(wp)

        self.__classifier = tf.contrib.learn.Estimator(
            model_fn=self.rnn_model_fn,
            model_dir=MODEL_DIR)

    def rnn_model_fn(self, features, target, mode, params):
        size = len(self.__slot_processor.vocabulary_)
        inputs = features['inputs']
        length = features[RNNKeys.SEQUENCE_LENGTH_KEY]
        dropout = mode == tf.contrib.learn.ModeKeys.TRAIN and 0.5 or 1.0

        cell = tf.contrib.rnn.GRUCell(SlotFillingModel.CELL_SIZE)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
        outputs, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=inputs,
            sequence_length=length,
            dtype=tf.float32
        )

        weight = tf.Variable(tf.truncated_normal([SlotFillingModel.CELL_SIZE, size], stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[size]))

        output = tf.reshape(outputs, [-1, SlotFillingModel.CELL_SIZE])
        softmax = tf.nn.softmax(tf.matmul(output, weight) + bias)
        prediction = tf.reshape(softmax, [-1, SlotFillingModel.MAX_SENTENCE_LENGTH, size])

        loss = None
        if mode != tf.contrib.learn.ModeKeys.INFER:
            target = tf.one_hot(target, size)
            mask = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
            cross_entropy = -tf.reduce_sum(target * tf.log(prediction), reduction_indices=2) * mask
            cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1) / tf.cast(length, tf.float32)
            loss = tf.reduce_mean(cross_entropy)

        train_op = None
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.01,
                optimizer='Adam'
            )

        target = tf.argmax(target, 2)
        prediction = tf.argmax(prediction, 2)

        eval_metric_ops = None
        if mode != tf.contrib.learn.ModeKeys.INFER:
            eval_metric_ops = {
                "accuracy": tf.metrics.accuracy(
                    labels=target,
                    predictions=prediction
                )
            }

        return tf.contrib.learn.ModelFnOps(
            mode=mode,
            predictions={
                "predictions": prediction
            },
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops
        )

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
            'raw': raw,
            'data': data,
            'target': target,
            'length': length,
            'size': len(data)
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
        input_dict = {}

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
            steps=SlotFillingModel.STEP_SIZE
        )

        accuracy_score = self.__classifier.evaluate(
            input_fn=lambda: self.input_fn(self.load_data(dev_data), len(dev_data)),
            steps=1
        )['accuracy']
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
