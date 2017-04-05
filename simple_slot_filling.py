import copy
import random
import re

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
EMBEDDING_DIMENSION = 128
MAX_SENTENCE_LENGTH = 50
CELL_SIZE = 128
BATCH_SIZE = 64
UNK = '<unk>'


def load_slot_vocab(path: str):
    vocab = {UNK: 0}
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) > 0:
                vocab[line] = len(vocab)

    return vocab


def load_word_vocab(path: str):
    with open(path, 'r') as file:
        vocab = eval(file.read())
        for word, value in vocab.items():
            vocab[word] = np.frombuffer(value, dtype=np.float32)

        unk = np.zeros([EMBEDDING_DIMENSION], dtype=np.float32)
        unk.fill(-1)
        vocab[UNK] = unk

    return vocab


def load_dataset(data_path: str, size: int = -1):
    IOB_REGEX = re.compile('\(([^)]+)\)\[([^\]]+)\]')

    data = []
    target = []
    with open(data_path, 'r') as file:
        for line in file:
            for match in IOB_REGEX.finditer(line):
                tokens = match.group(1).split(' ')
                iob = ' '.join(['{}/{}-{}'.format(tokens[i], (i == 0 and 'b' or 'i'), match.group(2))
                                for i in range(len(tokens))]).strip()
                line = line.replace(match.group(0), iob)

            words = []
            tags = []
            tokens = line.strip().lower().split(' ')
            for token in tokens:
                if '/' in token:
                    part = token.partition('/')
                    words.append(part[0])
                    tags.append(part[2])
                else:
                    words.append(token)
                    tags.append('o')

            data.append(words)
            target.append(tags)

    if size > -1:
        indices = random.sample([x for x in range(len(data))], size)
        data = [data[i] for i in indices]
        target = [target[i] for i in indices]

    return {
        'data': data,
        'target': target,
        'size': len(data)
    }


def rnn_model_fn(features, target, mode, params):
    num_classes = params['num_classes']
    learning_rate = params['learning_rate']
    dropout = mode == tf.contrib.learn.ModeKeys.TRAIN and 0.5 or 1.0

    cell = tf.contrib.rnn.GRUCell(CELL_SIZE)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)

    weight = tf.Variable(tf.truncated_normal([CELL_SIZE, num_classes], stddev=0.01))
    bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

    inputs = features['labeled_inputs']
    length = features['labeled_sequence_length']
    mask = features['mask']

    outputs, state = tf.nn.dynamic_rnn(
        cell=cell,
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32
    )

    output = tf.reshape(outputs, [-1, CELL_SIZE])
    softmax = tf.nn.softmax(tf.matmul(output, weight) + bias)
    prediction = tf.reshape(softmax, [-1, MAX_SENTENCE_LENGTH, num_classes])

    loss = None
    if mode != tf.contrib.learn.ModeKeys.INFER:
        target = tf.one_hot(target, num_classes)
        cross_entropy = -tf.reduce_sum(target * tf.log(prediction), reduction_indices=2) * mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1) / tf.cast(length, tf.float32)
        loss = tf.reduce_mean(cross_entropy)

    prediction = tf.argmax(prediction, 2)

    train_op = None
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=learning_rate,
            optimizer='RMSProp'
        )

    eval_metric_ops = None
    if mode != tf.contrib.learn.ModeKeys.INFER:
        target = tf.argmax(target, 2)
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                labels=target,
                predictions=prediction,
                weights=mask
            ),
            'predictions': prediction,
            'target': target,
            'length': length
        }

    return tf.contrib.learn.ModelFnOps(
        mode=mode,
        predictions={
            'predictions': prediction,
        },
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )


def input_fn(slot_vocab: dict, word_vocab: dict, dataset: dict, size: int = BATCH_SIZE):
    input_dict = {
        'labeled_inputs': [],
        'labeled_sequence_length': [],
        'mask': []
    }
    labels = []

    if size == dataset['size']:
        indices = [x for x in range(dataset['size'])]
    else:
        indices = random.sample([x for x in range(dataset['size'])], size)

    for i in indices:
        words = copy.deepcopy(dataset['data'][i])
        tags = copy.deepcopy(dataset['target'][i])
        if len(words) is not len(tags):
            raise ValueError('length is not same. ({})'.format(words))
        if len(words) > MAX_SENTENCE_LENGTH:
            raise OverflowError('length is too long. ({})'.format(len(words)))
        length = len(words)

        mask = []
        for j in range(length):
            mask.append(1.0)

        for j in range(MAX_SENTENCE_LENGTH - length):
            words.append(UNK)
            tags.append(UNK)
            mask.append(0.0)

        for j in range(MAX_SENTENCE_LENGTH):
            if tags[j] not in slot_vocab:
                raise ValueError('{} is not included in slot vocab'.format(tags[j]))
            else:
                tags[j] = slot_vocab[tags[j]]

        for j in range(MAX_SENTENCE_LENGTH):
            if words[j] not in word_vocab:
                words[j] = word_vocab[UNK]
            else:
                words[j] = word_vocab[words[j]]

        input_dict['labeled_inputs'].append(words)
        input_dict['labeled_sequence_length'].append(length)
        input_dict['mask'].append(mask)
        labels.append(tags)

    with open('./label.tmp', 'w') as f:
        for label in labels:
            f.write('{}\n'.format(label))

    input_dict['labeled_inputs'] = tf.constant(np.array(input_dict['labeled_inputs']))
    input_dict['labeled_sequence_length'] = tf.constant(input_dict['labeled_sequence_length'])
    input_dict['mask'] = tf.constant(input_dict['mask'])
    labels = tf.constant(labels)

    return input_dict, labels


def main(unused_argv):
    slot_vocab = load_slot_vocab('./atis.slot')
    word_vocab = load_word_vocab('./word2vec.embeddings')

    training_set = load_dataset('./atis.train')
    validation_set = load_dataset('./atis.dev')
    test_set = load_dataset('./atis.test')

    print('# training_set (%d)' % training_set['size'])
    print('# validation_set (%d)' % validation_set['size'])
    print('# test_set (%d)' % test_set['size'])

    classifier = tf.contrib.learn.Estimator(
        model_fn=rnn_model_fn,
        params={
            'num_classes': len(slot_vocab),
            'learning_rate': 0.01
        },
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=300),
        model_dir='./model/'
    )

    monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=lambda: input_fn(slot_vocab, word_vocab, validation_set, validation_set['size']),
        eval_steps=1,
        every_n_steps=10,
    )

    classifier.fit(
        input_fn=lambda: input_fn(slot_vocab, word_vocab, training_set),
        monitors=[monitor],
        steps=1000
    )

    accuracy_score = classifier.evaluate(
        input_fn=lambda: input_fn(slot_vocab, word_vocab, test_set, test_set['size']),
        steps=1
    )["accuracy"]
    print('Accuracy: {0:f}'.format(accuracy_score))

    result = classifier.evaluate(
        input_fn=lambda: input_fn(slot_vocab, word_vocab, test_set, test_set['size']),
        steps=1
    )
    print('Accuracy: {0:f}'.format(result["accuracy"]))

    # total = 0
    # count = 0
    # for prediction, target, length in zip(result['predictions'].tolist(), result['target'].tolist(), result['length'].tolist()):
    #     count += length
    #     for p, t in zip(prediction[:length], target[:length]):
    #         if p == t:
    #             total += 1
    # print('Accuracy: {0:f}({1}/{2})'.format(total/count, total, count))


if __name__ == "__main__":
    tf.app.run()
