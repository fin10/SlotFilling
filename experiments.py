import datetime
import json
import os
import random

import tensorflow as tf

from slot_filling import SlotFilling

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.logging.set_verbosity(tf.logging.INFO)

RANDOM_SEED = 10
EMBEDDING_DIMENSION = 100
CELL_SIZE = 100
LEARNING_RATE = 0.001

random.seed(RANDOM_SEED)

config_plain = {
    'name': 'plain',
    'drop_out': 0.2,
    'gpu_memory': 0.5,
    'embedding_mode': None,
    'one_hot': False,
}

config_plain_ne = {
    'name': 'plain_ne',
    'drop_out': 0.2,
    'gpu_memory': 0.5,
    'embedding_mode': 'ne',
    'one_hot': False,
}

config_plain_pos = {
    'name': 'plain_pos',
    'drop_out': 0.2,
    'gpu_memory': 0.5,
    'embedding_mode': 'pos',
    'one_hot': False,
}

config_plain_ne_pos = {
    'name': 'plain_ne_pos',
    'drop_out': 0.2,
    'gpu_memory': 0.5,
    'embedding_mode': 'ne_pos',
    'one_hot': False,
}

config_plain_ne_oh = {
    'name': 'plain_ne_oh',
    'drop_out': 0.2,
    'gpu_memory': 0.5,
    'embedding_mode': 'ne',
    'one_hot': True,
}

config_plain_pos_oh = {
    'name': 'plain_pos_oh',
    'drop_out': 0.2,
    'gpu_memory': 0.5,
    'embedding_mode': 'pos',
    'one_hot': True,
}

config = config_plain

common = {
    'pkl': './data/atis.pkl',
    'train': './data/atis_pos.pkl.train.json',
    'test': './data/atis_pos.pkl.test.json',
    'dict': './data/atis_pos.pkl.dict.json',
}

if __name__ == '__main__':

    def convert(dataset):
        return ([dataset[index]['words'] for index in range(len(dataset))],
                [dataset[index]['entities'] for index in range(len(dataset))],
                [dataset[index]['labels'] for index in range(len(dataset))],
                [dataset[index]['tags'] for index in range(len(dataset))])


    with open(common['train']) as f:
        train = convert(json.load(f))

    with open(common['test']) as f:
        test = convert(json.load(f))

    with open(common['dict']) as f:
        dicts = json.load(f)

    def divide(data, ratio: float):
        size = len(data[0])

        indices = random.sample([x for x in range(size)], size)
        data0 = [data[0][index] for index in indices]
        data1 = [data[1][index] for index in indices]
        data2 = [data[2][index] for index in indices]
        data3 = [data[3][index] for index in indices]

        pivot = int(size * ratio)
        return (data0[:pivot], data1[:pivot], data2[:pivot], data3[:pivot]), (
            data0[pivot:], data1[pivot:], data2[pivot:], data3[pivot:])

    train, dev = divide(train, 0.8)
    num_slot = len(dicts['labels2idx'].keys())
    vocab_size = len(dicts['words2idx'].keys())
    entity_size = len(dicts['tables2idx'].keys())
    tag_size = len(dicts['tag2idx'].keys())

    print('# training_set (%d)' % len(train))
    print('# validation_set (%d)' % len(dev))
    print('# test_set (%d)' % len(test))
    print('# num_slot (%d)' % num_slot)
    print('# vocab_size (%d)' % vocab_size)
    print('# entity_size (%d)' % entity_size)
    print('# tag_size (%d)' % tag_size)

    start = datetime.datetime.now()

    result = SlotFilling.run(
        training_set=train,
        dev_set=dev,
        test_set=test,
        num_slot=num_slot,
        gpu_memory=config['gpu_memory'],
        random_seed=RANDOM_SEED,
        vocab_size=vocab_size,
        entity_size=entity_size,
        tag_size=tag_size,
        drop_out=config['drop_out'],
        cell_size=CELL_SIZE,
        embedding_dimension=EMBEDDING_DIMENSION,
        learning_rate=LEARNING_RATE,
        embedding_mode=config['embedding_mode'],
        one_hot=config['one_hot']
    )

    print('# Accuracy: {0:f}'.format(result['accuracy']))
    print('# F1 score: {0:f}'.format(result['f_measure']))

    diff = datetime.datetime.now() - start
    print('# %s' % diff)

    if not os.path.exists('./out'):
        os.mkdir('./out')

    with open(os.path.join('./out', '{}.csv'.format(config['name'])), mode='a') as output:
        output.write('# Accuracy: {0:f}\n'.format(result['accuracy']))
        output.write('# F1 score: {0:f}\n'.format(result['f_measure']))
