import os
import pickle
import random

import tensorflow as tf

from data_set import DataSet
from pos_tagging import PosTagging
from slot_filling import SlotFilling

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.logging.set_verbosity(tf.logging.INFO)

RANDOM_SEED = 10
EMBEDDING_DIMENSION = 100
CELL_SIZE = 100
LEARNING_RATE = 0.001

config_plain = {
    'name': 'plain',
    'drop_out': 0.2,
    'gpu_memory': 0.8,
}

config_pos = {
    'name': 'pos',
    'drop_out': 0.3,
    'gpu_memory': 0.8,
    'pos_model': True
}

common = {
    'pkl': './data/atis.pkl'
}

if __name__ == '__main__':

    config = config_plain

    with open(common['pkl'], 'rb') as f:
        train, test, dicts = pickle.load(f, encoding='latin1')


    def divide(data: list, ratio: float):
        size = len(data[0])

        indices = random.sample([x for x in range(size)], size)
        data0 = [data[0][index] for index in indices]
        data1 = [data[1][index] for index in indices]
        data2 = [data[2][index] for index in indices]

        pivot = int(size * ratio)
        return (data0[:pivot], data1[:pivot], data2[:pivot]), (data0[pivot:], data1[pivot:], data2[pivot:])


    train, dev = divide(train, 0.8)
    num_slot = len(dicts['labels2idx'].keys())
    vocab_size = len(dicts['words2idx'].keys())

    print('# training_set (%d)' % len(train))
    print('# validation_set (%d)' % len(dev))
    print('# test_set (%d)' % len(test))
    print('# num_slot (%d)' % num_slot)
    print('# vocab_size (%d)' % vocab_size)

    pos_model = None
    if 'pos_model' in config:
        pos_set = DataSet('./data/atis.pos.slot', './data/atis.pos.train')
        print('# POS pre-training set (%d)' % pos_set.size())

        pos_model = PosTagging.run(
            training_set=pos_set,
            gpu_memory=config['gpu_memory'],
            random_seed=RANDOM_SEED,
            vocab_size=vocab_size,
            drop_out=config['drop_out'],
            cell_size=CELL_SIZE,
            embedding_dimension=EMBEDDING_DIMENSION,
            learning_rate=LEARNING_RATE
        )

    result = SlotFilling.run(
        training_set=train,
        dev_set=dev,
        test_set=test,
        num_slot=num_slot,
        gpu_memory=config['gpu_memory'],
        random_seed=RANDOM_SEED,
        vocab_size=vocab_size,
        drop_out=config['drop_out'],
        cell_size=CELL_SIZE,
        embedding_dimension=EMBEDDING_DIMENSION,
        learning_rate=LEARNING_RATE,
        pos_model_dir=pos_model,
    )

    print('# Accuracy: {0:f}'.format(result['accuracy']))
    print('# F1 score: {0:f}\n'.format(result['f_measure']))

    if not os.path.exists('./out'):
        os.mkdir('./out')

    with open(os.path.join('./out', '{}.csv'.format(config['name'])), mode='w') as output:
        output.write('# Accuracy: {0:f}\n'.format(result['accuracy']))
        output.write('# F1 score: {0:f}\n'.format(result['f_measure']))
