import os

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
    'train': './data/atis.pkl.train',
    'dev': './data/atis.pkl.dev',
    'test': './data/atis.pkl.test',
    'slot': './data/atis.pkl.slot',
}

if __name__ == '__main__':

    config = config_plain

    # for vocab size
    DataSet('./data/atis.pkl.slot', './data/atis.pkl.train')
    DataSet('./data/atis.pos.slot', './data/atis.pos.train')

    slot = common['slot']
    training_set = DataSet(slot, common['train'])
    validation_set = DataSet(slot, common['dev'])
    test_set = DataSet(slot, common['test'])

    print('# training_set (%d)' % training_set.size())
    print('# validation_set (%d)' % validation_set.size())
    print('# test_set (%d)' % test_set.size())

    pos_model = None
    if 'pos_model' in config:
        pos_set = DataSet('./data/atis.pos.slot', './data/atis.pos.train')
        print('# POS pre-training set (%d)' % pos_set.size())

        pos_model = PosTagging.run(
            training_set=pos_set,
            gpu_memory=config['gpu_memory'],
            random_seed=RANDOM_SEED,
            vocab_size=DataSet.vocab_size(),
            drop_out=config['drop_out'],
            cell_size=CELL_SIZE,
            embedding_dimension=EMBEDDING_DIMENSION,
            learning_rate=LEARNING_RATE
        )

    result = SlotFilling.run(
        training_set=training_set,
        dev_set=validation_set,
        test_set=test_set,
        gpu_memory=config['gpu_memory'],
        random_seed=RANDOM_SEED,
        vocab_size=DataSet.vocab_size(),
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
