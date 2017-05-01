import os

import tensorflow as tf

from data_set import DataSet
from pos_tagging import PosTagging
from slot_filling_pos_tagging import SlotFilling

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.logging.set_verbosity(tf.logging.INFO)

POS_STEPS = 2000
RANDOM_SEED = 10
EMBEDDING_DIMENSION = 100
CELL_SIZE = 100
DROP_OUT = 0.5
LEARNING_RATE = 0.001

config_plain = {
    'name': 'plain',
    'pseudo_set': './data/pseudo_dummy.train',
}

config_pos = {
    'name': 'pos',
    'pos_model': POS_STEPS,
    'pseudo_set': './data/pseudo_dummy.train',
}

config_pseudo_label = {
    'name': 'pl',
    'pseudo_set': './data/atis.pkl.train',
}

config_pos_pseudo_label = {
    'name': 'pl_with_pos',
    'pos_model': POS_STEPS,
    'pseudo_set': './data/atis.pkl.train',
}

experiments = [{
    'train': './data/atis.pkl.train_1',
    'gpu_memory': 0.3,
}, {
    'train': './data/atis.pkl.train_2',
    'gpu_memory': 0.3,
}, {
    'train': './data/atis.pkl.train_3',
    'gpu_memory': 0.5,
}, {
    'train': './data/atis.pkl.train_4',
    'gpu_memory': 0.5,
}, {
    'train': './data/atis.pkl.train_5',
    'gpu_memory': 0.5,
}, {
    'train': './data/atis.pkl.train_6',
    'gpu_memory': 0.5,
}, {
    'train': './data/atis.pkl.train_7',
    'gpu_memory': 0.7,
}, {
    'train': './data/atis.pkl.train_8',
    'gpu_memory': 0.7,
}, {
    'train': './data/atis.pkl.train_9',
    'gpu_memory': 0.7,
}, {
    'train': './data/atis.pkl.train_10',
    'gpu_memory': 0.7,
}]

default = {
    'dev': './data/atis.pkl.dev',
    'test': './data/atis.pkl.test',
    'slot': './data/atis.pkl.slot',
}

if __name__ == '__main__':

    config = config_plain
    experiments = experiments[:1]

    print('# Experiments (%d)' % len(experiments))
    for experiment in experiments:
        experiment.update(default)

    if not os.path.exists('./out'):
        os.mkdir('./out')

    # for vocab size
    DataSet('./data/atis.pkl.slot', './data/atis.pkl.train')
    DataSet('./data/atis.pos.slot', './data/atis.pos.train')

    pos_model = None
    if 'pos_model' in config:
        pos_set = DataSet('./data/atis.pos.slot', './data/atis.pos.train')
        print('# Pre-training')
        print('# POS training set (%d)' % pos_set.size())

        pos_model = PosTagging.run(
            training_set=pos_set,
            steps=config['pos_model'],
            gpu_memory=0.2,
            random_seed=RANDOM_SEED,
            vocab_size=DataSet.vocab_size(),
            drop_out=DROP_OUT,
            cell_size=CELL_SIZE,
            embedding_dimension=EMBEDDING_DIMENSION,
            learning_rate=LEARNING_RATE
        )

    accuracies = []
    corrects = []
    no_matches = []
    mismatches = []
    over_matches = []

    for idx, experiment in enumerate(experiments):
        slot = experiment['slot']
        training_set = DataSet(slot, experiment['train'])
        validation_set = DataSet(slot, experiment['dev'])
        test_set = DataSet(slot, experiment['test'])
        pseudo_set = DataSet(slot, config_pos['pseudo_set'])

        print('# [%d] %s' % (idx, experiment['train']))
        print('# training_set (%d)' % training_set.size())
        print('# validation_set (%d)' % validation_set.size())
        print('# test_set (%d)' % test_set.size())

        result = SlotFilling.run(
            training_set=training_set,
            dev_set=validation_set,
            test_set=test_set,
            pseudo_set=pseudo_set,
            gpu_memory=experiment['gpu_memory'],
            random_seed=RANDOM_SEED,
            vocab_size=DataSet.vocab_size(),
            drop_out=DROP_OUT,
            cell_size=CELL_SIZE,
            embedding_dimension=EMBEDDING_DIMENSION,
            learning_rate=LEARNING_RATE,
            pos_model_dir=pos_model
        )
        print('# Accuracy: {0:f}\n'.format(result['accuracy']))

        accuracies.append(str(result['accuracy']))
        corrects.append(str(result['correct']))
        no_matches.append(str(result['no_match']))
        mismatches.append(str(result['mismatch']))
        over_matches.append(str(result['over_match']))

        with open(os.path.join('./out', '{}.csv'.format(config['name'])), mode='w') as output:
            output.write('accuracy,%s\n' % ','.join(accuracies))
            output.write('correct,%s\n' % ','.join(corrects))
            output.write('no_match,%s\n' % ','.join(no_matches))
            output.write('mismatch,%s\n' % ','.join(mismatches))
            output.write('over_match,%s\n' % ','.join(over_matches))
