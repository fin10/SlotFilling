import os

import tensorflow as tf

from data_set import DataSet
from pos_tagging import PosTagging
from slot_filling_pos_tagging import SlotFilling

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.logging.set_verbosity(tf.logging.INFO)

POS_STEPS = 3000
RANDOM_SEED = 10
EMBEDDING_DIMENSION = 100
CELL_SIZE = 100
LEARNING_RATE = 0.001


config_plain = {
    'name': 'plain',
    'pseudo_set': './data/pseudo_dummy.train',
    'drop_out': 0.1,
    'pseudo_params': [0, 0, 0]
}

config_pos = {
    'name': 'pos',
    'pos_model': POS_STEPS,
    'pseudo_set': './data/pseudo_dummy.train',
    'drop_out': 0.2,
    'pseudo_params': [0, 0, 0]
}

config_pseudo_label = {
    'name': 'pl',
    'pseudo_set': './data/atis.pkl.train',
    'drop_out': 0.2,
    'pseudo_params': [150, 550, 1]
}

config_pos_pseudo_label = {
    'name': 'pl_with_pos',
    'pos_model': POS_STEPS,
    'pseudo_set': './data/atis.pkl.train',
    'drop_out': 0.2,
    'pseudo_params': [150, 550, 1]
}

experiments = [{
    'train': './data/atis.pkl.train_1',
    'gpu_memory': 0.5,
}, {
    'train': './data/atis.pkl.train_2',
    'gpu_memory': 0.5,
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
    'gpu_memory': 0.8,
}]

common = {
    'dev': './data/atis.pkl.dev',
    'test': './data/atis.pkl.test',
    'slot': './data/atis.pkl.slot',
}

if __name__ == '__main__':

    config = config_plain
    experiments = experiments[9:10]

    if not os.path.exists('./out'):
        os.mkdir('./out')

    # for vocab size
    DataSet('./data/atis.pkl.slot', './data/atis.pkl.train')
    DataSet('./data/atis.pos.slot', './data/atis.pos.train')

    slot = common['slot']
    validation_set = DataSet(slot, common['dev'])
    test_set = DataSet(slot, common['test'])

    print('# Experiments (%d)' % len(experiments))
    print('# validation_set (%d)' % validation_set.size())
    print('# test_set (%d)' % test_set.size())

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
            drop_out=config['drop_out'],
            cell_size=CELL_SIZE,
            embedding_dimension=EMBEDDING_DIMENSION,
            learning_rate=LEARNING_RATE
        )

    ev = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f-measure': [],
    }
    accuracies = []
    corrects = []
    no_matches = []
    mismatches = []
    over_matches = []

    for idx, experiment in enumerate(experiments):
        training_set = DataSet(slot, experiment['train'])
        pseudo_set = DataSet(slot, config['pseudo_set'])

        print('# [%d] %s' % (idx, experiment['train']))
        print('# training_set (%d)' % training_set.size())
        print('# pseudo_set (%d)' % pseudo_set.size())

        result = SlotFilling.run(
            training_set=training_set,
            dev_set=validation_set,
            test_set=test_set,
            pseudo_set=pseudo_set,
            gpu_memory=experiment['gpu_memory'],
            random_seed=RANDOM_SEED,
            vocab_size=DataSet.vocab_size(),
            drop_out=config['drop_out'],
            cell_size=CELL_SIZE,
            embedding_dimension=EMBEDDING_DIMENSION,
            learning_rate=LEARNING_RATE,
            pos_model_dir=pos_model,
            pseudo_params=config['pseudo_params']
        )
        print('# Accuracy: {0:f}'.format(result['accuracy']))
        print('# F-Measure: {0:f}\n'.format(result['ev_f_measure']))

        ev['accuracy'].append(str(result['ev_accuracy']))
        ev['precision'].append(str(result['ev_precision']))
        ev['recall'].append(str(result['ev_recall']))
        ev['f-measure'].append(str(result['ev_f_measure']))

        accuracies.append(str(result['accuracy']))
        corrects.append(str(result['correct']))
        no_matches.append(str(result['no_match']))
        mismatches.append(str(result['mismatch']))
        over_matches.append(str(result['over_match']))

        with open(os.path.join('./out', '{}.csv'.format(config['name'])), mode='w') as output:
            output.write('ev_accuracy,%s\n' % ','.join(ev['accuracy']))
            output.write('ev_precision,%s\n' % ','.join(ev['precision']))
            output.write('ev_recall,%s\n' % ','.join(ev['recall']))
            output.write('ev_f_measure,%s\n' % ','.join(ev['f-measure']))
            output.write('accuracy,%s\n' % ','.join(accuracies))
            output.write('correct,%s\n' % ','.join(corrects))
            output.write('no_match,%s\n' % ','.join(no_matches))
            output.write('mismatch,%s\n' % ','.join(mismatches))
            output.write('over_match,%s\n' % ','.join(over_matches))
