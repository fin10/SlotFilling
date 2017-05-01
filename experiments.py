import os
import time

from slot_filling_pos_tagging import SlotFilling

experiments = [{
    'labeled_train': './data/atis.train_0.1',
    'gpu_memory': 0.3,
}, {
    'labeled_train': './data/atis.train_0.2',
    'gpu_memory': 0.3,
}, {
    'labeled_train': './data/atis.train_0.3',
    'gpu_memory': 0.5,
}, {
    'labeled_train': './data/atis.train_0.4',
    'gpu_memory': 0.5,
}, {
    'labeled_train': './data/atis.train_0.5',
    'gpu_memory': 0.5,
}, {
    'labeled_train': './data/atis.train_0.6',
    'gpu_memory': 0.5,
}, {
    'labeled_train': './data/atis.train_0.7',
    'gpu_memory': 0.7,
}, {
    'labeled_train': './data/atis.train_0.8',
    'gpu_memory': 0.7,
}, {
    'labeled_train': './data/atis.train_0.9',
    'gpu_memory': 0.7,
}, {
    'labeled_train': './data/atis.train_1.0',
    'gpu_memory': 0.7,
}]

if __name__ == '__main__':
    default = {
        'name': 'unlabeled',
        'dev': './data/atis.dev',
        'test': './data/atis.test',
        'labeled_slot': './data/atis.slot',
        'unlabeled_slot': './data/atis_pos.slot',
        'unlabeled_train': './data/atis_pos.train',
        # 'unlabeled_train': None,
    }

    print('Experiments (%d)' % len(experiments))
    accuracies = []
    corrects = []
    no_matches = []
    mismatches = []
    over_matches = []

    for experiment in experiments:
        experiment.update(default)

    if not os.path.exists('./out'):
        os.mkdir('./out')

    for idx, experiment in enumerate(experiments):
        print('# [%d] %s' % (idx, experiment['labeled_train']))
        result = SlotFilling.run(
            dev=experiment['dev'],
            test=experiment['test'],
            labeled_slot=experiment['labeled_slot'],
            labeled_train=experiment['labeled_train'],
            unlabeled_slot=experiment['unlabeled_slot'],
            unlabeled_train=experiment['unlabeled_train'],
            gpu_memory=experiment['gpu_memory']
        )
        print('# Accuracy: {0:f}\n'.format(result['accuracy']))

        accuracies.append(str(result['accuracy']))
        corrects.append(str(result['correct']))
        no_matches.append(str(result['no_match']))
        mismatches.append(str(result['mismatch']))
        over_matches.append(str(result['over_match']))

        with open(os.path.join('./out', '{}-{}.csv'.format(experiment['name'], time.strftime('%H-%M-%S'))),
                  mode='w') as output:
            output.write('accuracy,%s\n' % ','.join(accuracies))
            output.write('correct,%s\n' % ','.join(corrects))
            output.write('no_match,%s\n' % ','.join(no_matches))
            output.write('mismatch,%s\n' % ','.join(mismatches))
            output.write('over_match,%s\n' % ','.join(over_matches))

    with open(os.path.join('./out', '{}_{}.csv'.format(default['name'], 'total')),
              mode='w') as output:
        output.write('accuracy,%s\n' % ','.join(accuracies))
        output.write('correct,%s\n' % ','.join(corrects))
        output.write('no_match,%s\n' % ','.join(no_matches))
        output.write('mismatch,%s\n' % ','.join(mismatches))
        output.write('over_match,%s\n' % ','.join(over_matches))
