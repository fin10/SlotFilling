import os

from slot_filling import SlotFilling

experiments = [
    {
        'name': 'labeled_10p',
        'training': './data/atis.train_0.1',
        'unlabeled': None
    }, {
        'name': 'labeled_20p',
        'training': './data/atis.train_0.2',
        'unlabeled': None
        # }, {
        #     'name': 'labeled_30p',
        #     'training': './data/atis.train_0.3',
        #     'unlabeled': None
        # }, {
        #     'name': 'labeled_40p',
        #     'training': './data/atis.train_0.4',
        #     'unlabeled': None
        # }, {
        #     'name': 'labeled_50p',
        #     'training': './data/atis.train_0.5',
        #     'unlabeled': None
        # }, {
        #     'name': 'labeled_60p',
        #     'training': './data/atis.train_0.6',
        #     'unlabeled': None
        # }, {
        #     'name': 'labeled_70p',
        #     'training': './data/atis.train_0.7',
        #     'unlabeled': None
        # }, {
        #     'name': 'labeled_80p',
        #     'training': './data/atis.train_0.8',
        #     'unlabeled': None
        # }, {
        #     'name': 'labeled_90p',
        #     'training': './data/atis.train_0.9',
        #     'unlabeled': None
        # }, {
        #     'name': 'labeled_100p',
        #     'training': './data/atis.train_1.0',
        #     'unlabeled': None
    },
]

if __name__ == '__main__':
    if not os.path.exists('./out'):
        os.mkdir('./out')

    print('Experiments (%d)' % len(experiments))

    corrects = []
    no_matches = []
    mismatches = []
    over_matches = []
    for experiment in experiments:
        print('# %s' % experiment['name'])
        result = SlotFilling.run(
            slot='./data/atis.slot',
            train=experiment['training'],
            dev='./data/atis.dev',
            test='./data/atis.test',
            unlabeled_slot='./data/pos.slot',
            unlabeled_train=experiment['unlabeled'],
            steps=1
        )
        print('# Accuracy: {0:f}\n'.format(result['accuracy']))

        corrects.append(str(result['correct']))
        no_matches.append(str(result['no_match']))
        mismatches.append(str(result['mismatch']))
        over_matches.append(str(result['over_match']))

    with open(os.path.join('./out', 'labeled' + '.csv'), mode='w') as output:
        output.write('correct,%s\n' % ','.join(corrects))
        output.write('no_match,%s\n' % ','.join(no_matches))
        output.write('mismatch,%s\n' % ','.join(mismatches))
        output.write('over_match,%s\n' % ','.join(over_matches))
