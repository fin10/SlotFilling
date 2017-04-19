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
    }, {
        'name': 'labeled_30p',
        'training': './data/atis.train_0.3',
        'unlabeled': None
    }, {
        'name': 'labeled_40p',
        'training': './data/atis.train_0.4',
        'unlabeled': None
    }, {
        'name': 'labeled_50p',
        'training': './data/atis.train_0.5',
        'unlabeled': None
    }, {
        'name': 'labeled_60p',
        'training': './data/atis.train_0.6',
        'unlabeled': None
    }, {
        'name': 'labeled_70p',
        'training': './data/atis.train_0.7',
        'unlabeled': None
    }, {
        'name': 'labeled_80p',
        'training': './data/atis.train_0.8',
        'unlabeled': None
    }, {
        'name': 'labeled_90p',
        'training': './data/atis.train_0.9',
        'unlabeled': None
    }, {
        'name': 'labeled_100p',
        'training': './data/atis.train_1.0',
        'unlabeled': None
    }, {
        'name': 'unlabeled_10p',
        'training': './data/atis.train_0.1',
        'unlabeled': './data/pos.unlabeled'
    }, {
        'name': 'unlabeled_20p',
        'training': './data/atis.train_0.2',
        'unlabeled': './data/pos.unlabeled'
    }, {
        'name': 'unlabeled_30p',
        'training': './data/atis.train_0.3',
        'unlabeled': './data/pos.unlabeled'
    }, {
        'name': 'unlabeled_40p',
        'training': './data/atis.train_0.4',
        'unlabeled': './data/pos.unlabeled'
    }, {
        'name': 'unlabeled_50p',
        'training': './data/atis.train_0.5',
        'unlabeled': './data/pos.unlabeled'
    }, {
        'name': 'unlabeled_60p',
        'training': './data/atis.train_0.6',
        'unlabeled': './data/pos.unlabeled'
    }, {
        'name': 'unlabeled_70p',
        'training': './data/atis.train_0.7',
        'unlabeled': './data/pos.unlabeled'
    }, {
        'name': 'unlabeled_80p',
        'training': './data/atis.train_0.8',
        'unlabeled': './data/pos.unlabeled'
    }, {
        'name': 'unlabeled_90p',
        'training': './data/atis.train_0.9',
        'unlabeled': './data/pos.unlabeled'
    }, {
        'name': 'unlabeled_100p',
        'training': './data/atis.train_1.0',
        'unlabeled': './data/pos.unlabeled'
    },
]

if __name__ == '__main__':
    if not os.path.exists('./out'):
        os.mkdir('./out')

    print('Experiments (%d)' % len(experiments))
    for experiment in experiments:
        print('# %s' % experiment['name'])
        result = SlotFilling.run(
            slot='./data/atis.slot',
            train=experiment['training'],
            dev='./data/atis.dev',
            test='./data/atis.test',
            unlabeled_slot='./data/pos.slot',
            unlabeled_train=experiment['unlabeled'],
            steps=1000
        )
        print('# Accuracy: {0:f}'.format(result['accuracy']))

        with open(os.path.join('./out', experiment['name'] + '.csv'), mode='w') as output:
            for k, v1, v2, v3 in zip(result['correct'].keys(), result['correct'].values(), result['no_match'].values(),
                                     result['mismatch'].values()):
                output.write('{},{},{},{}\n'.format(k, v1, v2, v3))
