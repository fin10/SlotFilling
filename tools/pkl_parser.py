import argparse
import pickle
import random
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input a pkl file')
    args = parser.parse_args()

    with open(args.input, 'rb') as f:
        train, test, dicts = pickle.load(f, encoding='latin1')


    def invert(old: dict):
        inverted = dict()
        for item in old.items():
            inverted[item[1]] = item[0].lower()
        return inverted


    idx2words = invert(dicts['words2idx'])
    idx2labels = invert(dicts['labels2idx'])
    idx2tables = invert(dicts['tables2idx'])

    slots = set(idx2labels.values())
    print('# Slots (%d)' % len(slots))


    def parse(words_list, labels_list):
        result = []
        for words, labels in zip(words_list, labels_list):
            tokens = []
            for word, label in zip(words, labels):
                word = idx2words[word]
                label = idx2labels[label]

                if re.match('(digit)+', word):
                    word = '<num>'

                tokens.append('{}/{}'.format(word, label))
            result.append(' '.join(tokens))
        return result


    def divide(data: list, ratio: float):
        random.shuffle(data)
        pivot = int(len(data) * ratio)
        return data[:pivot], data[pivot:]


    train_data = parse(train[0], train[2])
    train_data, dev_data = divide(train_data, 0.8)
    test_data = parse(test[0], test[2])
    print('# Train (%d), Dev (%d), Test (%d) are generated.' % (len(train_data), len(dev_data), len(test_data)))

    with open(args.input + '.slot', 'w') as output:
        output.write('\n'.join(slots))

    with open(args.input + '.train', 'w') as output:
        output.write('\n'.join(train_data))

    with open(args.input + '.dev', 'w') as output:
        output.write('\n'.join(dev_data))

    with open(args.input + '.test', 'w') as output:
        output.write('\n'.join(test_data))
