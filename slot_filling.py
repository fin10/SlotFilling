import argparse
import os

from slot_filling_model import SlotFillingModel


def read_file(path: str):
    with open(path, 'r', encoding='utf-8') as lines:
        result = [x.strip() for x in lines]

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')
    parser.add_argument('--data')
    args = parser.parse_args()

    if args.mode == 'train':
        slots = []
        train = []
        dev = []
        for file in os.listdir(args.data):
            if file.endswith('slot'):
                slots += read_file(os.path.join(args.data, file))
            elif file.endswith('train'):
                train += read_file(os.path.join(args.data, file))
            elif file.endswith('dev'):
                dev += read_file(os.path.join(args.data, file))

        print('Slot (%d), Train (%d), Dev (%d)' % (len(slots), len(train), len(dev)))

        model = SlotFillingModel(slots)
        model.train(train, dev)
    elif args.mode == 'test':
        slots = []
        test = []
        for file in os.listdir(args.data):
            if file.endswith('slot'):
                slots += read_file(os.path.join(args.data, file))
            elif file.endswith('test'):
                test += read_file(os.path.join(args.data, file))

        print('Slot (%d), Test (%d)' % (len(slots), len(test)))

        model = SlotFillingModel(slots)
        model.test(test)
