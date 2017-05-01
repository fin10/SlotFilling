import argparse
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()

    with open(args.input) as file:
        data = [line.strip() for line in file if len(line.strip())]

    size = len(data)

    for idx in range(1, 11):
        random.shuffle(data)
        subset = data[:int(size * 0.1 * idx)]
        with open('{}_{}'.format(args.input, idx), mode='w') as output:
            output.write('\n'.join(subset))
        print('#%d (%d)' % (idx, len(subset)))
