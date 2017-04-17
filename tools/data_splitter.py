import argparse
import collections
import re

IOB_REGEX = re.compile('\(([^)]+)\)\[([^\]]+)\]')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('slot')
    parser.add_argument('input')
    parser.add_argument('ratio', type=float)
    args = parser.parse_args()

    slots = dict()
    with open(args.slot) as file:
        for line in file:
            line = line.strip()
            if len(line) > 0:
                slots[line] = 0

    data = []
    with open(args.input) as file:
        for line in file:
            line = line.strip()
            if len(line) > 0:
                data.append(line)
    print('# %s (%d)' % (args.input, len(data)))

    target = []
    for line in data:
        for match in IOB_REGEX.finditer(line):
            tokens = match.group(1).split(' ')
            iob = ' '.join(['{}/{}-{}'.format(tokens[i], (i == 0 and 'b' or 'i'), match.group(2))
                            for i in range(len(tokens))]).strip()
            line = line.replace(match.group(0), iob)

        tags = []
        tokens = line.strip().lower().split(' ')
        for token in tokens:
            if '/' in token:
                tags.append(token.partition('/')[2])
            else:
                tags.append('o')
        target.append(tags)

    for tags in target:
        for tag in tags:
            slots[tag] += 1

    for k, v in slots.items():
        slots[k] = int(v * args.ratio)

    slots = collections.OrderedDict(sorted(slots.items(), key=lambda x: x[0]))
    with open(args.input + '_' + str(args.ratio) + '.csv', 'w') as file:
        for k, v in slots.items():
            file.write('{},{}\n'.format(k, v))

    result = []
    for sentence, tags in zip(data, target):
        satisfied = True
        for tag in tags:
            slots[tag] -= 1
            if slots[tag] < 0:
                satisfied = False
                break

        if satisfied:
            result.append(sentence)

    print('# %d are generated.' % len(result))
    with open(args.input + '_' + str(args.ratio), 'w') as file:
        file.write('\n'.join(result))
