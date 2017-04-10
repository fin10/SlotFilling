import copy
import random
import re

import numpy as np


class DataSet:
    IOB_REGEX = re.compile('\(([^)]+)\)\[([^\]]+)\]')

    EMBEDDING_DIMENSION = 128
    MAX_SENTENCE_LENGTH = 100
    UNK = '<unk>'

    __word_vocab = None

    def __init__(self, slot_vocab, data_path: str = None):
        if DataSet.__word_vocab is None:
            raise ValueError('Need to initialize DataSet.')

        if type(slot_vocab) is str:
            self.__slot_vocab = DataSet.__load_slot_vocab(slot_vocab)
        elif type(slot_vocab) is dict:
            self.__slot_vocab = slot_vocab
        else:
            raise ValueError('slot_vocab error.')

        self.__epoch = 1
        self.__last_idx = 0

        if data_path is None:
            self.__inputs = []
            self.__lengths = []
            self.__masks = []
            self.__labels = []
            self.__size = 0
        else:
            data = []
            target = []
            with open(data_path, 'r') as file:
                for line in file:
                    for match in DataSet.IOB_REGEX.finditer(line):
                        tokens = match.group(1).split(' ')
                        iob = ' '.join(['{}/{}-{}'.format(tokens[i], (i == 0 and 'b' or 'i'), match.group(2))
                                        for i in range(len(tokens))]).strip()
                        line = line.replace(match.group(0), iob)

                    words = []
                    tags = []
                    tokens = line.strip().lower().split(' ')
                    for token in tokens:
                        if '/' in token:
                            part = token.partition('/')
                            words.append(part[0])
                            tags.append(part[2])
                        else:
                            words.append(token)
                            tags.append('o')

                    if len(words) > DataSet.MAX_SENTENCE_LENGTH:
                        raise OverflowError('size:%d, %s' % (len(words), line))

                    data.append(words)
                    target.append(tags)

            result = self.__parse_data(data, target)
            self.__inputs = result['inputs']
            self.__lengths = result['lengths']
            self.__masks = result['masks']
            self.__labels = result['labels']
            self.__size = len(data)

    def inputs(self):
        return self.__inputs

    def lengths(self):
        return self.__lengths

    def masks(self):
        return self.__masks

    def labels(self):
        return self.__labels

    def size(self):
        return self.__size

    def epoch(self):
        return self.__epoch

    def sample(self, size: int):
        if size == -1 or size == self.size():
            return self

        indices = random.sample([x for x in range(self.size())], size)
        new = DataSet(self.__slot_vocab)
        new.__inputs = [self.__inputs[index] for index in indices]
        new.__lengths = [self.__lengths[index] for index in indices]
        new.__masks = [self.__masks[index] for index in indices]
        new.__labels = [self.__labels[index] for index in indices]
        new.__size = size

        return new

    def get_batch(self, size: int):
        if size == -1 or size == self.size():
            self.__epoch += 1
            return self

        start = self.__last_idx
        self.__last_idx = start + size
        if self.__last_idx >= self.size():
            start = 0
            self.__last_idx = size
            self.__epoch += 1
            self.shuffle()

        new = DataSet(self.__slot_vocab)
        new.__inputs = self.__inputs[start:self.__last_idx]
        new.__lengths = self.__lengths[start:self.__last_idx]
        new.__masks = self.__masks[start:self.__last_idx]
        new.__labels = self.__labels[start:self.__last_idx]
        new.__size = size

        return new

    def shuffle(self):
        indices = random.shuffle([x for x in range(self.size())])
        self.__inputs = [self.__inputs[index] for index in indices]
        self.__lengths = [self.__lengths[index] for index in indices]
        self.__masks = [self.__masks[index] for index in indices]
        self.__labels = [self.__labels[index] for index in indices]

    @staticmethod
    def init():
        DataSet.__word_vocab = DataSet.__load_word_vocab('./word2vec.embeddings')

    @staticmethod
    def __load_slot_vocab(path: str):
        vocab = {
            DataSet.UNK: 0
        }

        with open(path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) > 0:
                    vocab[line] = len(vocab)

        return vocab

    @staticmethod
    def __load_word_vocab(path: str):
        with open(path, 'r') as file:
            vocab = eval(file.read())
            for word, value in vocab.items():
                vocab[word] = np.frombuffer(value, dtype=np.float32)

            unk = np.zeros([DataSet.EMBEDDING_DIMENSION], dtype=np.float32)
            unk.fill(-1)
            vocab[DataSet.UNK] = unk

        return vocab

    def __parse_data(self, data, target):
        inputs = []
        lengths = []
        masks = []
        labels = []

        for d, t in zip(data, target):
            words = copy.deepcopy(d)
            tags = copy.deepcopy(t)
            if len(words) is not len(tags):
                raise ValueError('length is not same. ({})'.format(words))
            if len(words) > DataSet.MAX_SENTENCE_LENGTH:
                raise OverflowError('length is too long. ({})'.format(len(words)))
            length = len(words)

            mask = []
            for j in range(length):
                mask.append(1.0)

            for j in range(DataSet.MAX_SENTENCE_LENGTH - length):
                words.append(DataSet.UNK)
                tags.append(DataSet.UNK)
                mask.append(0.0)

            for j in range(DataSet.MAX_SENTENCE_LENGTH):
                if tags[j] not in self.__slot_vocab:
                    raise ValueError('{} is not included in slot vocab'.format(tags[j]))
                else:
                    tags[j] = self.__slot_vocab[tags[j]]

            for j in range(DataSet.MAX_SENTENCE_LENGTH):
                if words[j] not in self.__word_vocab:
                    words[j] = self.__word_vocab[DataSet.UNK]
                else:
                    words[j] = self.__word_vocab[words[j]]

            inputs.append(words)
            lengths.append(length)
            masks.append(mask)
            labels.append(tags)

        return {
            'inputs': inputs,
            'lengths': lengths,
            'masks': masks,
            'labels': labels
        }

    def num_classes(self):
        return len(self.__slot_vocab)
