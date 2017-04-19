import argparse
import os
import re
from xml.etree.ElementTree import parse

SPECIAL_LETTER = re.compile('[?!.,\-\"/]')
BRACKET = re.compile('\[[^]]+\]|\([^)]+\)')


def normalize(text: str):
    if 'unintelligible' in text \
            or 'uninteligible' in text \
            or 'unintelligable' in text \
            or 'unintellible' in text \
            or 'unintelligble' in text \
            or 'unintelligibile' in text \
            or 'intelligible' in text:
        return ''

    text = BRACKET.sub('', text)
    text = SPECIAL_LETTER.sub(' ', text)
    text = text.lower().strip()
    return text \
        .replace('nononono', '') \
        .replace('okhow', 'ok how') \
        .replace('areaanywhere', 'area anywhere') \
        .replace('eurpoean', 'european') \
        .replace('modetate pricerange', 'moderate price range') \
        .replace('restaurent', 'restaurant') \
        .replace('retaurant', 'restaurant') \
        .replace('restuarant', 'restaurant') \
        .replace('restauarant', 'restaurant') \
        .replace('restaraunt', 'restaurant') \
        .replace('restaruant', 'restaurant') \
        .replace('resturant', 'restaurant') \
        .replace('moderatly', 'moderately') \
        .replace('modertely', 'moderately') \
        .replace('moderatley', 'moderately') \
        .replace('eentre', 'centre') \
        .replace('whare', 'where') \
        .replace('doesn t', 'does not') \
        .replace('excatly', 'exactly') \
        .replace('liketurkish', 'like turkish') \
        .replace('somenting', 'something') \
        .replace('somthing', 'something') \
        .replace('yougood', 'you good') \
        .replace('yeahthe', 'yeah the') \
        .replace('seving', 'serving') \
        .replace('expenisve', 'expensive') \
        .replace('espensive', 'expensive') \
        .replace('extansive', 'expensive') \
        .replace('phonenumber', 'phone number') \
        .replace('vietenam', 'vietnam') \
        .replace('allright', 'alright') \
        .replace('lookig', 'looking') \
        .replace('thankl', 'thank') \
        .replace('andyou', 'and you') \
        .replace('okwhat', 'ok what') \
        .replace('acan', 'can') \
        .replace('uhum', 'uh um') \
        .replace('okum', 'ok um') \
        .replace('alrightbye', 'alright bye') \
        .replace('byebye', 'bye bye') \
        .replace('goobye', 'good bye') \
        .replace('goodbye', 'good bye') \
        .replace('goodbey', 'good bye') \
        .replace('okthank', 'ok thank') \
        .replace('thankyou', 'thank you') \
        .replace('yeshello', 'yes hello') \
        .replace('terkish', 'turkish') \
        .replace('turkesh', 'turkish') \
        .replace('yesserving', 'yes serving') \
        .replace('adress', 'address') \
        .replace('koreanian', 'korean') \
        .replace('koresn', 'korean') \
        .replace('portiguese', 'portuguese') \
        .replace('portugese', 'portuguese') \
        .replace('portugeuse', 'portuguese') \
        .replace('i\'m', 'i am') \
        .replace('i\'m', 'i am') \
        .replace('i\'d', 'i would') \
        .replace('don\'t', 'do not') \
        .replace('it\'s', 'it is') \
        .replace('that\'s', 'that is') \
        .replace('what\'s', 'what is')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()

    root = parse(args.input).getroot()
    datums = root.findall('datum')
    print('# datum (%d)' % len(datums))

    slots = {'o'}
    data = set()
    params_pattern = re.compile('\w+\(([^)]+)\)')
    param_pattern = re.compile('(\w+)=([\w]+|\"[\w\s]+\")')
    for datum in datums:
        transcription = datum.findtext('transcription')
        uact = datum.findtext('uact')

        transcription = normalize(transcription)
        if len(transcription) <= 0:
            continue

        tokens = transcription.split(' ')

        passed = True
        founds = params_pattern.findall(uact)
        if len(founds) > 0:
            params = founds[0]
            params = params.split(',')
            for param in params:
                founds = param_pattern.findall(param)
                if len(founds) == 0:
                    continue

                slot = founds[0][0]
                word = founds[0][1].replace('\"', '')
                if tokens.count(word) > 1:
                    passed = False
                else:
                    for i in range(len(tokens)):
                        if tokens[i] == word:
                            tokens[i] = '({})[{}]'.format(word, slot)
                            slots.add('{}-{}'.format('b', slot))
                            slots.add('{}-{}'.format('i', slot))
                            break

        if passed:
            data.add(' '.join([token.strip() for token in tokens if len(token.strip()) > 0]))

    if not os.path.exists('./out'):
        os.mkdir('./out')

    with open(os.path.join('./out', 'semafor.slot'), 'w') as output:
        output.write('\n'.join(slots))
        print('# %d are generated.' % len(slots))

    with open(os.path.join('./out', 'semafor.output'), 'w') as output:
        output.write('\n'.join(data))
        print('# %d are generated.' % len(data))
