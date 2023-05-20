
import json
path = 'dataloader\\data'


def t2id(token, vocab):
    """""
    returns identifier
    corresponding to
    given token.
    """""
    assert isinstance(token, str), "token should be a string"

    if vocab == 'word':
        with open(f'{path}\\word2ind.json', 'r') as file:
            vcb = json.load(file)

        if token in vcb:
            return vcb[token]
        else:
            return vcb['--OOV--']

    elif vocab == 'char':
        with open(f'{path}\\char2ind.json', 'r') as file:
            vcb = json.load(file)

        if token in vcb:
            return vcb[token]
        else:
            return vcb['--OOV--']


def id2t(idx, vocab):
    """""
    returns token
    corresponding to
    given identifier.
    """""
    assert isinstance(idx, int), "identifier should be a integer"

    if vocab == 'word':
        with open(f'{path}\\ind2word.json', 'r') as file:
            vcb = json.load(file)

        if str(idx) in vcb:
            return vcb[idx]
        else:
            return vcb['1']

    elif vocab == 'char':
        with open(f'{path}\\ind2char.json', 'r') as file:
            vcb = json.load(file)

        if str(idx) in vcb:
            return vcb[idx]
        else:
            return vcb['1']


