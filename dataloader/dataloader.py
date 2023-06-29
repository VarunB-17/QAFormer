"""""
Load SQuAD dataset and
pre-process sequences
according to QANets
tokenizing procedure.
"""""

import logging
import pandas as pd
import spacy
from datasets import load_dataset
from torchtext.vocab import GloVe
import numpy as np
import json
from tqdm import tqdm
import torch
from collections import Counter
from params import _params

tqdm.pandas()
vocab_dist = Counter()
char_dist = Counter()
glove = GloVe(name=_params.glove_param, dim=_params.glove_dim)  # glove 300D vector embeddings
logging.basicConfig(level=logging.INFO)
path = f'{_params.root}\\data'

# define embedding and tokenizer
spacy.cli.download("en_core_web_sm")  # download model
nlp = spacy.load("en_core_web_sm")  # uncased model


def word_tokenize(sent, col_type):
    """""
    Convert string to tokenized list of word tokens,
    and removes punctuation from a Doc object
    with the exception of '?' for queries.
    """""
    if type(sent) != np.ndarray:
        sent = sent.replace(
            "''", '" ').replace("``", '" ')
        doc = nlp(sent)
        token_list = []
        pos_list = []
        for token in doc:
            if col_type == 'question':
                if not token.is_punct or token.text == '?':
                    pos_list.append(token.pos_)
                    token_list.append(token.text.lower())
                    vocab_dist[token.text.lower()] += 1
            elif col_type == 'context':
                if not token.is_punct:
                    pos_list.append(token.pos_)
                    token_list.append(token.text.lower())
                    vocab_dist[token.text.lower()] += 1
            elif col_type == 'answer':
                if not token.is_punct:
                    token_list.append(token.text.lower())
        if pos_list:
            char_list = [list(token) for token in token_list]
            for word in char_list:
                for char in word:
                    char_dist[char] += 1

            return {'token': token_list,
                    'pos': pos_list,
                    'char': char_list
                    }
        else:
            return token_list
    else:
        token_list = []
        for sen in sent:
            _temp = []
            doc = nlp(sen)
            for token in doc:
                if not token.is_punct:
                    _temp.append(token.text.lower())
            token_list.append(_temp)
        return token_list


def pt_embedding(counter, type_emb):
    """""
    Converts counter object of characters
    or words to their corresponding embedding
    dictionary and create additional
    token2id and id2token and id2embedding
    dictionaries for lookup.
    """""
    embedding_dict = {}
    words = [k for k, v in counter.items() if v > -1]

    # word embeddings
    if type_emb == 'word':
        for w in words:
            if w in glove.stoi:
                embedding_dict[w] = glove['w']

    # character embeddings
    elif type_emb == 'char':
        for c in words:
            embedding_dict[c] = torch.randn(_params.char_dim, requires_grad=True)

    # give identifiers to out-of-vocab tokens
    oov = "--OOV--"
    pad = "--PAD--"

    # convert tokens to identifiers
    token2id = {token: i for i, token in enumerate(embedding_dict.keys(), 2)}
    token2id[pad] = 0
    token2id[oov] = 1
    if type_emb == 'word':
        embedding_dict[oov] = torch.zeros(_params.glove_dim)
        embedding_dict[pad] = torch.zeros(_params.glove_dim)
    elif type_emb == 'char':
        embedding_dict[oov] = torch.zeros(_params.char_dim)
        embedding_dict[pad] = torch.zeros(_params.char_dim)

    # convert identifiers to original token
    id2embed = {i: embedding_dict[token] for token, i in token2id.items()}
    emb_mat = [id2embed[idx] for idx in range(len(id2embed))]
    # returns embedding matrix and token to id converter as .pt and .json files
    return emb_mat, token2id


def save(dct, fn):
    # Open a file to write to
    with open(f"{fn}.json", "w") as f:
        # Use pickle to dump the dictionary to the file
        json.dump(dct, f)


def span(x):
    tens = np.zeros(_params.max_context)
    tens[x] = 1.
    return tens


def preprocess() -> [pd.DataFrame, pd.DataFrame]:
    """""
    load the raw SQuAD dataset, and converts
    the pyarrow file to pandas for easy
    augmentation and readability.
    """""

    # load data
    squad = load_dataset('squad', cache_dir='dataset')
    train = squad['train'].to_pandas()
    val = squad['validation'].to_pandas()
    train, val = train.drop(columns=['id', 'title']), val.drop(columns=['id', 'title'])
    ############################################################################
    # separate answer string from answer start token
    train['answer'] = train['answers'].apply(lambda x: x['text'][0])
    train['start'] = train['answers'].apply(lambda x: x['answer_start'][0])

    val['answer'] = val['answers'].apply(lambda x: x['text'])
    val['start'] = val['answers'].apply(lambda x: x['answer_start'])

    train, val = train.drop(columns=['answers']), val.drop(columns=['answers'])
    train.name = 'train'
    val.name = 'val'
    ############################################################################
    # apply tokenization
    cols = ['context', 'question', 'answer']
    df = [train, val]
    for att in cols:
        for data in df:
            tqdm.pandas(desc=f'Tokenizing {att} {data.name}')

            data[att] = data[att].progress_apply(lambda x: word_tokenize(x, att))
    ############################################################################
    train['context_pos'] = train['context'].apply(lambda x: x['pos'])
    train['context_char'] = train['context'].apply(lambda x: x['char'])
    train['context'] = train['context'].apply(lambda x: x['token'])

    val['context_pos'] = val['context'].apply(lambda x: x['pos'])
    val['context_char'] = val['context'].apply(lambda x: x['char'])
    val['context'] = val['context'].apply(lambda x: x['token'])

    train['question_pos'] = train['question'].apply(lambda x: x['pos'])
    train['question_char'] = train['question'].apply(lambda x: x['char'])
    train['question'] = train['question'].apply(lambda x: x['token'])

    val['question_pos'] = val['question'].apply(lambda x: x['pos'])
    val['question_char'] = val['question'].apply(lambda x: x['char'])
    val['question'] = val['question'].apply(lambda x: x['token'])
    ############################################################################
    # remove instances with context that are longer than 600 tokens
    train = train[train['context'].apply(lambda x: len(x) <= _params.max_context)]
    train = train.reset_index(drop=True)
    train = train.sort_values(by=['context'], key=lambda x: x.str.len(), ascending=True)
    train = train.reset_index(drop=True)

    val = val[val['context'].apply(lambda x: len(x) <= _params.max_context)]
    val = val.reset_index(drop=True)
    val = val.sort_values(by=['context'], key=lambda x: x.str.len(), ascending=True)
    val = val.reset_index(drop=True)

    # remove instances where answer length is longer then 30.
    train = train[train['answer'].apply(lambda x: len(x) <= _params.maxq)]
    val = val[val['answer'].apply(lambda x: all(len(sublist) <= _params.maxq for sublist in x))]
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    # remove instances for which the answer length exceeds the context length
    train = train[train['start'].apply(lambda x: (x + _params.maxq) <= _params.max_context)]
    val = val[val['start'].apply(lambda x: all(num + _params.maxq <= _params.max_context for num in x))]

    # create an additional columns that specifies the end of an answer in the context
    train['ans_len'] = train['answer'].apply(lambda x: len(x))
    train['end'] = train[['start', 'ans_len']].sum(axis=1)
    train['start'] = train['start']
    train = train.drop(columns=['ans_len'])

    val['ans_len'] = val['answer'].apply(lambda x: [len(ans) for ans in x])
    val['end'] = [list(map(sum, zip(lst1, lst2))) for lst1, lst2 in zip(val['start'], val['ans_len'])]
    val['end'] = val['end']
    val['start'] = val['start']
    val = val.drop(columns=['ans_len'])

    train = train.sort_values(by=['context'], key=lambda x: x.str.len(), ascending=True)
    train = train.reset_index(drop=True)
    val = val.sort_values(by=['context'], key=lambda x: x.str.len(), ascending=True)
    val = val.reset_index(drop=True)

    ############################################################################
    # store conversion files
    word_emb, token2ind = pt_embedding(vocab_dist, 'word')
    char_emb, char2ind, = pt_embedding(char_dist, 'char')

    to_file = [token2ind, char2ind, word_emb, char_emb, train, val]
    to_file_str = ['token2ind', 'char2ind', 'word_emb', 'char_emb', 'train', 'val']

    # Saving all dictionaries and dataframe for batch processing
    tqdm(desc='Saving vocabulary')
    for i, item in enumerate(tqdm(to_file)):
        if i <= 1:
            pass
            save(item, f'{path}\\{to_file_str[i]}')
        elif 1 < i < 4:
            pass
            torch.save(item, f'{path}\\{to_file_str[i]}.pt')
        else:
            item.to_parquet(f'{path}\\{to_file_str[i]}.parquet')



# Run file
if __name__ == "__main__":
    preprocess()
