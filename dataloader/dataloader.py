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
    else:
        for c in words:
            embedding_dict[c] = torch.rand(_params.char_dim, requires_grad=True)

    # give identifiers to out-of-vocab tokens
    oov = "--OOV--"
    null = "--NULL--"

    # convert tokens to identifiers
    token2id = {token: i for i, token in enumerate(embedding_dict.keys(), 2)}
    token2id[oov] = 0
    token2id[null] = 1

    embedding_dict[oov] = torch.zeros(_params.glove_dim)
    embedding_dict[null] = torch.zeros(_params.glove_dim)
    # convert identifiers to original token
    id2token = {i: token for token, i in token2id.items()}
    id2embed = {i: embedding_dict[token] for token, i in token2id.items()}
    # returns embedding dictionary, token2id, id2token, and id2embedding
    return embedding_dict, token2id, id2token, id2embed


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
    train = train[train['context'].progress_apply(lambda x: len(x) <= _params.max_context)]
    train = train.reset_index(drop=True)
    train = train.sort_values(by=['context'], key=lambda x: x.str.len(), ascending=True)
    train = train.reset_index(drop=True)

    # print(train['context'].apply(lambda x: len(x)).tail())
    val = val[val['context'].progress_apply(lambda x: len(x) <= _params.max_context)]
    val = val.reset_index(drop=True)
    val = val.sort_values(by=['context'], key=lambda x: x.str.len(), ascending=True)
    val = val.reset_index(drop=True)

    # remove instances where answer length is longer then 30.
    train = train[train['answer'].progress_apply(lambda x: len(x) <= _params.maxq)]
    val = val[val['answer'].apply(lambda x: all(len(sublist) <= _params.maxq for sublist in x))]
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    # had to hardcode this part
    train = train[train['start'].apply(lambda x: (x + 30) <= 400)]

    train['ans_len'] = train['answer'].progress_apply(lambda x: len(x))
    train['end'] = train[['start', 'ans_len']].sum(axis=1).apply(lambda x: span(x))
    train['start'] = train['start'].progress_apply(lambda x: span(x))
    train = train.drop(columns=['ans_len'])

    ############################################################################
    # store conversion files as pickle files
    word_emb, word2ind, ind2word, id2embed_word = pt_embedding(vocab_dist, 'word')
    char_emb, char2ind, ind2char, id2_embed_char = pt_embedding(char_dist, 'char')

    to_json = [word2ind, ind2word, char2ind, ind2char, word_emb, char_emb, id2embed_word, id2_embed_char, train, val]
    to_json_str = ['word2ind', 'ind2word', 'char2ind', 'ind2char', 'word_emb', 'char_emb', 'id2embed_word',
                   'id2_embed_char', 'train', 'val']

    # Saving all dictionaries and dataframe for batch processing
    tqdm(desc='Saving vocabulary')
    for i, item in enumerate(tqdm(to_json)):
        if i <= 3:
            save(item, f'{path}\\{to_json_str[i]}')
        elif 3 < i < 8:
            torch.save(item, f'{path}\\{to_json_str[i]}.pt')
        else:
            item.to_parquet(f'{path}\\{to_json_str[i]}.parquet')

    logging.info('Pre-processing has finished!')


# Run file
if __name__ == "__main__":
    preprocess()
