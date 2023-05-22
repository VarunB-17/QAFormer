"""""
File handle conversion of .parquet files
of the training and validation data
to tensors ready for training.
Accordingly, there will be 2 files create
training.pt and validation.pt
that hold the data in tensor form.
"""""

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pyarrow.parquet as pq
from features import t2id
from config import _params

t_path, v_path = '..\\dataloader\\data\\train.parquet', '..\\dataloader\\data\\val.parquet'
train, validation = pq.read_table(t_path).to_pandas(), pq.read_table(v_path).to_pandas()


# TODO store training and validation data as variables -> DONE
# TODO import features.py function -> DONE

def padding(tens, normal=False):
    """""
    truncates/pads questions and context to 400 tokens,
    and, character versions of questions and context
    to 16 tokens.
    """""

    # handles question and context sequences
    if normal:
        len_tens = tens.size(0)

        if len_tens < _params.context_len:  # pads instance until length 400
            pad_len = _params.context_len - len_tens
            to_pad = torch.zeros(pad_len, dtype=torch.int64)
            padded_tensor = torch.cat((tens, to_pad))
            return padded_tensor
        else:
            return tens
    # handles character represents of words in the context and sequences
    else:
        padded_chars = []

        for t in tens:
            len_tens = t.size(0)

            if len_tens >= _params.word_len:  # pads character instance until length 16
                t = t[:_params.word_len]
                padded_chars.append(t)

            else:
                pad_len = _params.word_len - len_tens
                to_pad = torch.zeros(pad_len, dtype=torch.int64)
                padded_tensor = torch.cat((t, to_pad))
                padded_chars.append(padded_tensor)

        return torch.stack(padded_chars)


def collate(data, info=None, typeset=None):
    """""
    converts sequences to integers representations.
    """""
    if info == 'context' or info == 'question':
        _temp = torch.tensor([t2id(x, 'word') for x in data[info]], dtype=torch.int64)
        return padding(_temp, normal=True)

    elif info == 'context_char' or info == 'question_char':
        _temp = [torch.tensor([t2id(y, 'char') for y in x], dtype=torch.int64) for x in data[info]]
        return padding(_temp)

    else:
        if typeset == 'train':
            return torch.tensor([t2id(x, 'word') for x in data[info]], dtype=torch.int64)

        else:
            return [torch.tensor([t2id(y, 'word') for y in x], dtype=torch.int64) for x in data[info]]


def span(x):
    """""
    convert start/end of an answer
    to a spanning range.
    """""
    tens = np.zeros(_params.context_len)
    tens[x] = 1.
    return tens


class SQData(Dataset):
    """""
    wrapper for a pandas dataframe
    into a pytorch dataset. helpful
    for compatibility with batching.
    """""

    def __init__(self, data, typeset=None):
        self.data = data
        self.type = typeset

    def __getitem__(self, index):
        start, end, span_ans = None, None, None
        item = self.data.iloc[index]
        context = collate(item, 'context')
        context_char = collate(item, 'context_char')
        question = collate(item, 'question')
        question_char = collate(item, 'question_char')

        answer = None
        if self.type == 'train':
            answer = collate(item, 'answer', self.type)
            start = torch.tensor(item['start'], dtype=torch.float32)
            end = torch.tensor(item['end'], dtype=torch.float32)
            span_ans = torch.cat((start.unsqueeze(0), end.unsqueeze(0)), dim=0)
        elif self.type == 'validation':
            answer = collate(item, 'answer', self.type)
            start = [torch.tensor(x, dtype=torch.float32) for x in item['start']]
            start = torch.stack(start)
            end = [torch.tensor(x, dtype=torch.float32) for x in item['end']]
            end = torch.stack(end)
            span_ans = torch.transpose(torch.stack((start, end)), 0, 1)  # torch.Size([3, 2, 400])

        return {'ct': context,
                'ctc': context_char,
                'q': question,
                'qc': question_char,
                'ans': answer,
                'span': span_ans}

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    training = SQData(data=train, typeset='train')
    validation = SQData(data=validation, typeset='validation')

    training_loader = DataLoader(training,
                                 batch_size=_params.batch_size,
                                 shuffle=False)
    validation_loader = DataLoader(validation,
                                   batch_size=_params.batch_size,
                                   shuffle=False)

    if _params.save:
        torch.save(training_loader, '../dataloader/data/training.pt')
        torch.save(validation_loader, '../dataloader/data/validation.pt')
    else:
        print('Files already exist!')
