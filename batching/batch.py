"""""
File handle conversion of .parquet files
of the training and validation data
to tensors ready for training.
Accordingly, there will be 2 files create
training.pt and validation.pt
that hold the data in tensor form.
"""""
import sys



import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
from batching.features import t2id
from batching.params import _params


def padding(tens, normal=False, info=None):
    """""
    truncates/pads questions and context to 400 tokens,
    and, character versions of questions and context
    to 16 tokens.
    """""

    # handles question and context sequences
    if normal:
        hyper_len = None
        len_tens = tens.size(0)

        if info == 'context':
            hyper_len = _params.context_len
        elif info == 'question':
            hyper_len = _params.question_len

        if len_tens < hyper_len:  # pads instance until length 400
            pad_len = hyper_len - len_tens
            to_pad = torch.zeros(pad_len, dtype=torch.int64)
            padded_tensor = torch.cat((tens, to_pad))
            return padded_tensor
        else:
            return tens
    # handles character represents of words in the context and sequences
    else:
        padded_chars = []
        hyper_len = None

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

        if info == 'context_char':
            hyper_len = _params.context_len
        elif info == 'question_char':
            hyper_len = _params.question_len

        if len(padded_chars) < hyper_len:
            pad_len = hyper_len - len(padded_chars)
            to_pad = torch.zeros(_params.word_len, dtype=torch.int64)
            for i in range(pad_len):
                padded_chars.append(to_pad)

        return torch.stack(padded_chars)


def pad_ans(tens, typeset):
    if typeset == 'train':
        len_tens = tens.size(0)
        if len_tens < _params.maxq:  # pads instance until length 400
            pad_len = _params.maxq - len_tens
            to_pad = torch.zeros(pad_len, dtype=torch.int64)
            padded_tensor = torch.cat((tens, to_pad))
            return padded_tensor
        else:
            return tens

    elif typeset == 'validation':
        padded_chars = []

        for t in tens:
            len_tens = t.size(0)

            if len_tens >= _params.maxq:  # pads character instance until length 16
                t = t[:_params.maxq]
                padded_chars.append(t)

            else:
                pad_len = _params.maxq - len_tens
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
        return padding(_temp, normal=True, info=info)

    elif info == 'context_char' or info == 'question_char':
        _temp = [torch.tensor([t2id(y, 'char') for y in x], dtype=torch.int64) for x in data[info]]
        return padding(_temp, info=info)

    else:
        if typeset == 'train':
            _temp = torch.tensor([t2id(x, 'word') for x in data[info]], dtype=torch.int64)
            return pad_ans(_temp, typeset)

        else:
            _temp = [torch.tensor([t2id(y, 'word') for y in x], dtype=torch.int64) for x in data[info]]
            return pad_ans(_temp, typeset)


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
        super().__init__()
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
            start = torch.tensor(item['start'], dtype=torch.float32)
            end = torch.tensor(item['end'], dtype=torch.float32)
            span_ans = torch.cat((start.unsqueeze(0), end.unsqueeze(0)), dim=0)

        elif self.type == 'test':
            start = [torch.tensor(x, dtype=torch.float32) for x in item['start']]
            start = torch.stack(start)
            end = [torch.tensor(x, dtype=torch.float32) for x in item['end']]
            end = torch.stack(end)

        return {'ct': context,
                'ctc': context_char,
                'q': question,
                'qc': question_char,
                'start': start,
                'end': end
                }

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    def get_device():
        """
        Checks if gpu is available
        """
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        print(device)
        return device
    dev = get_device()
    t_path, v_path = '..\\dataloader\\data\\train.parquet', '..\\dataloader\\data\\val.parquet'
    train, validation = pq.read_table(t_path).to_pandas()[:3000], pq.read_table(v_path).to_pandas()[:3000]
    len_t = len(train)
    training, training_ans = SQData(data=train[:round(len_t*0.8)], typeset='train')
    val, val_ans = SQData(data=train[round(len_t*0.8):], typeset='train')
    test, test_ans = SQData(data=validation, typeset='validation')
    print('dataset created')
    train = DataLoader(training_ans, batch_size=32, num_workers=4, pin_memory=True)
