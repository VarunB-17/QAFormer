"""""
File handle conversion of .parquet files
of the training and validation data
to tensors ready for training.
Accordingly, there will be 2 files create
training.pt and validation.pt
that hold the data in tensor form.
"""""
import torch
from torch.utils.data import DataLoader, Dataset
import os
import pyarrow.parquet as pq
from features import t2id
import sys

t_path, v_path = 'dataloader\\data\\train.parquet', 'dataloader\\data\\val.parquet'
train, validation = pq.read_table(t_path).to_pandas(), pq.read_table(v_path).to_pandas()


# TODO store training and validation data as variables -> DONE
# TODO import features.py function -> DONE
# TODO use params.py to get batch parameters

def collate(data, info=None, typeset=None):
    if info is ('context' or 'question'):
        return torch.tensor([t2id(x, 'word') for x in data[info]], dtype=torch.int64)

    elif info is ('context_char' or 'question_char'):
        return [torch.tensor([t2id(y, 'char') for y in x], dtype=torch.int64) for x in data[info]]

    else:
        if typeset == 'train':
            return torch.tensor([t2id(x, 'word') for x in data[info]], dtype=torch.int64)
        else:
            return [torch.tensor([t2id(y, 'word') for y in x], dtype=torch.int64) for x in data[info]]


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
        item = self.data.iloc[index]
        context = collate(item, 'context')
        context_char = collate(item, 'context_char')
        question = collate(item, 'question')
        question_char = collate(item, 'question_char')

        answer = None
        if self.type == 'train':
            answer = collate(item, 'answer', self.type)
        elif self.type == 'validation':
            answer = collate(item, 'answer', self.type)

        start = torch.tensor(item['start'], dtype=torch.float32)
        end = torch.tensor(item['end'], dtype=torch.float32)
        span = torch.cat((start.unsqueeze(0), end.unsqueeze(0)), dim=0)

        return {'ct': context,
                'ctc': context_char,
                'q': question,
                'qc': question_char,
                'ans': answer,
                'span': span}

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    print(validation.columns)
    sample = train.iloc[0]
    val = validation.iloc[0]
    print('Testing dataset features')
    print("TRAIN QUESTION")
    print(sample['question'])
    test_1 = [t2id(x, 'word') for x in sample['question']]
    print(test_1)
    print(sample['question_char'])
    test_2 = [[t2id(y, 'char') for y in x] for x in sample['question_char']]
    print(test_2)
    print('---------------------------------------------------------------------------------')
    print("VALIDATION ANSWER")
    print(val['answer'])
    test_3 = [[t2id(y, 'word') for y in x] for x in val['answer']]
    print(test_3)
    print('---------------------------------------------------------------------------------')
    sys.exit(0)
