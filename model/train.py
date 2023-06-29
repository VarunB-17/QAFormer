import torch
from torch.optim import Adam
from torch.nn.functional import nll_loss
from qaformer import QaFormer
from config import _modelcfg
from torch.utils.data import DataLoader
import pyarrow.parquet as pq
from batching.batch import SQData
import os
import wandb
from tqdm import tqdm
import logging

os.environ['NUMEXPR_MAX_THREADS'] = '16'


def get_device():
    """
    Checks if gpu is available
    """
    if torch.cuda.is_available():
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')

    print(dev)
    return dev


def train_model(model, optimizer, dataset, scheduler, epochs):
    losses = []
    model.train()
    model.to(device)
    for j in range(epochs):
        lr_update = optimizer.param_groups[0]['lr']
        global_loss = 0

        for i, batch in enumerate(tqdm(dataset)):
            optimizer.zero_grad()
            C, Cc, Q, Qc, start, end = batch['ct'], batch['ctc'], batch['q'], batch['qc'], batch['start'].to(
                torch.uint8), batch['end'].to(torch.long)
            C, Cc, Q, Qc, start, end = C.to(device), Cc.to(device), Q.to(device), Qc.to(device), start.to(
                device), end.to(device)

            p1, p2 = model(C, Cc, Q, Qc)

            loss_start = nll_loss(p1, start)
            loss_end = nll_loss(p2, end)
            loss = loss_start + loss_end

            # log intermediate loss
            wandb.log({'start_loss': loss_start, 'end_loss': loss_end, 'total_loss': loss})

            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            # check for lr update
            scheduler.step(loss)
            if i % _modelcfg.patience == 0:
                wandb.log({'learning_rate': lr_update})

            global_loss += loss.item()

        # log
        wandb.log({'model_loss': global_loss / len(dataset)})

    # save
    torch.save(model.state_dict(), f'qaf_{_modelcfg.model_dim}_{_modelcfg.heads}_{_modelcfg.enc_layer}.pt')


def f1(p1, p2, start, end):
    """""
    Computes the amount of overlapping
    indices between the prediction
    and ground truth and averages
    over the amount of possibilities
    given for the ground truth.
    It computes a measure of overlap
    between both and indicate the ratio
    of overlap.
    """""
    scores = []
    prediction_range = torch.arange(p1.item(), p2.item() + 1).tolist()

    for i in range(len(start)):
        truth_range = torch.arange(start.item(), end.item() + 1).tolist()
        common_values = len(set(prediction_range) & set(truth_range))

        if common_values == 0:
            scores.append(0)
        else:
            precision = 1.0 * common_values / len(prediction_range)
            recall = 1.0 * common_values / len(truth_range)
            f1_score = (2 * precision * recall) / (precision + recall)
            scores.append(f1_score)
    return float(sum(scores) / len(scores))


def em(p1, p2, start, end):
    """""
    return exact match score averaged over 
    all 3 possibilities over each instance for
    both the start and end of the answer
    prediction.
    """""

    score = []

    for i in range(len(start)):
        if start[i] == p1 and end[i] == p2:
            score.append(1.)
        else:
            score.append(0.)

    return float(sum(score) / len(score))


def test_model(model, test):
    # need to create the model first by setting _modelcfg.debug to 1
    # change value back to 2 to test performance
    model.load_state_dict(torch.load('qaf_128_8_4.pt'))
    model.eval()

    for batch in tqdm(test):
        C, Cc, Q, Qc, start, end = batch['ct'], batch['ctc'], batch['q'], batch['qc'], batch['start'].to(
            torch.uint8), batch['end'].to(torch.long)
        C, Cc, Q, Qc, start, end = C.to(device), Cc.to(device), Q.to(device), Qc.to(device), start.to(
            device), end.to(device)

        # print(start)
        p1, p2 = model(C, Cc, Q, Qc)
        p1, p2 = torch.argmax(p1, dim=1), torch.argmax(p2, dim=1)

        exact_match = em(p1, p2, start, end)
        f1_score = f1(p1, p2, start, end)

        wandb.log({'em': exact_match, 'f1': f1_score})


if __name__ == '__main__':

    device = get_device()
    train_path = _modelcfg.train_path
    test_path = _modelcfg.test_path

    if _modelcfg.debug == 0:
        wandb.init(
            # set the wandb project where this run will be logged
            project=f"QAFormer",
            name=f'DIM_{_modelcfg.model_dim}_HEAD_{_modelcfg.heads}_ENC_{_modelcfg.enc_layer}',
            config={'model_dim': _modelcfg.model_dim,
                    'batch_size': _modelcfg.batch_size,
                    'kernel_size': _modelcfg.kernel_size,
                    'lr': _modelcfg.learning_rate,
                    'heads': _modelcfg.heads,
                    'epochs': _modelcfg.epochs}

        )
        # run if you want to test whether data gets converted
        model = QaFormer().to(device)
        optimizer = Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-7, weight_decay=5e-8,
                         lr=_modelcfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               patience=_modelcfg.patience,
                                                               factor=0.002)
        data = pq.read_table(train_path).to_pandas()
        train, validation = data[42000:43000], data[5000:6000]
        train = SQData(data=train, typeset='train')
        train = DataLoader(train, batch_size=32, num_workers=_modelcfg.workers)

        train_model(model, optimizer, train, scheduler, _modelcfg.epochs)

    elif _modelcfg.debug == 1:
        wandb.init(
            # set the wandb project where this run will be logged
            project=f"QAFormer",
            name=f'DIM_{_modelcfg.model_dim}_HEAD_{_modelcfg.heads}_ENC_{_modelcfg.enc_layer}',
            config={'model_dim': _modelcfg.model_dim,
                    'batch_size': _modelcfg.batch_size,
                    'kernel_size': _modelcfg.kernel_size,
                    'lr': _modelcfg.learning_rate,
                    'heads': _modelcfg.heads,
                    'epochs': _modelcfg.epochs}

        )
        # train model
        model = QaFormer().to(device)
        optimizer = Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-7, weight_decay=3e-7,
                         lr=_modelcfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               patience=_modelcfg.patience,
                                                               factor=0.0001)
        data = pq.read_table(train_path).to_pandas()
        data_len = len(data)
        train = SQData(data=data, typeset='train')
        train = DataLoader(train, batch_size=_modelcfg.batch_size, num_workers=_modelcfg.workers, pin_memory=True)
        train_model(model, optimizer, train, scheduler, _modelcfg.epochs)

    elif _modelcfg.debug == 2:
        wandb.init(
            # set the wandb project where this run will be logged
            project=f"QAFormer_test",
            name=f'DIM_{_modelcfg.model_dim}_HEAD_{_modelcfg.heads}_ENC_{_modelcfg.enc_layer}',
            config={'model_dim': _modelcfg.model_dim,
                    'batch_size': _modelcfg.batch_size,
                    'kernel_size': _modelcfg.kernel_size,
                    'lr': _modelcfg.learning_rate,
                    'heads': _modelcfg.heads,
                    'epochs': _modelcfg.epochs}

        )
        # test model
        data = pq.read_table(test_path).to_pandas()
        data_len = len(data)
        test = SQData(data, typeset='test')
        test = DataLoader(test, batch_size=1)
        model = QaFormer().to(device)
        test_model(model, test)
