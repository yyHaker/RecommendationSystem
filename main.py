#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Author  :   yyhaker 
@Contact :   572176750@qq.com
@Time    :   2020/05/06 10:31:55
'''

# here put the import lib
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import argparse
import logging

from dataset import AvazuDataset
from models import *

logger = logging.getLogger(__name__)


def get_dataset(name, path):
    if name == 'avazu':
        return AvazuDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    if name == 'lr':
        logger.info("use logistic regression")
        return LogisticRegressionModel(field_dims)
    elif name == 'fm':
        logger.info("use factorization machine model")
        return FactorizationMachineModel(field_dims, embed_dim=16)
    elif name == 'ffm':
        logger.info("use field aware factorization machine model")
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim=4)
    elif name == 'wd':
        logger.info("use Wide&Deep model")
        return WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'dfm':
        logger.info("use Deep Factorization Machine Model")
        return DeepFactorizationMachineModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'nffm':
        logger.info('use NeuralFFM model')
        return NeuralFieldAwareFactorizationMachineModel(field_dims, embed_dim=4, mlp_dims=(4, 4), dropout=0.2)
    else:
        raise ValueError('unknown model name: ' + name)


def train(epoch, model, optimizer, data_loader, criterion, device, log_interval=1000):
    model.train()
    total_loss = 0
    for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            logger.info("epoch {}, loss: {}".format(epoch, total_loss / log_interval))
            total_loss = 0


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


def main(dataset_name,
         dataset_path,
         model_name,
         epochs,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir):
    # get dataset
    logger.info("loading data...")
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))

    # get dataloader
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    logger.info("loading data done, train size: {}, valid size: {}, test size: {}".format(train_length, valid_length, test_length))

    # get model
    model = get_model(model_name, dataset).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    logger.info("begin training...")
    for epoch in range(epochs):
        train(epoch, model, optimizer, train_data_loader, criterion, device)
        auc = test(model, valid_data_loader, device)
        logger.info("epoch: {}, validation auc: {}".format(epoch, auc))
    auc = test(model, test_data_loader, device)
    logging.info("training is done!")
    logging.info("test the model, test auc is {}".format(auc))
    torch.save(model, f'{save_dir}/{model_name}.pt')


if __name__ == '__main__':
    # set params
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='avazu')
    parser.add_argument('--dataset_path', default='data/Avazu/train', help='criteo/train.txt, avazu/train, or ml-1m/ratings.dat')
    parser.add_argument('--model_name', default='nffm')
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048 * 4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--save_dir', default='result')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    logger.info("use device {}".format(args.device))

    # run main
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir)