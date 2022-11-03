import argparse

import numpy as np
import torch
import torch.nn as nn


def get_dataset(feature_dir='./data/', num_gpus=4, rank=0, device='cuda'):

    X = np.load('%s/trainX.npy' % feature_dir)
    y = np.load('%s/trainY.npy' % feature_dir)
    num_samples = X.shape[0] // num_gpus
    X = X[num_samples * rank:num_samples * (rank + 1)]
    y = y[num_samples * rank:num_samples * (rank + 1)]

    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.from_numpy(y)

    X = np.load('%s/valX.npy' % feature_dir)
    y = np.load('%s/valY.npy' % feature_dir)
    X_val = torch.tensor(X, dtype=torch.float32)
    y_val = torch.from_numpy(y)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    return X_train, y_train, X_val, y_val


class simple(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(simple, self).__init__()

        self.fc1 = nn.Linear(in_dim, out_dim * 8)
        self.fc2 = nn.Linear(out_dim * 8, out_dim)
        self.act = nn.Tanh()

        nn.init.normal_(self.fc1.weight, std=in_dim**-.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def l2_reg(self):
        loss = self.fc1.weight.pow(2).sum()
        loss = loss + self.fc2.weight.pow(2).sum()
        return loss


def train_model(weight_decay, feature_dir='./data/'):
    num_gpus = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device('cuda')

    data = get_dataset(feature_dir=feature_dir,
                       num_gpus=num_gpus,
                       rank=rank,
                       device=device)
    X_train, y_train, X_val, y_val = data

    model = simple(X_train.shape[1], y_train.max().item() + 1)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[rank],
                                                      output_device=rank)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = torch.optim.LBFGS(model.parameters())

    def closure():
        optimizer.zero_grad()
        output = model(X_train)
        loss = loss_fn(output, y_train)
        loss = loss + model.module.l2_reg() * weight_decay / 2
        torch.distributed.all_reduce(loss)
        loss /= num_gpus
        loss.backward()
        return loss

    for idx in range(150):
        optimizer.step(closure)

    output = model(X_train)
    _, pred = output.max(1)
    acc0 = (pred == y_train).float().mean().item() * 100

    output = model(X_val)
    _, pred = output.max(1)
    acc = (pred == y_val).float().mean().item() * 100
    if rank == 0:
        print('Train accuracy: %.2f, val accuracy: %.2f' % (acc0, acc))
        torch.save(model.module.state_dict(), 'model.pth')
    return


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--feature_dir', type=str, default='./data/')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    torch.distributed.init_process_group(backend='nccl')
    train_model(args.wd, feature_dir=args.feature_dir)
