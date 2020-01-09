import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import copy

from network import Net
from imageloader import SELoader

import Augmentor

def momentum_update(model_q, model_k, beta = 0.999):
    param_k = model_k.state_dict()
    param_q = model_q.named_parameters()
    for n, q in param_q:
        if n in param_k:
            param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
    model_k.load_state_dict(param_k)

def queue_data(data, k):
    return torch.cat([data, k], dim=0)

def dequeue_data(data, K=4096):
    if len(data) > K:
        return data[-K:]
    else:
        return data

def initialize_queue(model_k, device, train_loader):
    queue = torch.zeros((0, 128), dtype=torch.float) 
    queue = queue.to(device)

    for batch_idx, (data, name) in enumerate(train_loader):
        x_k = data[1]
        x_k = x_k.to(device)
        k = model_k(x_k)
        k = k.detach()
        queue = queue_data(queue, k)
        queue = dequeue_data(queue, K = 10)
        break
    return queue

def train(model_q, model_k, device, train_loader, queue, optimizer, epoch, temp=0.07):
    model_q.train()
    total_loss = 0

    for batch_idx, (data, name) in enumerate(train_loader):
        x_q = data[0]
        x_k = data[1]
        x_q, x_k = x_q.to(device), x_k.to(device)
        q = model_q(x_q)
        k = model_k(x_k)
        k = k.detach()

        N = x_q.shape[0]
        K = queue.shape[0]
        l_pos = torch.bmm(q.view(N,1,-1), k.view(N,-1,1))
        l_neg = torch.mm(q.view(N,-1), queue.T.view(-1,K))

        logits = torch.cat([l_pos.view(N, 1), l_neg], dim=1)

        labels = torch.zeros(N, dtype=torch.long)
        labels = labels.to(device)

        cross_entropy_loss = nn.CrossEntropyLoss()
        loss = cross_entropy_loss(logits/temp, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        momentum_update(model_q, model_k)

        queue = queue_data(queue, k)
        queue = dequeue_data(queue)

    total_loss /= len(train_loader.dataset)

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, total_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoCo example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    batchsize = args.batchsize
    epochs = args.epochs
    out_dir = args.out
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True}

    p = Augmentor.Pipeline()
    p.rotate(probability=0.8, max_left_rotation=5, max_right_rotation=5)
    p.zoom_random(probability=0.5, percentage_area=0.8)
    p.resize(probability=1.0, width=64, height=64)

    imagedir = '/media/kashgar/data/pnn_training/pix2pix/A/train'
    maskdir = '/media/kashgar/data/pnn_training/pix2pix/B/train'

    train_se = SELoader(imagedir, maskdir, p)
    
    train_loader = torch.utils.data.DataLoader(train_se, batch_size=batchsize, shuffle=True, **kwargs)
    
    model_q = Net().to(device)
    model_k = copy.deepcopy(model_q)
    optimizer = optim.SGD(model_q.parameters(), lr=0.01, weight_decay=0.0001)
    
    queue = initialize_queue(model_k, device, train_loader)
   
    for epoch in range(1, epochs + 1):
        train(model_q, model_k, device, train_loader, queue, optimizer, epoch)
    
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model_q.state_dict(), os.path.join(out_dir, 'model.pth'))
