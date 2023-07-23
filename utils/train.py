from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import os


class Record(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    """
        Adjust learning rate only for warm-up.
    """
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def adjust_learning_rate(args, optimizer, epoch):
    """
        Adjust learning rate only for decay.
    """
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def calculate_accuracy(output, target):
    """
        Compute the accuracy.
    """
    with torch.no_grad():
        _, pred_cls = torch.max(output, dim=1)  # [batch_size, 1]
        correct = torch.eq(pred_cls,target).float().mean().item()
        return correct
    

def read_prototype(opt, data_info, device):
    """
        Read features of prototypes in form of numpy array.
        data_info: dict['class_name': class_index]
    """
    prototypes = []
    class_inds = []
    read_path = opt.read_prototype
    for class_name, class_index in data_info.items():
        read_npy = os.path.join(read_path, class_name+'.npy')
        ptt = np.load(read_npy)
        prototypes.append(ptt)
        class_inds.append(class_index)

    # sorted
    sorted_prototypes = []
    sort = np.argsort(np.array(class_inds))
    for index in sort:
        ptt = prototypes[index]
        ptt = torch.tensor(ptt).float().to(device).squeeze(1)
        sorted_prototypes.append(ptt)
    # List[tensor(num_point, feat_dim)*class]
    return sorted_prototypes
