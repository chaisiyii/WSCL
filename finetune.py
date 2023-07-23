"""
Finetuning the Classifier.
"""
from __future__ import print_function

import os
import argparse
import time
import math
import json

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np

from tqdm import tqdm
from data.loader import load_train_data, load_val_data
from utils.train import Record, warmup_learning_rate, adjust_learning_rate
from utils.train import save_model, calculate_accuracy, read_prototype
from model.resnet_simclr import SimCLRResNet, CamcClassifier


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--n_cls', type=int, default=2,
                        help='number of classes')
    parser.add_argument('--cali_index', type=str, default="1",
                        help='the class index to be calibrated')
    parser.add_argument('--n_prototype', type=int, default=128,
                        help='number of prototypes')
    parser.add_argument('--read_data', type=str, default='./dataset/pseudo/',
                        help='data ready to train')
    parser.add_argument('--read_prototype', type=str, default='./dataset/prototype/',
                        help='read features of prototypes in form of ONLY numpy array')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='15,20,25',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model
    parser.add_argument('--model', type=str, default='resnet50')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, 
                        default='./save/model/resnet50_lr_0.04_bsz_16_temp_0.07_teacherstu_True_ce_0.5_con_1.0/teacher_last.pth',
                        help='path to pre-trained model')

    opt = parser.parse_args()
    opt.model_path = './save/model/'
    if not os.path.exists("./save"):
        os.mkdir('./save')
    if not os.path.exists(opt.model_path):
        os.mkdir(opt.model_path)

    # decay iterations
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # calibration class index
    cali_index = opt.cali_index.split(',')
    opt.cali_index = list([])
    for ind in cali_index:
        opt.cali_index.append(int(ind))

    opt.model_name = 'finetune_{}_lr_{}_bsz_{}_proto_{}'.\
        format(opt.model, opt.learning_rate, opt.batch_size, opt.n_prototype)

    if opt.cosine:
        opt.model_name = 'finetune_{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = 'finetune_{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
    
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)

    return opt


def set_model(opt):
    model = SimCLRResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = CamcClassifier(name=opt.model, num_class=opt.n_cls, num_prototype=opt.n_prototype)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """
        One epoch training.
    """
    model.eval()
    classifier.train()

    batch_time, data_time = Record(), Record()
    losses, accuracy = Record(), Record()
    train_logs = []

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute output
        feat, feat_map = model.encoder(images) # output: out, feat_map
        output_sim = model.classify(feat)
        output_camc = classifier(feat_map)  # [batch_size, num_class]
        # comprehensive output
        output = []
        for i in range(opt.n_cls):
            if i in opt.cali_index:
                output.append(output_camc[:, i].unsqueeze(-1))
            else:  # no need to calibration
                output.append(output_sim[:, i].unsqueeze(-1))
        output = torch.cat(output, dim=1)

        # compute loss
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc = calculate_accuracy(output, labels)
        accuracy.update(acc, bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        # record log
        if (idx + 1) % opt.print_freq == 0:
            log = "Train: [{}][{}/{}], Batch time: {} / average {}, Data time: {} / average {},\
               Loss: {}/ average {}, Accuracy: {}/ average {}.".format(
               epoch, idx+1, len(train_loader), batch_time.val, batch_time.avg, data_time.val, data_time.avg, 
               losses.val, losses.avg, accuracy.val, accuracy.avg)
            train_logs.append(log)
            print(log)

    return losses.avg, accuracy.avg, train_logs


def validate(val_loader, model, classifier, criterion, epoch, opt):
    """
        validation
    """
    model.eval()
    classifier.eval()

    batch_time = Record()
    losses = Record()
    accuracy = Record()

    with torch.no_grad():
        end = time.time()
        with tqdm(total=len(val_loader)) as pbar:
            for _, (images, labels) in enumerate(val_loader):
                images = images.float().cuda()
                labels = labels.cuda()
                bsz = labels.shape[0]

                # compute output
                feat, feat_map = model.encoder(images) # output: out, feat_map
                output_sim = model.classify(feat)
                output_camc = classifier(feat_map)  # [batch_size, num_class]
                # comprehensive output
                output = []
                for i in range(opt.n_cls):
                    if i in opt.cali_index:
                        output.append(output_camc[:, i].unsqueeze(-1))
                    else:  # no need to calibration
                        output.append(output_sim[:, i].unsqueeze(-1))
                output = torch.cat(output, dim=1)
                
                # compute loss
                loss = criterion(output, labels)

                # update metric
                losses.update(loss.item(), bsz)
                acc = calculate_accuracy(output, labels)
                accuracy.update(acc, bsz)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                pbar.update(1)

        # record log
        val_log = "Validation: [{}], Loss: {}, Accuracy: {}.".format(epoch, losses.avg, accuracy.avg)
        print(val_log)

    return losses.avg, accuracy.avg, val_log


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader = load_train_data(batch_size=opt.batch_size, num_workers=opt.num_workers, path=opt.read_data)
    val_loader = load_val_data(batch_size=opt.batch_size, num_workers=opt.num_workers)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # load prototype
    print("======> Load Prototypes")
    data_info = train_loader.dataset.class_to_idx
    prototypes = read_prototype(opt, data_info, next(model.parameters()).device)
    classifier._load_prototypes(prototypes)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc, train_logs = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print("========================================================")
        print('Train epoch {}, total time {:.2f}, loss: {:.3f}, accuracy:{:.3f}'.format(
            epoch, time2 - time1, loss, acc))

        # save model
        save_file = os.path.join(opt.save_folder, 'classifier_last.pth')
        save_model(classifier, optimizer, opt, epoch, save_file)
        save_file = os.path.join(opt.save_folder, 'model_last.pth')
        save_model(model, optimizer, opt, epoch, save_file)

        # eval for one epoch
        loss, val_acc, val_log = validate(val_loader, model, classifier, criterion, epoch, opt)
        if val_acc > best_acc:
            best_acc = val_acc
            save_file = os.path.join(opt.save_folder, 'classifier_best.pth')
            save_model(classifier, optimizer, opt, epoch, save_file)
            save_file = os.path.join(opt.save_folder, 'model_best.pth')
            save_model(model, optimizer, opt, epoch, save_file)

        # record log
        save_log = os.path.join(opt.save_folder, 'train_logs.txt')
        with open(save_log, 'a') as fp:
            for log in train_logs:
                fp.write(log)
                fp.write('\n')
            fp.write(val_log)
            fp.write('\n')
            fp.close()

    print('best accuracy: {:.5f}'.format(best_acc))


if __name__ == '__main__':
    main()
