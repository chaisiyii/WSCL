"""
Training SimCLR architecture for feature extraction in PyTorch.
Adapted From: https://github.com/HobbitLong/SupContrast
"""
from __future__ import print_function

import os
import argparse
import time
import math
import json

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from tqdm import tqdm
from data.loader import load_data_simclr, load_data_simclr_val
from model.resnet_simclr import SimCLRResNet
from utils.loss import ContrastiveLoss, DecayCrossEntropyLoss
from utils.ema import ExponentialMovingAverage
from utils.train import Record, warmup_learning_rate, adjust_learning_rate, save_model

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=2,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--ema', type=bool, default=True,
                        help='use ema student-teacher model or not' )

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.04,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='70,80,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model
    parser.add_argument('--model', type=str, default='resnet50')

    # temperature etc., hyper-parameters
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--ce_decay', type=float, default=0.95,
                        help='the base of decay cross entropy loss')
    parser.add_argument('--stop_epoch', type=int, default=30,
                        help='after the stop epoch the ce-loss of corresponding labels are 0')
    parser.add_argument('--alpha_ce', type=float, default=0.5,
                        help='the coefficient of cross-entropy loss')
    parser.add_argument('--alpha_con', type=float, default=1.0,
                        help='the coefficient of contrastive loss')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    opt = parser.parse_args()

    if not os.path.exists("./save"):
        os.mkdir('./save')
    opt.model_path = './save/model/'
    opt.tb_path = './save/tensorboard/'
    
    if not os.path.exists(opt.model_path):
        os.mkdir(opt.model_path)
    if not os.path.exists(opt.tb_path):
        os.mkdir(opt.tb_path)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_lr_{}_bsz_{}_temp_{}_teacherstu_{}_ce_{}_con_{}'.\
        format(opt.model, opt.learning_rate, opt.batch_size, opt.temp, opt.ema, opt.alpha_ce, opt.alpha_con)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_model(opt):
    model = SimCLRResNet(name=opt.model)
    criterion_ce = DecayCrossEntropyLoss(stop_epoch=opt.stop_epoch, base=opt.ce_decay)
    criterion_con = ContrastiveLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion_ce = criterion_ce.cuda()
        criterion_con = criterion_con.cuda()
        cudnn.benchmark = True

    return model, criterion_ce, criterion_con


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def train(train_loader, model, criterion_ce, criterion_con, optimizer, epoch, opt, teacher_model=None):
    """
        One epoch training.
    """
    model.train()
    train_logs = []

    batch_time, data_time = Record(), Record()
    ce_losses, con_losses, losses = Record(), Record(), Record()

    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        batch_size = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # inference on original model
        features, output = model(images)
        
        if opt.alpha_con != 0:
            f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # [B, 2, H, W]
            # if use teacher-student model
            if opt.ema:
                assert teacher_model is not None, "teacher model should not be empty!"
                teacher_model.update(model)
                features_teacher, _ = teacher_model(images)
                ft1, ft2 = torch.split(features_teacher, [batch_size, batch_size], dim=0)
                features_teacher = torch.cat([ft1.unsqueeze(1), ft2.unsqueeze(1)], dim=1)  # [B, 2, H, W]
                features = torch.cat([features, features_teacher], dim=1)  # [B, 4, H, W]           
            con_loss = opt.alpha_con * criterion_con(features)

        else:
            con_loss = torch.tensor(0).cuda()

        # compute loss
        ce_loss = opt.alpha_ce * criterion_ce(output, labels, epoch)
        loss = ce_loss + con_loss

        # update losses
        ce_losses.update(ce_loss, batch_size)
        con_losses.update(con_loss, batch_size)
        losses.update(loss, batch_size)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # record log
        if (idx + 1) % opt.print_freq == 0:
            log = "Train: [{}][{}/{}], Batch time: {} / average {}, Data time: {} / average {},\
               CeLoss: {}/ average {}, ConLoss: {}/ average {}, Loss: {}/ average {}.".format(
               epoch, idx+1, len(train_loader), batch_time.val, batch_time.avg, data_time.val, data_time.avg, 
               ce_losses.val, ce_losses.avg, con_losses.val, con_losses.avg, losses.val, losses.avg)
            train_logs.append(log)
            print(log)

    return losses.avg, ce_losses.avg, con_losses.avg, train_logs
    

def val(val_loader, model, criterion_ce, criterion_con, epoch, opt, teacher_model=None):
    """
        One epoch validation.
    """
    val_ce_loss = 0
    val_con_loss = 0
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as pbar:
            for images, labels in val_loader:
                images = torch.cat([images[0], images[1]], dim=0)
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                batch_size = labels.shape[0]

                # inference on original model
                features, output = model(images)

                if opt.alpha_con != 0:
                    f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # [B, 2, H, W]
                    # if use teacher-student model
                    if opt.ema:
                        assert teacher_model is not None, "teacher model should not be empty!"
                        teacher_model.update(model)
                        features_teacher, _ = teacher_model(images)
                        ft1, ft2 = torch.split(features_teacher, [batch_size, batch_size], dim=0)
                        features_teacher = torch.cat([ft1.unsqueeze(1), ft2.unsqueeze(1)], dim=1)  # [B, 2, H, W]
                        features = torch.cat([features, features_teacher], dim=1)  # [B, 4, H, W]
                    con_loss = opt.alpha_con * criterion_con(features)

                else:
                    con_loss = torch.tensor(0).cuda()

                # compute loss
                ce_loss = opt.alpha_ce * criterion_ce(output, labels, epoch=0)

                val_ce_loss = val_ce_loss + ce_loss.item()
                val_con_loss = val_con_loss + con_loss.item()
                
                pbar.update(1)
        
        # calculate total loss
        val_ce_loss = val_ce_loss/len(val_loader.dataset)
        val_con_loss = val_con_loss/len(val_loader.dataset)
        val_loss = val_ce_loss + val_con_loss

        # record log
        val_log = "Validation: [{}], CeLoss: {}, ConLoss: {}, Loss: {}".format(
               epoch, val_ce_loss, val_con_loss, val_loss)
        print(val_log)

    return val_loss, val_ce_loss, val_con_loss, val_log
    

def main():
    opt = parse_option()

    # build dataloader
    train_loader = load_data_simclr(batch_size=opt.batch_size, num_workers=opt.num_workers)
    val_loader = load_data_simclr_val(batch_size=opt.batch_size, num_workers=opt.num_workers)

    # build model and optimizer
    model, criterion_ce, criterion_con = set_model(opt)
    optimizer = set_optimizer(opt, model)

    # build tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # build teacher model or pass
    if opt.ema:
        teacher_model = ExponentialMovingAverage(model)
    else:
        teacher_model = None

    # training
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, ce_loss, con_loss, train_logs = train(train_loader, model, criterion_ce, 
                            criterion_con, optimizer, epoch, opt, teacher_model)
        time2 = time.time()
        print("========================================================")
        print('epoch {}, total time {:.2f}, loss: {:.3f}.'.format(epoch, time2 - time1, loss))

        # tensorboard
        logger.log_value('loss', loss, epoch)
        logger.log_value('CeLoss', ce_loss, epoch)
        logger.log_value('ConLoss', con_loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # validation
        val_loss, val_ce_loss, val_con_loss, val_log = val(val_loader, model, criterion_ce, 
                                criterion_con, epoch, opt, teacher_model)
        logger.log_value('val_loss', val_loss, epoch)
        logger.log_value('val_CeLoss', val_ce_loss, epoch)
        logger.log_value('val_ConLoss', val_con_loss, epoch)

        # save
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{}_student.pth'.format(epoch)
            )
            save_model(model, optimizer, opt, epoch, save_file)
            if opt.ema:
                save_file = os.path.join(
                    opt.save_folder, 'ckpt_epoch_{}_teacher.pth'.format(epoch)
                )
                save_model(teacher_model.model, optimizer, opt, epoch, save_file)

        # write log per epoch
        save_log = os.path.join(opt.save_folder, 'train_simclr_logs.txt')
        with open(save_log, 'a') as fp:
            for log in train_logs:
                fp.write(log)
                fp.write('\n')
            fp.write(val_log)
            fp.write('\n')
            fp.close()

    # save the last model
    save_file = os.path.join(opt.save_folder, 'student_last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
    if opt.ema:
        save_file = os.path.join(opt.save_folder, 'teacher_last.pth')
        save_model(teacher_model.model, optimizer, opt, opt.epochs, save_file)


if __name__ == "__main__":
    main()

