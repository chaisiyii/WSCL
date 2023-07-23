"""
Pseudo Labeling Based on Uncertainty.
"""


from __future__ import print_function

import os
import math
import argparse
import torch
import shutil
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from tqdm import tqdm
from data.augmentation import data_transform
from torchvision.datasets import ImageFolder
from model.resnet_simclr import SimCLRResNet


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--inference_times', type=int, default=3,
                        help='the times of inferences in generating pseudo labels')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--checkpoint', type=str, 
                        default='./save/model/resnet50_lr_0.04_bsz_16_temp_0.07_teacherstu_True_ce_0.5_con_1.0/teacher_last.pth',
                        help='model checkpoint for feature extraction and should be SimCLRResNet')
    parser.add_argument('--read_data', type=str, default='/opt/data/private/dataset6_Dataset/train/',
                        help='data ready to cluster')
    parser.add_argument('--save_data', type=str, default='./dataset/pseudo/',
                        help='path to save generated pseudo data')
    parser.add_argument('--save_prototype', type=str, default='./dataset/prototype/',
                        help='path to save prototype' )
    parser.add_argument('--thresh_score', type=float, default=0.95,
                        help='the minimal threshold of pseudo label scores')
    parser.add_argument('--thresh_var', type=float, default=0.001,
                        help='the maximal thershold of pseudo label var')

    opt = parser.parse_args()
    
    if not os.path.exists("./dataset"):
        os.mkdir('./dataset')
    if not os.path.exists(opt.save_data):
        os.mkdir(opt.save_data)
    if not os.path.exists(opt.save_prototype):
        os.mkdir(opt.save_prototype)

    return opt


def set_model(opt):
    """
        Only when classifier type is simclr.
    """
    model = SimCLRResNet(name=opt.model)
    ckpt = torch.load(opt.checkpoint, map_location='cpu')
    
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
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model


def generate_pseudo_label(model, train_dataset, inference_times=3):
    model.eval()
    features = []  # prototype features
    
    # extract prototype features
    with tqdm(total=len(train_dataset)) as pbar:
        for image, _ in train_dataset:
            torch.cuda.empty_cache()
            image = image.unsqueeze(0).cuda()
            feat = model.extract(image)
            features.append(feat.detach().cpu())
            pbar.update(1)
    
    # generate pseudo labels
    pseudo_labels = []
    pseudo_scores = []
    for i in range(inference_times):
        labels_per_time = []
        scores_per_time = []
        if i != 0:
            # set dropout rate except for the first inference
            model.on_dropout()
        with tqdm(total=len(train_dataset)) as pbar:
            for feat, sample in zip(features, train_dataset):
                """
                IF YOU NEED FIX PROTOTYPE
                """
                output = model.classify(feat.cuda())[0]
                output = F.softmax(output, dim=0)
                pscore, plabel = torch.max(output, dim=0)
                plabel = int(plabel.detach().cpu().item())
                pscore = pscore.detach().cpu().item()
                """
                IF YOU NEED FASTER.
                _, label = sample
                if label == 0:
                    # negative instance no need to inference
                    plabel = 0
                    pscore = float(1)
                else:
                    # posotive instance no need to inference
                    output = model.classify(feat.cuda())[0]
                    output = F.softmax(output, dim=0)
                    pscore, plabel = torch.max(output, dim=0)
                    plabel = int(plabel.detach().cpu().item())
                    pscore = pscore.detach().cpu().item()
                """
                # collect pseudo labels
                labels_per_time.append(plabel)
                scores_per_time.append(pscore)
                pbar.update(1)
        # After one time's inference
        pseudo_labels.append(labels_per_time)
        pseudo_scores.append(scores_per_time)

    # After three times inferences, start to generate pseudo scores
    pseudo_scores = np.array(pseudo_scores)
    pseudo_vars = np.var(pseudo_scores, axis=0)
    pseudo_scores = list(np.sum(pseudo_scores, axis=0)/inference_times)
    
    # pseudo
    if inference_times == 1:
        pseudo_labels = pseudo_labels[0]
    else:
        pseudo_labels = 1-np.array(pseudo_labels).astype(np.int8)  # 1 neg, 0 pos.
        plabels = np.multiply(pseudo_labels[0], pseudo_labels[1])
        for k in range(2, inference_times, 1):
            plabels = np.multiply(plabels, pseudo_labels[k])
        pseudo_labels = list(1-plabels)  # 1 pos, 0 neg. All negs lead to neg, otherwise pos.
    pseudo_labels = [int(_) for _ in pseudo_labels]

    features = [f.detach().cpu().numpy() for f in features]    
    
    return features, pseudo_labels, pseudo_scores, pseudo_vars


def main():
    print("===============> Setting Models and Dataset")
    opt = parse_option()
    model = set_model(opt)
    train_dataset = ImageFolder(opt.read_data, transform=data_transform['val'])
    
    print("===============> Generating Pseudo Labels")
    features, pseudo_labels, pseudo_scores, pseudo_vars = \
        generate_pseudo_label(model, train_dataset, opt.inference_times)
    paths = [s[0] for s in train_dataset.samples]
    gt_labels = [s[1] for s in train_dataset.samples]

    if os.path.exists(opt.save_data + 'neg/'):
        shutil.rmtree(opt.save_data + 'neg/')
    os.mkdir(opt.save_data + 'neg/')
    if os.path.exists(opt.save_data + 'pos/'):
        shutil.rmtree(opt.save_data + 'pos/')
    os.mkdir(opt.save_data + 'pos/')
    save_record = os.path.join(opt.save_prototype, 'record_pseudo.txt')
    if os.path.exists(save_record):
        os.remove(save_record)
    save_path = opt.save_prototype + 'pos.npy'
    if os.path.exists(save_path):
        os.remove(save_path)
    save_path = opt.save_prototype + 'neg.npy'
    if os.path.exists(save_path):
        os.remove(save_path)
    
    # sort the predition confidence
    sorted_index = np.argsort(-np.array(pseudo_scores))  # from big to small
    
    # generate pseudo labels and prototypes from confidence high to low
    pseudo_records = []
    neg_features, pos_features = [], []
    with tqdm(total=len(train_dataset)) as pbar:
        for ind in sorted_index:
            # read information
            path = paths[ind]
            gt_label = gt_labels[ind]
            pseudo_label = pseudo_labels[ind]
            pseudo_score = pseudo_scores[ind]
            pseudo_var = pseudo_vars[ind]
            feat = features[ind]
            # judge
            if gt_label == 0:
                # dst_path = opt.save_data + 'neg/'
                # shutil.copy(path, dst_path)
                if pseudo_label == 0:
                    neg_features.append(feat)
                continue
            else:  # ground-truth is 1, positive
                if pseudo_label == 0 and pseudo_score > opt.thresh_score and pseudo_var < opt.thresh_var:
                    # after pseudo labeing is neg
                    neg_features.append(feat)
                    record = "Pseudo Label: neg. Score: {}, var:{}, path: {}"\
                        .format(pseudo_score, pseudo_var, path)
                    dst_path = opt.save_data + 'neg/'
                    shutil.copy(path, dst_path)
                    pseudo_records.append(record)
                else:
                    pos_features.append(feat)
                    # after pseudo labeing is pos
                    dst_path = opt.save_data + 'pos/'
                    shutil.copy(path, dst_path)
            pbar.update(1)

    # save prototypes
    print("===============> Saving Prototypes")
    with open(save_record, 'a', encoding='utf-8') as fp1:
        for pr in pseudo_records:
            fp1.write(pr)
            fp1.write('\n')
        fp1.close()
    
    neg_features = np.array(neg_features)
    save_path = opt.save_prototype + 'neg.npy'
    np.save(save_path, neg_features)
    
    pos_features = np.array(pos_features)
    save_path = opt.save_prototype + 'pos.npy'
    np.save(save_path, pos_features)


if __name__ == '__main__':
    main()
