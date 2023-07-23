"""
Visualize the CAM Map in SimCLR and CAMC Classifier.
"""
from __future__ import print_function

import os
import argparse
import cv2

import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from tqdm import tqdm
from data.loader import load_test_data
from utils.train import read_prototype
from model.resnet_simclr import SimCLRResNet, CamcClassifier


def parse_option():
    parser = argparse.ArgumentParser('argument for visualizing cam map')

    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--ckpt_model', type=str, 
                        default='./save/model/finetune_resnet50_lr_0.005_bsz_32_proto_128/model_best.pth',
                        help='path to pre-trained model in form of SimCLR')
    parser.add_argument('--ckpt_classifier', type=str, 
                        default='./save/model/finetune_resnet50_lr_0.005_bsz_32_proto_128/classifier_best.pth',
                        help='path to pre-trained CAMC classifier')
    parser.add_argument('--read_prototype', type=str, default='./dataset/prototype/',
                        help='read features of prototypes in form of ONLY numpy array')
    parser.add_argument('--n_prototype', type=int, default=128,
                        help='number of prototypes')
    
    parser.add_argument('--n_cls', type=int, default=2,
                        help='number of classes')
    parser.add_argument('--cali_index', type=str, default="1",
                        help='the class index to be calibrated')
    
    parser.add_argument('--save_folder', type=str, default='./save/cam/',
                        help='folder to save the visualize images')
    parser.add_argument('--ori_folder', type=str, default='original_cam/',
                        help='the path to save CAM map of SimCLR classifier')
    parser.add_argument('--camc_folder', type=str, default='camc_cam/',
                        help='the path to save CAM map of CAMC classifier')
    
    parser.add_argument('--vis_num', type=int, default=-1,
                        help='the number of visualization images')

    opt = parser.parse_args()
    if not os.path.exists("./save"):
        os.mkdir('./save')
    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)

    ori_path = os.path.join(opt.save_folder, opt.ori_folder)
    if not os.path.exists(ori_path):
        os.mkdir(ori_path)
    opt.ori_path = ori_path

    camc_path = os.path.join(opt.save_folder, opt.camc_folder)
    if not os.path.exists(camc_path):
        os.mkdir(camc_path)
    opt.camc_path = camc_path

    # calibration class index
    cali_index = opt.cali_index.split(',')
    opt.cali_index = list([])
    for ind in cali_index:
        opt.cali_index.append(int(ind))

    return opt


def set_model(opt):
    """
        Set SimCLR model and CMAC classifier.
    """
    model = SimCLRResNet(name=opt.model)
    classifier = CamcClassifier(name=opt.model, num_prototype=opt.n_prototype)

    # load model and classifier
    ckpt_model = torch.load(opt.ckpt_model, map_location='cpu')
    state_dict_model = ckpt_model['model']
    ckpt_cls = torch.load(opt.ckpt_classifier, map_location='cpu')
    state_dict_cls = ckpt_cls['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict_model.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict_model = new_state_dict
            new_state_dict = {}
            for k, v in state_dict_cls.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict_cls = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict_model)
        classifier.load_state_dict(state_dict_cls)

    return model, classifier


def get_cam_map(image, model, classifier, path, opt):
    """
        Get the CAM map of SimCLR and CAMC classifier.
        Args:
        image: tensor[1, C, H, W], batch_size = 1.
        path: corresponding path of image.
    """
    # output = classifier(model.encoder(image)[1])
    # CAMC Classifier
    feat, feat_map = model.encoder(image) # output: out, feat_map
    output_sim = model.classify(feat)  # feat is
    output_camc = classifier(feat_map)  # [batch_size, num_class]
    # comprehensive output, calibration
    output = []
    for i in range(opt.n_cls):
        if i in opt.cali_index:
            output.append(output_camc[:, i].unsqueeze(-1))
        else:  # no need to calibration
            output.append(output_sim[:, i].unsqueeze(-1))
    output = torch.cat(output, dim=1)
    # confidence
    output = F.softmax(output, dim=1)
    pred_conf_camc, pred_cls_camc = torch.max(output, dim=1)
    pred_cls_camc = int(pred_cls_camc.cpu().detach())
    pred_conf_camc = float(pred_conf_camc.cpu().detach())

    # Linear Classifier
    ori_featmap, ori_weight, pred_cls_sim, pred_conf_sim = model.get_cam(image)    # [B, C, H, W] and [1, C]
    cam_featmap, cam_weight = classifier.get_cam(ori_featmap, pred_cls_camc)    # [B, C, H, W] and [1, C]
    pred_cls_sim = int(pred_cls_sim.cpu().detach())
    pred_conf_sim = float(pred_conf_sim.cpu().detach())

    # get visualized map
    ori_cam_map = vis_cam(ori_featmap, ori_weight, path, pred_cls_sim, pred_conf_sim)
    cam_cam_map = vis_cam(cam_featmap, cam_weight, path, pred_cls_camc, pred_conf_camc)

    return ori_cam_map, cam_cam_map
    

def vis_cam(feat_map, weight, ori_path, pred_cls, pred_conf):
    """
        Visualize the CAM map.
        Args:
            feat_map: tensor[B, C, H, W] and B is 1.
            weight: tensor[1, C]
    """
    img = cv2.imread(ori_path)
    H, W = img.shape[0], img.shape[1]

    # tensor information and tranformation
    weight = weight.cpu().detach().numpy()
    feat_map = feat_map.squeeze(0).cpu().detach().numpy()  # [C, H, W]

    # cam
    nc, h, w = feat_map.shape
    cam = weight.dot(feat_map.reshape(nc, h*w))  # (1,nc)*(nc*hw)
    cam = cam.reshape(h, w)
    cam = (cam-np.min(cam))/(np.max(cam)-np.min(cam))
    cam = np.uint8(255 * cam)

    # resize
    cam = cv2.resize(cam, (H, W))
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    vis_cam = heatmap * 0.3 + img * 0.7

    # put class and conf prediction
    pred_cls = 'benign' if pred_cls==0 else 'malignant'
    texts = "predict: {}, score: {}".format(pred_cls, pred_conf)
    vis_cam = cv2.putText(vis_cam, texts, (20, 20), 
            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)

    return vis_cam


def main():
    opt = parse_option()

    # build data loader
    test_loader = load_test_data()
    test_dataset = test_loader.dataset

    model, classifier = set_model(opt)
    model.eval()
    classifier.eval()

    # load prototype
    print("======> Load Prototypes")
    data_info = test_loader.dataset.class_to_idx
    prototypes = read_prototype(opt, data_info, next(model.parameters()).device)
    classifier._load_prototypes(prototypes)

    data_idx = {}
    for k, v in data_info.items():
        data_idx[v] = k

    with tqdm(total=len(test_dataset)) as pbar:
        for i in range(len(test_dataset)):
            path, _ = test_dataset.samples[i]
            image, _ = test_dataset[i]
            image = image.unsqueeze(0).float().cuda()
        
            ori_cam_map, cam_cam_map = get_cam_map(image, model, classifier, path, opt)

            save_path = path.split('/')[-1]

            cv2.imwrite(opt.ori_path+save_path, ori_cam_map)
            cv2.imwrite(opt.camc_path+save_path, cam_cam_map)
            pbar.update(1)


if __name__ == '__main__':
    main()