"""
Evaluating the Linear or CAMC or Finetune Classifier.
"""
from __future__ import print_function

import argparse
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn

from data.loader import load_test_data
from utils.train import read_prototype
from sklearn.metrics import accuracy_score, classification_report
from model.resnet_simclr import SimCLRResNet, LinearClassifier, CamcClassifier


def parse_option():
    parser = argparse.ArgumentParser('argument for evaluating')

    parser.add_argument('--classifier_type', type=str, 
                        choices=['simclr', 'linear', 'camc', 'finetune'],
                        default='finetune', help='type of the classifier')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--n_prototype', type=int, default=128,
                        help='number of prototypes')
    parser.add_argument('--n_cls', type=int, default=2,
                        help='number of classes')
    parser.add_argument('--cali_index', type=str, default="1",
                        help='the class index to be calibrated')
    parser.add_argument('--ckpt_model', type=str, 
                        default='./save/model/finetune_resnet50_lr_0.0005_bsz_32_proto_128/model_best.pth',
                        help='path to pre-trained model consistent with classifier type')
    parser.add_argument('--ckpt_classifier', type=str, 
                        default='./save/model/finetune_resnet50_lr_0.0005_bsz_32_proto_128/classifier_best.pth',
                        help='path to pre-trained model consistent with classifier type')
    parser.add_argument('--read_prototype', type=str, default='./dataset/prototype/teacher_ce_0.5_con_1.0/',
                        help='read features of prototypes in form of ONLY numpy array')

    opt = parser.parse_args()

    # calibration class index
    cali_index = opt.cali_index.split(',')
    opt.cali_index = list([])
    for ind in cali_index:
        opt.cali_index.append(int(ind))

    return opt


def set_model_simclr(opt):
    """
        Only when classifier type is simclr.
    """
    model = SimCLRResNet(name=opt.model)
    ckpt = torch.load(opt.ckpt_model, map_location='cpu')
    
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


def set_model(opt):
    """
        When classifier type is linear or camc.
    """
    model = SimCLRResNet(name=opt.model)

    if opt.classifier_type == "camc" or opt.classifier_type == "finetune":
        classifier = CamcClassifier(name=opt.model, num_prototype=opt.n_prototype)
    else:  # opt.classifier_type == "linear":
        classifier = LinearClassifier(name=opt.model)

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


def test_simclr(test_loader, model):
    """
        Test.
    """
    model.eval()
    classes = test_loader.dataset.classes
    num_class = len(classes)

    class_correct = [0]*num_class
    class_total = [0]*num_class
    y_gt, y_pred = [], []

    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for images, labels in test_loader:
                images = images.float().cuda()
                labels = labels.cuda()

                # forward
                _, output = model(images)
                _, predict_cls = torch.max(output, dim=1)
                c = (predict_cls == labels).squeeze()
                
                for i, label in enumerate(labels):
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                y_pred.extend(predict_cls.cpu().numpy())
                y_gt.extend(labels.cpu().numpy())
                pbar.update(1)

        for i in range(num_class):
            print("Accuracy of {:5s}: {:2.3f}%".format(classes[i], 100 * class_correct[i] / class_total[i]))

        ac = accuracy_score(y_gt, y_pred)
        cr = classification_report(y_gt, y_pred, target_names=classes, digits=5)
        print("Accuracy is :", ac)
        print(cr)


def test(test_loader, model, classifier, opt):
    """
        Test.
    """
    model.eval()
    classifier.eval()
    classes = test_loader.dataset.classes
    num_class = len(classes)

    class_correct = [0]*num_class
    class_total = [0]*num_class
    y_gt, y_pred = [], []

    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for images, labels in test_loader:
                images = images.float().cuda()
                labels = labels.cuda()

                # forward
                if opt.classifier_type == 'camc':
                    feat_map = model.encoder(images)[1]  # feat_map
                    output = classifier(feat_map)
                elif opt.classifier_type == 'linear':
                    feat = model.encoder(images)[0]  # features
                    output = classifier(feat)
                else:  # finetune
                    feat, feat_map = model.encoder(images)
                    output_sim = model.classify(feat)
                    output_camc = classifier(feat_map)
                    output = []
                    for i in range(opt.n_cls):
                        if i in opt.cali_index:
                            output.append(output_camc[:, i].unsqueeze(-1))
                        else:  # no need to calibration
                            output.append(output_sim[:, i].unsqueeze(-1))
                    output = torch.cat(output, dim=1)
                
                # make this output
                _, predict_cls = torch.max(output, dim=1)
                c = (predict_cls == labels).squeeze()

                for i, label in enumerate(labels):
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                y_pred.extend(predict_cls.cpu().numpy())
                y_gt.extend(labels.cpu().numpy())
                pbar.update(1)

        for i in range(num_class):
            print("Accuracy of {:5s}: {:2.3f}%".format(classes[i], 100 * class_correct[i] / class_total[i]))

        ac = accuracy_score(y_gt, y_pred)
        cr = classification_report(y_gt, y_pred, target_names=classes, digits=5)
        print("Accuracy is :", ac)
        print(cr)


def main():
    opt = parse_option()

    # build data loader
    test_loader = load_test_data()

    assert opt.classifier_type in ['simclr', 'linear', 'camc', 'finetune'], \
        "No support for {} types of classifiers.".format(opt.classifier_type)
    
    # simclr
    if opt.classifier_type == "simclr":
        model = set_model_simclr(opt)
        test_simclr(test_loader, model)
    
    # linear or camc classifier
    else:
        model, classifier = set_model(opt)
        # load prototype
        if opt.classifier_type == "camc" or opt.classifier_type == "finetune":
            print("======> Load Prototypes")
            data_info = test_loader.dataset.class_to_idx
            prototypes = read_prototype(opt, data_info, next(model.parameters()).device)
            classifier._load_prototypes(prototypes)
        test(test_loader, model, classifier, opt)


if __name__ == '__main__':
    main()
