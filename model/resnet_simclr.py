"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + self.shortcut(x)
        preact = out
        out = F.relu(out)

        if self.is_last:
            return out, preact
        else:
            return out
        

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + self.shortcut(x)
        preact = out
        out = F.relu(out)

        if self.is_last:
            return out, preact
        else:
            return out
        

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        feat_map = out
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out, feat_map


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}


class SyncBatchNorm(nn.Module):
    """
        Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose.
    """
    def __init__(self, dim, affine=True):
        super(SyncBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x
    

class SimCLRResNet(nn.Module):
    """
        Encoder + Projection head + Classification head.
    """
    def __init__(self, name='resnet50', head='mlp', num_class=2, feat_dim=128):
        super(SimCLRResNet, self).__init__()
        # Encoder
        model, in_dim = model_dict[name]
        self.encoder = model()

        # Projection head
        assert head in ["linear", "mlp"], "Head not supported: {}".format(head)
        if head == 'linear':
            self.proj_head = nn.Linear(in_dim, feat_dim)
        else:  # MLP
            self.proj_head = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, feat_dim)
            )
        
        # Classification head
        if num_class == 0:
            self.cls_head = None
        else:
            self.cls_head = nn.Sequential(
                nn.Linear(in_dim, num_class),
                nn.Dropout(p=0.5, inplace=True),
                # nn.Softmax(dim=1)
            )

    def on_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def forward(self, x):
        feats, _ = self.encoder(x)
        # features to classification
        if self.cls_head is not None:
            output = self.cls_head(feats)
        else:
            output = None
        # features continue to projection
        feats = F.normalize(self.proj_head(feats), dim=1)
        return feats, output
    
    def extract(self, x):
        """
            Extract features. Not featmap but features.
        """
        return self.encoder(x)[0]
    
    def classify(self, feats):
        """
            Use extracted features to classifiy.
            Used in "generate_pseudo.py".
        """
        output = self.cls_head(feats)
        return output
    
    def get_cam(self, x):
        """
            Return feature maps and their average pooling features in specific class.
        """
        out, feat_map = self.encoder(x)  # [B, C, H, W]
        out = self.cls_head(out)
        out = F.softmax(out, dim=1)
        cls_conf, cls_ind = torch.max(out, dim=1)
        # get weight for each feature map
        for name, param in self.named_parameters():
            if "cls_head" in name and "weight" in name:
                weight = param  # [num_class, C]
                break
        cls_weight = weight[cls_ind, :]  # [1, C]
        return feat_map, cls_weight, cls_ind, cls_conf
        

class LinearClassifier(nn.Module):
    """
        Linear Classifier.
    """
    def __init__(self, name='resnet50', num_class=2):
        super(LinearClassifier, self).__init__()
        _, in_dim = model_dict[name]
        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, num_class),
            nn.Dropout(p=0.5, inplace=True),
            # nn.Softmax(dim=1)
        )
    
    def forward(self, features):  # Return confidence and class index
        output = self.cls_head(features)
        return output
    

class CamcClassifier(nn.Module):
    """
        Classifier with CAM Calibration module.
    """
    def __init__(self, name='resnet50', num_class=2, num_prototype=1024):
        super(CamcClassifier, self).__init__()
        self.num_prototype = num_prototype
        self.num_class = num_class
        self.prototypes = None
        _, in_dim = model_dict[name]
        
        # CAMC and Class Head for each class
        # ATTENTION: order of prototype should keep the same as order of classification head
        camc_convs, cls_heads = [], []
        for ind in range(num_class):
            camc_conv = nn.Conv2d(num_prototype, 1, kernel_size=3, stride=1, padding=1, bias=False)
            cls_head = nn.Sequential(
                nn.Linear(in_dim, 1),
                nn.Dropout(p=0.5, inplace=True))
            self.add_module("camc_conv{}".format(ind), camc_conv)
            self.add_module("cls_head{}".format(ind), cls_head)
            camc_convs.append(camc_conv)
            cls_heads.append(cls_head)
        
        self.camc_convs = camc_convs
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_heads = cls_heads
        self.linear_cls_head = nn.Sequential(
            nn.Linear(in_dim, num_class),
            nn.Dropout(p=0.5, inplace=True),
            # nn.Softmax(dim=1)
        )

    def _load_prototypes(self, prototypes):
        """
            prototypes (List[tensor]): List for feature embeddings in shape (num_prototype, in_dim).
            ATTENTION: order of prototype should keep the same as order of classification head
        """
        new_prototypes = []
        # ptt is in form of numpy.array
        for ptt in prototypes:
            if ptt.shape[0] >= self.num_prototype:
                new_prototypes.append(ptt[0:self.num_prototype, :])
            else:
                n_ptt = self.num_prototype // ptt.shape[0] + 1
                ptt = ptt.tile((n_ptt, 1))
                new_prototypes.append(ptt[0:self.num_prototype, :])

        self.prototypes = new_prototypes

    def forward(self, feat_map):
        scores = []
        for i in range(self.num_class):
            # feat_map: tensor[B,in_dim,H,W]
            # ATTENTION: order of prototype should keep the same as order of classification head
            prototype = self.prototypes[i]  # tensor[num_prototype, in_dim]
            camc_conv = self.camc_convs[i]
            cls_head = self.cls_heads[i]
            # CAM Calibration
            cali_map = feat_map.permute(0, 2, 3, 1)    # [B,H,W,in_dim]
            cali_map = torch.matmul(cali_map, prototype.T) # [B,H,W,num_prototype]
            cali_map = cali_map.permute(0, 3, 1, 2)    # [B,num_prototype,H,W]
            cali_map = F.sigmoid(camc_conv(cali_map))  # [B,1,H,W]
            feat_map = (1+cali_map)*feat_map           # [B,in_dim,H,W]
            # Classification
            feat = self.avgpool(feat_map)
            feat = torch.flatten(feat, 1)  # [B,in_dim]
            feat = F.normalize(feat, dim=1)  # [B, in_dim]
            logits = cls_head(feat)  # [B,1]
            scores.append(logits)
        scores = torch.cat(scores, dim=1)  # [B,num_class]
        return scores

    def get_cam(self, feat_map, cls_ind):
        """
            Return feature maps and their average pooling features.
        """
        # forward the feat_map
        cali_map = feat_map.permute(0, 2, 3, 1)
        cali_map = torch.matmul(cali_map, self.prototypes[cls_ind].T) # [B,H,W,num_prototype]
        cali_map = cali_map.permute(0, 3, 1, 2)    # [B,num_prototype,H,W]
        cali_map = F.sigmoid(self.camc_convs[cls_ind](cali_map))  # [B,1,H,W]
        feat_map = (1+cali_map)*feat_map           # [B,1,H,W]

        # get weight for each feature map
        for name, param in self.named_parameters():
            if "cls_head"+str(cls_ind) in name and "weight" in name:
                cls_weight = param  # [1, C]
                break
        return feat_map, cls_weight