"""
Implementation of Unsupervised Contrastive Loss and Decay Cross-entropy Loss.
Adapted From: https://github.com/HobbitLong/SupContrast
"""
from __future__ import print_function

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features: Tensor):
        """
        Compute the unsupervised contrastive loss for SimCLR.
        Args:
            features: tensor[B, n_views, ...]
        Returns:
            A loss scalar.
        """
        device = features.device
        # device = torch.device('cuda') if features.is_cuda else torch.device('cpu')

        assert len(features.shape) >= 3, "features needs to be [B, n_views, ...], at least 3 dimensions required."
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)  # [B, n_views, ...]

        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B,B]
        
        contrast_count = features.shape[1]  # n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [B*n_views, ...]

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T),
            self.temperature)  # [B*n_views, B*n_views]
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # decrease value but keep gradient

        # tile mask
        mask = mask.repeat(contrast_count, contrast_count)  # [B*n_views, B*n_views]
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
            0
        )  # Big mask [B*n_views, B*n_views] where diagnal value is 0 else 1.
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # [B*n_views, B*n_views]
        log_prob = logits - torch.log(torch.sum(exp_logits, dim=1, keepdim=True))

        # compute mean value of log-likelihood over positive pairs
        mean_log_pro_pos = torch.sum(mask*log_prob, dim=1)/torch.sum(mask, dim=1)  # [B*n_views, 1]

        # loss
        loss = -torch.mean(mean_log_pro_pos)
        
        return loss
    

class DecayCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, stop_epoch=30, base=0.98, num_class=2, decay_labels=[1]):
        super(DecayCrossEntropyLoss, self).__init__()
        self.stop_epoch = stop_epoch
        self.decay_labels = decay_labels
        self.num_class = num_class
        self.base = base
    
    def forward(self, input: Tensor, target: Tensor, epoch: int):
        # calculate coefficient
        device = target.device
        if epoch > self.stop_epoch:
            alpha = torch.tensor(math.pow(self.base, self.stop_epoch)).to(device)
        else:
            alpha = torch.tensor(math.pow(self.base, epoch)).to(device)

        weight = torch.ones(self.num_class).to(device)
        
        for label in self.decay_labels:
            weight[label] = alpha

        # multiple transforms lead to multiple repeat
        n_repeat = input.shape[0]//target.shape[0]
        if n_repeat > 1:
            target = target.repeat(n_repeat)
        
        return F.cross_entropy(input, target, weight=weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)
    
