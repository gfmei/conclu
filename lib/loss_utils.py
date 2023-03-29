#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/29/2023 4:36 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : loss_utils.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConCriterion(nn.Module):
    """
    Taken from: https://github.com/google-research/simclr/blob/master/objective.py
    Converted to pytorch, and decomposed for a clearer understanding.
    batch_size (integer): Number of data_samples per batch.
    normalize (bool, optional): Whether to normalise the representation. (Default: True)
    temperature (float, optional): The temperature parameter of the NT_Xent loss. (Default: 1.0)
    z_i (Tensor): Representation of view 'i'
    z_j (Tensor): Representation of view 'j'
    Returns: loss (Tensor): NT_Xent loss between z_i and z_j
    """

    def __init__(self, temperature=0.1):
        super().__init__()

        self.temperature = temperature

    def forward(self, z_i, z_j, normalize=True):
        device = z_i.device
        batch_size = z_i.shape[0]
        labels = torch.zeros(batch_size * 2).long().to(device)
        mask = torch.ones((batch_size, batch_size), dtype=bool).fill_diagonal_(0).to(device)
        if normalize:
            z_i_norm = F.normalize(z_i, p=2, dim=-1)
            z_j_norm = F.normalize(z_j, p=2, dim=-1)
        else:
            z_i_norm = z_i
            z_j_norm = z_j
        bsz = z_i_norm.size(0)
        ''' Note: **
        Cosine similarity matrix of all samples in batch: a = z_i, b = z_j
         ____ ____
        | aa | ab |
        |____|____|
        | ba | bb |
        |____|____|
        Positives: Diagonals of ab and ba '\'
        Negatives: All values that do not lie on leading diagonals of aa, bb, ab, ba.
        '''
        # Cosine similarity between all views
        logits_aa = torch.mm(z_i_norm, z_i_norm.t()) / self.temperature
        logits_bb = torch.mm(z_j_norm, z_j_norm.t()) / self.temperature
        logits_ab = torch.mm(z_i_norm, z_j_norm.t()) / self.temperature
        logits_ba = torch.mm(z_j_norm, z_i_norm.t()) / self.temperature
        # Compute Positive Logits
        logits_ab_pos = logits_ab[torch.logical_not(mask)]
        logits_ba_pos = logits_ba[torch.logical_not(mask)]
        # Compute Negative Logits
        logit_aa_neg = logits_aa[mask].reshape(bsz, -1)
        logit_bb_neg = logits_bb[mask].reshape(bsz, -1)
        logit_ab_neg = logits_ab[mask].reshape(bsz, -1)
        logit_ba_neg = logits_ba[mask].reshape(bsz, -1)
        # Positive Logits over all samples
        pos = torch.cat((logits_ab_pos, logits_ba_pos)).unsqueeze(1)
        # Negative Logits over all samples
        neg_a = torch.cat((logit_aa_neg, logit_ab_neg), dim=1)
        neg_b = torch.cat((logit_ba_neg, logit_bb_neg), dim=1)
        neg = torch.cat([neg_a, neg_b], dim=0)
        # Compute cross entropy
        logits = torch.cat([pos, neg], dim=1)
        loss = F.cross_entropy(logits, labels)

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, temperature=0.01):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.tau = temperature

    def forward(self, x, y, normalize=True):
        """
        :param normalize:
        :param x: (bs, n, d)
        :param y: (bs, n, d)
        :return: loss
        """
        bs, n, dim = y.shape
        labels = torch.zeros((2 * n,), dtype=torch.long, device=x.device).expand(bs, -1).reshape(-1)
        mask = torch.ones((n, n), dtype=bool, device=x.device).fill_diagonal_(0).expand(bs, -1, -1)
        if normalize:
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)
        score_xy = torch.einsum('bmd,bnd->bmn', x, y) / self.tau
        score_yx = torch.einsum('bmd,bnd->bmn', y, x) / self.tau
        score_xx = torch.einsum('bmd,bnd->bmn', x, x) / self.tau
        score_yy = torch.einsum('bmd,bnd->bmn', y, y) / self.tau
        # Compute Postive Logits
        logits_xy_pos = score_xy[torch.logical_not(mask)].view(bs, -1)
        logits_yx_pos = score_yx[torch.logical_not(mask)].view(bs, -1)
        # Compute Negative Logits
        logit_xx_neg = score_xx[mask].reshape(bs, n, -1)
        logit_yy_neg = score_yy[mask].reshape(bs, n, -1)
        logit_xy_neg = score_xy[mask].reshape(bs, n, -1)
        logit_yx_neg = score_yx[mask].reshape(bs, n, -1)
        # Postive Logits over all samples
        pos = torch.cat((logits_xy_pos, logits_yx_pos), dim=1).unsqueeze(-1)
        # Negative Logits over all samples
        neg_x = torch.cat((logit_xx_neg, logit_xy_neg), dim=2)
        neg_y = torch.cat((logit_yx_neg, logit_yy_neg), dim=2)
        neg = torch.cat((neg_x, neg_y), dim=1)
        # Compute cross entropy
        logits = torch.cat((pos, neg), dim=2).view(-1, 2 * n - 1)
        loss = self.ce(logits, labels)

        return loss


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d, device=trans.device)[None, :, :]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


def loss_fn(x, y, trans=None, scale=0.001):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    loss = 2 - 2 * (x * y).sum(dim=-1)
    if trans is not None:
        return loss.mean() + scale * feature_transform_regularizer(trans)
    return loss.mean()


def loss_fnl(x, y, trans=None, scale=0.001):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    loss = 2 - 2 * torch.einsum('bdk,bd->bk', x, y).sum(dim=-1)
    if trans is not None:
        return loss.mean() + scale * feature_transform_regularizer(trans)
    return loss.mean()


class MetricLoss(nn.Module):
    def __init__(self, tau=0.02):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.tau = tau

    def get_metric_loss(self, x, ref):
        """
        :param x: (bs, n_rkhs)
        :param ref: (bs, n_rkhs, n_loc)
        :return: loss
        """
        bs, n_rkhs, n_loc = ref.size()
        ref = ref.transpose(0, 1).reshape(n_rkhs, -1)
        score = torch.matmul(x, ref) / self.tau  # (bs * n_loc, bs)
        score = score.view(bs, bs, n_loc).transpose(1, 2).reshape(bs * n_loc, bs)
        gt_label = torch.arange(bs, dtype=torch.long,
                                device=x.device).view(bs, 1).expand(bs, n_loc).reshape(-1)
        return self.ce(score, gt_label)

    def forward(self, x, refs):
        loss = 0.0
        for ref in refs:
            loss += self.get_metric_loss(x, ref)
        return loss


def chamloss(x, y):
    """
    :param x: (bs, np, 3)
    :param y: (bs, np, 3)
    :return: loss
    """
    x = x.unsqueeze(1)
    y = y.unsqueeze(2)
    dist = torch.sqrt(1e-6 + torch.sum(torch.pow(x - y, 2), -1))  # bs, ny, nx
    min1, _ = torch.min(dist, 1)
    min2, _ = torch.min(dist, 2)

    return min1.mean() + min2.mean()
