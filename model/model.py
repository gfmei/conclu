#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/29/2023 4:33 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : model.py
# @Software: PyCharm
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.loss_utils import loss_fn, ClusterLoss, chamloss, MetricLoss, ConCriterion, feature_transform_regularizer
from md_utils import square_distance


class DecoderFC(nn.Module):
    def __init__(self, n_features=(256, 256), latent_dim=512, output_pts=2048, bn=False):
        super(DecoderFC, self).__init__()
        self.n_features = list(n_features)
        self.output_pts = output_pts
        self.latent_dim = latent_dim

        model = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):
            fc_layer = nn.Linear(prev_nf, nf)
            model.append(fc_layer)
            if bn:
                bn_layer = nn.BatchNorm1d(nf)
                model.append(bn_layer)
            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf
        fc_layer = nn.Linear(self.n_features[-1], output_pts * 3)
        model.append(fc_layer)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        bs = x.shape[0]
        # [bz, num, dim]
        x = x.view(bs, self.output_pts, 3)
        return x


class FoldingNet(nn.Module):
    def __init__(self, in_channel, k=32):
        super().__init__()

        self.in_channel = in_channel
        self.k = k
        a = torch.linspace(-1., 1., steps=self.k,
                           dtype=torch.float).view(1, self.k).expand(self.k, self.k).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=self.k,
                           dtype=torch.float).view(self.k, 1).expand(self.k, self.k).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0)

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 3, 1),
        )

    def forward(self, x):
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, self.k * self.k)
        seed = self.folding_seed.view(
            1, 2, self.k * self.k).expand(bs, 2, self.k * self.k).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)
        # (batch_size, num_points, 3)
        return fd2.transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, in_size=512, out_size=256, hidden_size=1024, used='proj'):
        super().__init__()
        if used == 'proj':
            self.net = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, out_size)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                # nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, out_size)
            )

    def forward(self, x):
        return self.net(x)


class CONV(nn.Module):
    def __init__(self, in_size=512, out_size=256, hidden_size=1024, used='proj'):
        super().__init__()
        if used == 'proj':
            self.net = nn.Sequential(
                nn.Conv1d(in_size, hidden_size, 1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, hidden_size, 1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, out_size, 1)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(in_size, hidden_size, 1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, out_size, 1)
            )

    def forward(self, x):
        return self.net(x)


def log_boltzmann_kernel(cost, u, v, epsilon):
    kernel = (-cost + u.unsqueeze(-1) + v.unsqueeze(-2)) / epsilon
    return kernel


def get_module_device(module):
    return next(module.parameters()).device


def sinkhorn(cost, p, q, epsilon=1e-2, thresh=1e-2, max_iter=100):
    # Initialise approximation vectors in log domain
    u = torch.zeros_like(p).to(p)
    v = torch.zeros_like(q).to(q)
    # Stopping criterion, sinkhorn iterations
    for i in range(max_iter):
        u0, v0 = u, v
        # u^{l+1} = a / (K v^l)
        K = log_boltzmann_kernel(cost, u, v, epsilon)
        u_ = torch.log(p + 1e-8) - torch.logsumexp(K, dim=-1)
        u = epsilon * u_ + u
        # v^{l+1} = b / (K^T u^(l+1))
        Kt = log_boltzmann_kernel(cost, u, v, epsilon).transpose(-2, -1)
        v_ = torch.log(q + 1e-8) - torch.logsumexp(Kt, dim=-1)
        v = epsilon * v_ + v
        # Size of the change we have performed on u
        diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(torch.abs(v - v0), dim=-1)
        mean_diff = torch.mean(diff)
        if mean_diff.item() < thresh:
            break
    # Transport plan pi = diag(a)*K*diag(b)
    K = log_boltzmann_kernel(cost, u, v, epsilon)
    gamma = torch.exp(K)
    # Sinkhorn distance
    loss = torch.sum(gamma * cost, dim=(-2, -1))
    return gamma, loss


def clustersk(features, centroids, epsilon=1e-3, thresh=1e-3, max_iter=20):
    device = features.device
    batch_size, dim, num = features.shape
    k = centroids.shape[0]
    # both marginals are fixed with equal weights
    p = torch.empty(batch_size, num, dtype=torch.float,
                    requires_grad=False, device=device).fill_(1.0 / num).squeeze()
    q = torch.empty(batch_size, k, dtype=torch.float,
                    requires_grad=False, device=device).fill_(1.0 / k).squeeze()
    cost = 2.0 - 2.0 * torch.einsum('bdn,kd->bnk', features, centroids)
    gamma, loss = sinkhorn(cost, p, q, epsilon, thresh, max_iter)
    return gamma, loss


def contrastsk(x, y, epsilon=1e-3, thresh=1e-3, max_iter=30, dst='fe'):
    device = x.device
    batch_size, dim, num_x = x.shape
    num_y = y.shape[-1]
    # both marginals are fixed with equal weights
    p = torch.empty(batch_size, num_x, dtype=torch.float,
                    requires_grad=False, device=device).fill_(1.0 / num_x).squeeze()
    q = torch.empty(batch_size, num_y, dtype=torch.float,
                    requires_grad=False, device=device).fill_(1.0 / num_y).squeeze()
    if dst == 'eu':
        cost = square_distance(x.transpose(-1, -2), y.transpose(-1, -2))
    else:
        cost = 2.0 - 2.0 * torch.einsum('bdn,bdm->bnm', x, y)
    gamma, loss = sinkhorn(cost, p, q, epsilon, thresh, max_iter)
    return gamma, loss


def orth_reg(clusters, reg=1e-3, is_norm=True):
    if is_norm:
        clusters = F.normalize(clusters, p=2, dim=1)
    sym = torch.mm(clusters, torch.t(clusters))
    orth_loss = reg * torch.abs(sym - torch.eye(clusters.shape[0]).to(clusters)).sum()
    return orth_loss


class SimContra(nn.Module):
    def __init__(self,
                 backbone,
                 projector=None,
                 tau=0.01,
                 decoder=None):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.criterion = ConCriterion(temperature=tau)
        self.decoder = decoder

    def forward(self, x1, x2=None, return_embedding=False):
        """
        :param x1: [bz, dim, num]
        :param x2: [bz, dim, num]
        :param return_embedding:
        :return:
        """
        if return_embedding or x2 is None:
            return self.backbone(x1, True)
        x1_out = self.backbone(x1)
        x2_out = self.backbone(x2)
        if len(x1_out) == 2:
            out1, wise_x1 = x1_out
            out2, wise_x2 = x2_out
        else:
            out1, wise_x1, trans1 = x1_out
            out2, wise_x2, trans2 = x2_out
        z1 = self.projector(out1)
        z2 = self.projector(out2)
        r_loss = torch.tensor(0.0, requires_grad=True)
        if self.decoder is not None:
            r1_x = chamloss(self.decoder(out1), x1.transpose(1, 2))
            r2_x = chamloss(self.decoder(out2), x2.transpose(1, 2))
            r_loss = (r1_x + r2_x) / 2.0
        loss_sim = 10 * self.criterion(z1, z2)

        return loss_sim, r_loss


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=32, dim=1024, proj_dim=128, vlad=True, tau=1e-2):
        """
        Args:
        num_clusters : int The number of clusters
        dim : int Dimension of descriptors
        alpha : float Parameter of initialization. Larger value is harder assignment.
        normalize_input : bool If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = math.sqrt(dim)
        self.tau = tau
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim), requires_grad=True)
        self.conv = CONV(in_size=dim, out_size=num_clusters, hidden_size=dim // 2, used='proj')
        self.ln = nn.LayerNorm(dim)
        self.projector = CONV(in_size=dim, out_size=proj_dim, hidden_size=512, used='proj')
        self.predictor = MLP(in_size=dim, out_size=proj_dim, hidden_size=512, used='proj')
        self.vlad = vlad
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter((2.0 * self.alpha * self.centroids).unsqueeze(-1))
        self.conv.bias = nn.Parameter(-self.alpha * self.centroids.norm(dim=1))

    def forward(self, features):
        bs, dim, num = features.shape
        centroids = self.centroids
        cent = F.normalize(centroids, p=2, dim=1)
        # soft-assignment
        soft_assign = self.conv(features).view(bs, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)  # [b, k, n]
        # calculate residuals to each clusters
        if self.vlad:
            residual = features.expand(
                self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - centroids.expand(
                num, -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign.unsqueeze(2)
            vlad = residual.sum(dim=-1)
        else:
            vlad = torch.einsum('bkn,kd->bkd', soft_assign, centroids)
        # intra-normalization [bs, clusters, dim]
        # tot = soft_assign.sum(-1)
        # prob = tot / tot.sum(-1, keepdim=True).clip(min=1e-3)
        # entropy = -prob * torch.log(prob.clip(min=1e-3))
        # entropy = entropy.sum(-1).mean(-1)
        vlad = self.ln(vlad).transpose(-1, -2)
        vlad = self.bn(vlad)
        c_vlad = self.predictor(torch.max(vlad, dim=-1)[0])
        vlad = self.projector(vlad).transpose(-1, -2)
        # reg = 1e-5
        # norm_x = F.normalize(features, dim=1, p=2)
        # score = torch.einsum('bdn,kd->bkn', norm_x, cent.detach()) / self.tau
        # x_loss = -torch.mean(torch.sum(soft_assign.detach() * F.log_softmax(score, dim=1), dim=1))

        return orth_reg(cent, 2e-3, is_norm=True), vlad, c_vlad


class PointClu(nn.Module):

    def __init__(self, num_clusters=32, dim=1024, tau=1e-2):
        """
        num_clusters: int The number of clusters
        dim: int Dimension of descriptors
        alpha: float Parameter of initialization. Larger value is harder assignment.
        normalize_input: bool If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.conv = CONV(in_size=dim, out_size=num_clusters, hidden_size=dim // 2, used='proj')
        self.dim = dim
        self.tau = tau

    def forward(self, feature, xyz):
        bs, dim, num = feature.shape
        # soft-assignment
        log_score = self.conv(feature).view(bs, self.num_clusters, -1)
        score = F.softmax(log_score, dim=1)  # [b, k, n]
        pi = score.sum(-1).clip(min=1e-4).unsqueeze(1).detach()  # [b, 1, k]
        with torch.no_grad():
            mu_xyz = torch.einsum('bkn,bdn->bdk', score, xyz) / pi  # [b, d, k]
            assign_xyz, dis = contrastsk(xyz, mu_xyz.detach(), max_iter=25, dst='eu')
            assign_xyz = num * assign_xyz.transpose(-1, -2)  # [b, k, n]
            mu_fea = torch.einsum('bkn,bdn->bdk', score, feature) / pi  # [b, d, k]
            n_feature = F.normalize(feature, dim=1, p=2).detach()
            n_mu = F.normalize(mu_fea, dim=1, p=2).detach()
            assign_fea, dis = contrastsk(n_feature, n_mu, max_iter=25)
            assign_fea = num * assign_fea.transpose(-1, -2)
        loss_xyz = -torch.mean(torch.sum(assign_xyz.detach() * F.log_softmax(log_score, dim=1), dim=1))
        loss_fea = -torch.mean(torch.sum(assign_fea.detach() * F.log_softmax(log_score, dim=1), dim=1))
        return loss_xyz + loss_fea


class SKVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=32, dim=1024, proj_dim=128, tau=1e-2):
        """
        num_clusters: int The number of clusters
        dim: int Dimension of descriptors
        alpha: float Parameter of initialization. Larger value is harder assignment.
        normalize_input: bool If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.tau = tau
        self.conv = CONV(in_size=dim, out_size=num_clusters, hidden_size=dim // 2, used='proj')
        self.projector = CONV(in_size=dim, out_size=proj_dim, hidden_size=512, used='proj')
        self.predictor = MLP(in_size=dim, out_size=proj_dim, hidden_size=512, used='proj')

    def forward(self, features, xyz=None):
        bs, dim, num = features.shape
        # soft-assignment
        log_score = self.conv(features).view(bs, self.num_clusters, -1)
        score = F.softmax(log_score, dim=1)  # [b, k, n]
        pi = score.sum(-1).clip(min=1e-4).unsqueeze(1).detach()  # [b, 1, k]
        if xyz is None:
            mu = torch.einsum('bkn,bdn->bdk', score, features) / pi  # [b, d, k]
        else:
            mu = torch.einsum('bkn,bdn->bdk', score, xyz) / pi  # [b, d, k]
        with torch.no_grad():
            if xyz is None:
                n_features = F.normalize(features, dim=1, p=2).detach()
                n_mu = F.normalize(mu, dim=1, p=2).detach()
                assign, dis = contrastsk(n_features, n_mu, max_iter=25)
            else:
                assign, dis = contrastsk(xyz, mu, max_iter=25, dst='eu')
            # as_pi = torch.sum(assign, dim=-1, keepdim=True).clip(min=1e-4)  # [b, k, 1]
            # assign = assign / as_pi  # [b, n, k]
            assign = num * assign.transpose(-1, -2)  # [b, k, n]
        loss = -torch.mean(torch.sum(assign.detach() * F.log_softmax(log_score, dim=1), dim=1))
        vlad = torch.einsum('bkn,bdn->bdk', score, features) / pi  # [b, d, k]
        c_vlad = self.predictor(torch.max(vlad, dim=-1)[0])
        vlad = self.projector(vlad)

        return loss, vlad, c_vlad


class Cluster(nn.Module):
    def __init__(self,
                 backbone,
                 dim=1024,
                 clusters=64,
                 proj_dim=128,
                 tau=0.01,
                 decoder=None):
        super().__init__()
        self.backbone = backbone
        self.vlad = SKVLAD(num_clusters=clusters, dim=dim, tau=tau, proj_dim=proj_dim)
        self.decoder = decoder
        self.criterion = MetricLoss()
        self.projector = MLP(in_size=dim, out_size=proj_dim)

    def forward(self, x, return_embedding=False):
        """
        :param x: [bz, dim, num]
        :param return_embedding:
        :return:
        """
        if return_embedding:
            return self.backbone(x, True)
        out = self.backbone(x)
        trans_loss = torch.tensor(0.0, requires_grad=True)
        if len(out) == 2:
            feature, wise = out
        else:
            feature, wise, trans = out
            if trans is not None:
                trans_loss = 0.001 * feature_transform_regularizer(trans)
        r_loss = torch.tensor(0.0, requires_grad=True)
        if self.decoder is not None:
            r_loss = chamloss(self.decoder(feature), x.transpose(1, 2))
        # loss_rq = torch.tensor(0.0, requires_grad=True)
        loss_rq, wise, _ = self.vlad(wise, x)  # [b, k, d]

        return loss_rq, r_loss + trans_loss


class SiamCluster(nn.Module):
    def __init__(self,
                 backbone,
                 projector=None,
                 predictor=None,
                 dim=1024,
                 clusters=64,
                 l_type='g',
                 tau=0.01,
                 decoder=None):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.g_predictor = predictor
        self.l_predictor = copy.deepcopy(self.g_predictor)
        self.l_type = l_type
        self.vlad = PointClu(num_clusters=clusters, dim=dim, tau=tau)
        self.croloss = ConCriterion(temperature=tau)
        self.decoder = decoder
        self.criterion = MetricLoss()

    def forward(self, x1, x2=None, return_embedding=False, is_con=False):
        """
        :param is_con:
        :param x1: [bz, dim, num]
        :param x2: [bz, dim, num]
        :param return_embedding:
        :return:
        """
        if return_embedding or x2 is None:
            return self.backbone(x1, True)
        x1_out = self.backbone(x1)
        x2_out = self.backbone(x2)
        trans1, trans2 = None, None
        if len(x1_out) == 2:
            out1, wise_x1 = x1_out
            out2, wise_x2 = x2_out
        else:
            out1, wise_x1, trans1 = x1_out
            out2, wise_x2, trans2 = x2_out
        z1 = self.projector(out1)
        p1 = self.g_predictor(z1)
        z2 = self.projector(out2)
        p2 = self.g_predictor(z2)
        loss_w = torch.tensor(0.0, requires_grad=True)
        loss_sim = torch.tensor(0.0, requires_grad=True)
        r_loss = torch.tensor(0.0, requires_grad=True)
        if self.decoder is not None:
            r1_x = chamloss(self.decoder(out1), x1.transpose(1, 2))
            r2_x = chamloss(self.decoder(out2), x2.transpose(1, 2))
            r_loss = (r1_x + r2_x) / 2.0
        if self.l_type in ['gl', 'l']:
            loss_rq = self.vlad(wise_x1, x1)  # [b, k, d]
            loss_rk = self.vlad(wise_x2, x2)  # [b, k, d]
            loss_w = (loss_rq + loss_rk) / 2.0
        if self.l_type in ['gl', 'g']:
            if is_con:
                loss_sim = 10.0 * self.croloss(z1, z2)
            else:
                loss_sim = 15.0 * (loss_fn(p1, z2.detach(), trans1) + loss_fn(p2, z1.detach(), trans2))

        return loss_sim, loss_w + r_loss
