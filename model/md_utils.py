#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/29/2023 4:35 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : md_utils.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None, extra_dim=False):
    batch_size, num_dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if extra_dim is False:
            idx = knn(x, k=k)
        else:
            idx = knn(x[:, 6:], k=k)  # idx = knn(x[:, :3], k=k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx += idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature  # (batch_size, 2 * num_dims, num_points, k)


# STN -> Spatial Transformer Network
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)  # in-channel, out-channel, kernel size
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]  # global descriptors

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = Variable(torch.eye(3, dtype=torch.float32, device=x.device)).view(1, 9).repeat(B, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        B = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(
            torch.eye(self.k, dtype=torch.float32,
                      device=x.device)).view(1, self.k ** 2).repeat(B, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two src.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source src, [B, N, C]
        dst: target src, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        src: input src data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed src data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled point cloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all src, [B, N, 3]
        new_xyz: query src, [B, S, 3]
    Return:
        group_idx: grouped src index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint: sample src number from global point cloud.
        radius:
        nsample: number of local src
        xyz: input src position data, [B, N, 3]
        src: input src data, [B, N, D]
    Return:
        new_xyz: sampled src position data, [B, npoint, nsample, 3]
        new_points: sampled src data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points, unsup=False):
    """
    Input:
        xyz: input src position data, [B, N, 3]
        src: input src data, [B, N, D]
    Return:
        new_xyz: sampled src position data, [B, 1, 3]
        new_points: sampled src data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    if unsup:
        new_xyz = xyz
    else:
        new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class QueryGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of src to gather in the ball
    """

    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(
            self,
            xyz: torch.Tensor,
            new_xyz: torch.Tensor,
            features: torch.Tensor = None):
        r"""
        Parameters
        ----------
        fps_idx
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(
            xyz, idx
        )  # shape = (B, M, C) or shape = (B, M, K, C)
        grouped_xyz = grouped_xyz.permute(0, -1, 1, 2)  # (B, 3, npoint, nsample)
        raw_grouped_xyz = grouped_xyz
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if features is not None:
            grouped_features = index_points(features.transpose(-1, -2), idx)
            grouped_features = grouped_features.permute(0, -1, 1, 2)
            if self.use_xyz:
                new_features = torch.cat([raw_grouped_xyz, grouped_xyz, grouped_features],
                                         dim=1)  # (B, C + 3 + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = torch.cat([raw_grouped_xyz, grouped_xyz], dim=1)

        return new_features


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(
            self,
            xyz: torch.Tensor,
            features: torch.Tensor = None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features],
                                         dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features


def chamfer_loss(src, dst):
    """
    :param src: [bz, m, dim]
    :param dst: [bz, n, dim]
    :return: chamfer_loss for src and dst
    """
    assert src.size(0) == dst.size(0)
    bs, n, dim = dst.size()
    _, m, _ = src.size()
    src = src[:, :, 0:dim]
    x = src.unsqueeze(2)
    y = dst.unsqueeze(1)
    dis = torch.pow(x - y, 2).sum(-1).squeeze()
    return (torch.min(dis, 1)[0].mean() + torch.min(dis, 2)[0].mean()) / 2.0


def cos_loss(src, dst):
    x = F.normalize(src, dim=1, p=2)
    y = F.normalize(dst, dim=1, p=2)
    scores = torch.abs(x * y)
    scores = scores.sum(1)
    dis = 2 - 2 * scores
    return dis.mean()