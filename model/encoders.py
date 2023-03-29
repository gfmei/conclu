#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/21/2020 4:05 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : encoders.py
# @Software: PyCharm

import torch
import torch.nn.functional as F
from torch import nn

from torch.autograd import Variable

from model.md_utils import get_graph_feature, sample_and_group_all, sample_and_group, index_points, query_ball_point, \
    farthest_point_sample, square_distance


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss


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

        iden = Variable(torch.eye(3, dtype=torch.float32, device=x.device).view(1, 9)).repeat(B, 1)
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

        iden = Variable(torch.eye(self.k, dtype=torch.float32, device=x.device).view(
            1, self.k ** 2)).repeat(B, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNet(nn.Module):
    def __init__(self, dims=3, is_normal=False, feature_transform=False, detailed=False):
        # last_dims = 1024, , n_rkhs = 512, lg = False):
        super(PointNet, self).__init__()
        in_channel = 6 if is_normal else 3
        self.stn = STN3d(in_channel)  # Batch * 3 * 3
        self.conv1 = nn.Conv1d(in_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, dims, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(dims)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        self.detailed = detailed

    def forward(self, x, is_unsup=False):
        _, D, N = x.size()  # Batch Size, Dimension of Point Features, Num of Points
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            # pdb.set_trace()
            x, feature = x.split([3, D - 3], dim=2)
        xyz = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        # feature = torch.bmm(feature, trans)  # feature -> normals

        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        out1 = self.bn1(self.conv1(x))
        x = F.relu(out1)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None
        out2 = self.bn2(self.conv2(x))
        x = F.relu(out2)
        out3 = self.bn3(self.conv3(x))
        # x = self.bn3(self.conv3(x))
        x = torch.max(out3, 2, keepdim=False)[0]
        if is_unsup:
            return x, [xyz, out3]
        elif self.detailed:
            return out1, out2, out3, x
        # else:  # concatenate global and local feature together
        #     x = x.view(-1, 1024, 1).repeat(1, 1, N) , trans, trans_feat
        #     return torch.cat([x, pointfeat], 1), trans, trans_feat
        else:
            return x


class ClusterNet(nn.Module):
    def __init__(self, dims=512, n_cluster=20):
        super().__init__()
        self.cluster = nn.Sequential(
            nn.Linear(dims, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, n_cluster)
        )

    def forward(self, x):
        return self.cluster(x)


class DGCNN(nn.Module):
    def __init__(self, dims=512, k=20):
        super(DGCNN, self).__init__()
        self.k = k
        self.dims = dims
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False), self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, dims, kernel_size=1, bias=False), self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear = nn.Linear(dims * 2, dims, bias=False)
        self.bn6 = nn.BatchNorm1d(dims)

    def forward(self, x, is_unsup=False):
        batch_size = x.size(0)
        xyz = x
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x5 = x
        xm = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        xa = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((xm, xa), 1)
        x = F.leaky_relu(self.bn6(self.linear(x)), negative_slope=0.2)
        if is_unsup:
            return x, [xyz, x5]
        return x


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points, is_unsup=False):
        """
        Input:
        xyz: input src position data, [B, C, N]
        src: input src data, [B, D, N]
        Return:
        new_xyz: sampled src position data, [B, C, S]
        new_points_concat: sample src feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points, is_unsup)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled src position data, [B, npoint, C]
        # new_points: sampled src data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_xyz = new_xyz.permute(0, 2, 1)
        feature = torch.max(new_points, 2)[0]
        if is_unsup:
            return new_xyz, feature, new_points

        return new_xyz, feature


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input src position data, [B, C, N]
            src: input src data, [B, D, N]
        Return:
            new_xyz: sampled src position data, [B, C, S]
            new_points_concat: sample src feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input src position data, [B, C, N]
            xyz2: sampled input src position data, [B, C, S]
            points1: input src data, [B, D, N]
            points2: input src data, [B, D, S]
        Return:
            new_points: upsampled src data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class SSGEncoder(nn.Module):
    def __init__(self, dims=1024, is_normal=True):
        super(SSGEncoder, self).__init__()
        in_channel = 6 if is_normal else 3
        self.normal_channel = is_normal
        self.dims = dims
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel,
                                          mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3,
                                          mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, dims], group_all=True)

    def forward(self, xyz, is_unsup=False):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        output = self.sa3(l2_xyz, l2_points, is_unsup=is_unsup)
        l3_xyz, feature = output[0], output[1]
        feature = feature.squeeze(-1)
        if is_unsup:
            return feature, [l3_xyz, output[-1]]

        return feature


class SCGEncoder(nn.Module):
    def __init__(self, last_dims=1024, is_normal=True, n_rkhs=512):
        super(SCGEncoder, self).__init__()
        in_channel = 6 if is_normal else 3
        self.normal_channel = is_normal
        self.dims = 3 * n_rkhs
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.23, nsample=48, in_channel=in_channel,
                                          mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3,
                                          mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 512, last_dims], group_all=True)

        self.prediction_modules = nn.ModuleList()
        self.prediction_modules.append(
            nn.Sequential(
                nn.Conv1d(128, n_rkhs, 1),
                nn.BatchNorm1d(n_rkhs),
                nn.ReLU(inplace=True),
                nn.Conv1d(n_rkhs, n_rkhs, 1)
            )
        )

        self.prediction_modules.append(
            nn.Sequential(
                nn.Conv1d(256, n_rkhs, 1),
                nn.BatchNorm1d(n_rkhs),
                nn.ReLU(inplace=True),
                nn.Conv1d(n_rkhs, n_rkhs, 1)
            )
        )

        self.prediction_modules.append(
            nn.Sequential(
                nn.Conv1d(last_dims, n_rkhs, 1),
                nn.BatchNorm1d(n_rkhs),
                nn.ReLU(inplace=True),
                nn.Conv1d(n_rkhs, n_rkhs, 1)
            )
        )
        self.adaptive_maxpool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(n_rkhs * 3, n_rkhs * 3, bias=False)
        self.bn = nn.BatchNorm1d(n_rkhs * 3)

    def forward(self, xyz, is_unsup=False):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        pred1 = F.normalize(self.prediction_modules[0](l1_points), dim=1, p=2)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        pred2 = F.normalize(self.prediction_modules[1](l2_points), dim=1, p=2)
        output = self.sa3(l2_xyz, l2_points, is_unsup=True)
        l3_xyz = output[0]
        pred3 = F.normalize(self.prediction_modules[2](output[-1].squeeze(-1)), dim=1, p=2)
        out = [pred1, pred2, pred3]
        feature = torch.cat([self.adaptive_maxpool(now_out).squeeze(-1) for now_out in out], dim=-1)
        # feature = F.relu(self.bn(self.linear(feature)))
        if is_unsup:
            return feature, [l3_xyz, out]

        return feature


class MSGEncoder(nn.Module):
    def __init__(self, dims=1024, is_normal=True):
        super().__init__()
        in_channel = 3 if is_normal else 0
        self.normal_channel = is_normal
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, dims], True)
        self.dims = dims
        # self.fc1 = nn.Linear(1024, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.drop1 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.drop2 = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz, is_unsup=False):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        output = self.sa3(l2_xyz, l2_points, is_unsup=is_unsup)
        l3_xyz, feature = output[0], output[1]
        feature = feature.squeeze(-1)
        if is_unsup:
            return feature, [l3_xyz, output[-1]]

        return feature