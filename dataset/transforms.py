#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/29/2023 4:46 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : transforms.py
# @Software: PyCharm
import math
import random

import numpy as np
import torch
from sklearn.neighbors import KDTree

import so3


def pc_normalize(tensor, zero_mean=True):
    if zero_mean:
        m = tensor.mean(dim=0, keepdim=True)  # [N, D] -> [1, D]
        v = tensor - m
    else:
        v = tensor
    nn = v.norm(p=2, dim=1)  # [N, D] -> [N]
    nmax = torch.max(nn)
    return v / nmax


def pc_normalize_np(pc):
    centroid = np.mean(pc, axis=0)
    pc -= centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class RandomRotation(object):
    def __init__(self, mag=None, randomly=True):
        self.mag = math.pi if mag is None else mag
        self.randomly = randomly
        self.igt = None

    def __call__(self, tensor):
        # tensor: [N, 3]
        amp = torch.rand(1) if self.randomly else 1.0
        w = torch.randn(1, 3)
        w = w / w.norm(p=2, dim=1, keepdim=True) * amp * self.mag

        g = so3.exp(w).to(tensor)  # [1, 3, 3]
        self.igt = g.squeeze(0)  # [3, 3]

        p1 = so3.transform(g, tensor)  # [1, 3, 3] x [N, 3] -> [N, 3]
        return p1


class Jitter(object):
    def __init__(self, sigma=0.001, clip=0.005):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, points):
        N, C = points.shape
        points += np.clip(self.sigma * np.random.randn(N, C), -1 * self.clip, self.clip)
        return points


class KNNTrans(object):
    def __init__(self, knn=16, metric="euclidean"):
        """
        :param knn: default=16
        :param metric: distance type
        """
        self.knn = knn
        self.metric = metric

    def __call__(self, points):
        """
        Args:
        :param points: Nx3
        """
        assert (self.knn > 0)
        kdt = KDTree(points, leaf_size=30, metric=self.metric)
        # nbs[0]:NN distance,N*17. nbs[1]:NN index,N*17
        dist, idx = kdt.query(points, k=self.knn + 1, return_distance=True)
        trans_pts = np.take(points, idx[:, 1: self.knn], axis=0).mean(1)
        return trans_pts


class MixUp(object):
    def __init__(self, knn=16, metric="euclidean"):
        """
        :param knn: default=16
        :param metric: distance type
        """
        self.knn = knn
        self.metric = metric

    def __call__(self, src, tgt):
        """
        Args:
        :param src: Nx3
        """
        assert (self.knn > 0)
        src_kdt = KDTree(src, leaf_size=30, metric=self.metric)
        # nbs[0]:NN distance,N*17. nbs[1]:NN index,N*17
        p1 = src[random.choice(range(src.shape[0]))].reshape(1, -1)
        _, src_idx = src_kdt.query(p1, k=self.knn + 1, return_distance=True)
        tgt_kdt = KDTree(tgt, leaf_size=30, metric=self.metric)
        p2 = src[random.choice(range(tgt.shape[0]))].reshape(1, -1)
        _, tgt_idx = tgt_kdt.query(p2, k=self.knn + 1, return_distance=True)
        src[src_idx[:, 1: self.knn]] = tgt[tgt_idx[:, 1: self.knn]]
        return src


class KNNJitter(object):
    def __init__(self, sigma=0.001, clip=0.005, knn=4, metric="euclidean"):
        self.knn_trans = KNNTrans(knn, metric)
        self.jitter = Jitter(sigma, clip)

    def __call__(self, points):
        points = self.jitter(self.knn_trans(points))
        return points


def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere
    Source: https://gist.github.com/andrewbolster/10274979
    Args:
        num: Number of vectors to sample (or None if single)
    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)
    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)


class RandomCrop(object):
    """Randomly crops the *source* point cloud, approximately retaining half the src
    A direction is randomly sampled from S2, and we retain src which lie within the
    half-space oriented in this direction.
    If p_keep != 0.5, we shift the plane until approximately p_keep src are retained
    """

    def __init__(self, p_keep, multi=None):
        if p_keep is None:
            p_keep = [0.9, 0.9]  # Crop both clouds to 90%
        self.p_keep = np.array(p_keep, dtype=np.float32)

    @staticmethod
    def crop(points, p_keep):
        rand_xyz = uniform_2_sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane > 0
        else:
            mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

        return points[mask, :]

    def __call__(self, sample):

        if np.all(self.p_keep == 1.0):
            return sample  # No need crop
        if len(self.p_keep) == 1:
            sample['src'] = self.crop(sample['src'], self.p_keep[0])
        else:
            sample['src'] = self.crop(sample['src'], self.p_keep[0])
            sample['ref'] = self.crop(sample['ref'], self.p_keep[1])
        return sample


class RRR(object):
    def __init__(self):
        pass

    def r(self):
        phi = np.random.uniform(0.0, 2 * np.pi)
        return phi

    def __call__(self):
        return self.r(), self.r()


def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = batch_data.shape
    normal_data = np.zeros((B, N, C))
    for b in range(B):
        pc = batch_data[b]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        normal_data[b] = pc
    return normal_data


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:, idx, :]


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_with_normal(batch_xyz_normal):
    ''' Randomly rotate XYZ, normal point cloud.
        Input:
            batch_xyz_normal: B,N,6, first three channels are XYZ, last 3 all normal
        Output:
            B,N,6, rotated XYZ, normal point cloud
    '''
    for k in range(batch_xyz_normal.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_xyz_normal[k, :, 0:3]
        shape_normal = batch_xyz_normal[k, :, 3:6]
        batch_xyz_normal[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        batch_xyz_normal[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return batch_xyz_normal


def rotate_perturbation_point_cloud_with_normal(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx6 array, original batch of point clouds and point normals
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, 0:3]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx6 array, original batch of point clouds with normal
          scalar, angle of rotation
        Return:
          BxNx6 array, rotated batch of point clouds iwth normal
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc


if __name__ == '__main__':
    # Visualize sampling
    # from vtk_visualizer.plot3d import plotxyz
    # rand_2s = uniform_2_sphere(10000)
    # plotxyz(rand_2s, block=True)
    # print(rand_2s.shape)
    rrr = RRR()
    s = rrr()
    print(s[0], s[1])
