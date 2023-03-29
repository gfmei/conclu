#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/29/2023 4:39 PM
# @Author  : Guofeng Mei
# @Email   : Guofeng.Mei@student.uts.edu.au
# @File    : main.py
# @Software: PyCharm
import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'data'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from lib.svm_utils import validate
from model.encoders import PointNet, DGCNN, SCGEncoder
from model.model import MLP, SiamCluster
from dataset.dataloader import data_loader


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--model', default='BYOL', help='model name [default: BYOL]')

    parser.add_argument('--dataset', type=str,
                        default='/home/gmei/Data/data/modelnet40_normal_resampled/dataset/',
                        help="dataset path")
    parser.add_argument('--datatype', type=str, default='modelnet', help='[scanobject, modelnet]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--proj_dim', type=int, default=256, help='Project dimension [default: 128]')
    parser.add_argument('--tau', type=float, default=0.01, help='Temperature [default: 0.25]')
    parser.add_argument('--K', type=int, default=64, help='Cluster [default: 64]')
    parser.add_argument('--angle', type=float, default=0.0, help='1.0')
    parser.add_argument('--md', type=str, default='dg', help='[default: dgcnn]')
    parser.add_argument('--crop', type=list, default=[0.85, 0.85], help='crop')
    parser.add_argument('--dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--neighs', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--subset10', action='store_true', default=False,
                        help='Whether to use ModelNet10 [default: False]')
    parser.add_argument('--train', type=bool, default=True, help='Whether to use reconstruction [default: True]')
    parser.add_argument('--unsup', action='store_true', default=True,
                        help='Whether to use reconstruction [default: True]')
    parser.add_argument('--l_type', type=str, default='gl', help='Loss type')
    parser.add_argument('--is_recon', action='store_true', default=False,
                        help='Whether to use self_supervision [default: True]')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--log_dir', type=str, default='DT4LS', help='experiment root')

    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training [default: 64]')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer for training [default: Adam]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='lr decay rate [default: 0.5]')
    parser.add_argument('--aug', type=str, default='jitter', help='augmentation')

    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    return parser.parse_args()


def train(args, logger, train_loader, test_loader, exp='sinkhorn'):
    """MODEL LOADING"""
    dims = args.dims
    l_dim = dims
    if args.md == 'pn':
        net = PointNet(dims=args.dims, is_normal=False, feature_transform=True, feat_type='global')
    elif args.md == 'scg':
        dims = 512 * 3
        l_dim = 512 * 2
        net = SCGEncoder(last_dims=args.dims, is_normal=args.normal, n_rkhs=512)
    elif args.md == 'dg':
        net = DGCNN(dims=dims, k=args.neighs)
    else:
        raise NotImplementedError
    net = net.cuda()
    projector = MLP(in_size=dims, out_size=args.proj_dim).cuda()
    predictor = MLP(in_size=args.proj_dim, out_size=args.proj_dim, hidden_size=512, used='pred').cuda()
    # decoder = FoldingNet(dims, k=32).cuda()
    # decoder = DecoderFC(latent_dim=dims, output_pts=args.num_point).cuda()
    decoder = None
    ema_net = SiamCluster(net, projector, predictor, dim=l_dim, clusters=args.K,
                          tau=args.tau, l_type=args.l_type, decoder=decoder).cuda()
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/svm_best_model.pth')
        ema_net.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except Exception as e:
        logger.info('No existing model, starting training from scratch {}'.format(e))
    start_epoch = 0
    global_epoch = 0
    global_step = 0
    best_loss = float('inf')
    best_acc = 0.0
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(ema_net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=args.decay_rate)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(ema_net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                                      eps=1e-08, weight_decay=args.decay_rate, amsgrad=False)
    else:
        optimizer = torch.optim.SGD(ema_net.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=args.lr_decay)
    is_con = False
    writer = SummaryWriter(str(checkpoints_dir) + '/log')
    if exp == 'contras':
        is_con = True
    for epoch in range(start_epoch, args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        train_loss = []
        s_loss = list()
        for batch_id, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
            x1, x2, target = data
            x1 = x1.cuda()
            x1 = x1.transpose(2, 1)
            x2 = x2.cuda()
            x2 = x2.transpose(2, 1)
            optimizer.zero_grad()
            ema_net = ema_net.train()
            g_loss, l_loss = ema_net(x1, x2, is_con=is_con)
            loss = g_loss + 0.5 * l_loss
            loss.backward()
            optimizer.step()
            global_step += 1
            s_loss.append(l_loss.item())
            train_loss.append(loss.item())
            niter = epoch * len(train_loader) + batch_id
            writer.add_scalars('PointNet_Loss', {'train_loss': loss.data.item()}, niter)
        mean_loss = np.mean(train_loss)
        scheduler.step()
        logger.info('Train mean loss:{}, assistant loss: {} \n'.format(mean_loss, np.mean(s_loss)))
        print('Train mean loss:{}, assistant loss: {}'.format(mean_loss, np.mean(s_loss)))
        logger.info('Save model...')
        savepath = str(checkpoints_dir) + '/self_best_model.pth'
        logger.info('Saving at %s' % savepath)
        print('Saving at %s' % savepath)
        state = {
            'best_loss': best_loss,
            'model_state_dict': ema_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)
        if epoch > 100:
            acc = validate(train_loader, test_loader, ema_net, best_acc, logger, 'lsvc')
            if acc > best_acc:
                best_acc = acc
                savepath = str(checkpoints_dir) + '/svm_best_model.pth'
                state = {
                    'best_loss': best_loss,
                    'model_state_dict': ema_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
        k = 10
        if args.md == 'pn' or args.l_type == 'l':
            k = 5
        if (epoch + 1) % k == 0 and epoch <= 100:
            acc = validate(train_loader, test_loader, ema_net, best_acc, logger, 'lsvc')
            if acc > best_acc:
                best_acc = acc
                savepath = str(checkpoints_dir) + '/svm_best_model.pth'
                state = {
                    'best_loss': best_loss,
                    'model_state_dict': ema_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    '''CREATE DIR'''
    times = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    exp = 'sinkhorn'
    experiment_dir = experiment_dir.joinpath(exp)
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(times)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    '''DATA LOADING'''
    logger.info(args)
    logger.info('Load dataset ...')
    train_data, test_data = data_loader(args)
    train(args, logger, train_data, test_data, exp)
