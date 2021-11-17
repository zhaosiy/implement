import os
import random
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from progressbar import ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.distributions.multivariate_normal import MultivariateNormal

from config import gen_args
from data import PhysicsDataset, load_data
from models_kp import KeyPointNet
from models_dy import DynaNetGNN, HLoss
from utils import rand_int, count_parameters, Tee, AverageMeter, get_lr, to_np, set_seed

args = gen_args()
set_seed(args.random_seed)

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

os.system('mkdir -p ' + args.outf_kp)
os.system('mkdir -p ' + args.dataf)

if args.stage == 'dy':
    os.system('mkdir -p ' + args.outf_dy)
    tee = Tee(os.path.join(args.outf_dy, 'train.log'), 'w')
else:
    raise AssertionError("Unsupported env %s" % args.stage)

print(args)

# generate data
trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

datasets = {}
dataloaders = {}
data_n_batches = {}
for phase in ['train', 'valid']:
    datasets[phase] = PhysicsDataset(args, phase=phase, trans_to_tensor=trans_to_tensor)

    if args.gen_data:
        datasets[phase].gen_data()
    else:
        datasets[phase].load_data()

    dataloaders[phase] = DataLoader(
        datasets[phase], batch_size=args.batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=1)

    data_n_batches[phase] = len(dataloaders[phase])

args.stat = datasets['train'].stat

use_gpu = torch.cuda.is_available()


if args.stage == 'dy':

    if args.dy_model == 'gnn':
        model_dy = DynaNetGNN(args, use_gpu=False)
    else:
        raise AssertionError("Unknown dy_model %s" % args.dy_model)

    print("model_dy #params: %d" % count_parameters(model_dy))

    if args.dy_epoch >= 0:
        # if resume from a pretrained checkpoint
        model_dy_path = os.path.join(
            args.outf_dy, 'net_dy_epoch_%d_iter_%d.pth' % (args.dy_epoch, args.dy_iter))
        print("Loading saved ckp for dynamics net from %s" % model_dy_path)
        model_dy.load_state_dict(torch.load(model_dy_path))
else:
    raise AssertionError("Unknown stage %s" % args.stage)


# criterion
criterionMSE = nn.MSELoss()
criterionH = HLoss()

# optimizer
if args.stage == 'dy':
    params = model_dy.parameters()
else:
    raise AssertionError('Unknown stage %s' % args.stage)

optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, 0.999))
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=2, verbose=True)

if args.stage == 'dy':
    st_epoch = args.dy_epoch if args.dy_epoch > 0 else 0
    log_fout = open(os.path.join(args.outf_dy, 'log_st_epoch_%d.txt' % st_epoch), 'w')
else:
    raise AssertionError("Unknown stage %s" % args.stage)


best_valid_loss = np.inf
if __name__ == '__main__':
    save_folder = 'vcdn_log'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    writer = SummaryWriter('{}/'.format(save_folder))
    for epoch in range(st_epoch, args.n_epoch):
        phases = ['train', 'valid'] if args.eval == 0 else ['valid']

        for phase in phases:
            meter_loss = AverageMeter()
            meter_loss_contras = AverageMeter()

            if args.stage == 'dy':
                model_dy.train(phase == 'train')
                meter_loss_rmse = AverageMeter()
                meter_loss_kp = AverageMeter()
                meter_loss_H = AverageMeter()
                meter_acc = AverageMeter()
                meter_cor = AverageMeter()
                meter_num_edge_per_type = np.zeros(args.edge_type_num)

            bar = ProgressBar(max_value=data_n_batches[phase])
            loader = dataloaders[phase]

            for i, data in bar(enumerate(loader)):


                with torch.set_grad_enabled(phase == 'train'):
                    if args.stage == 'dy':
                        '''
                        hyperparameter on the length of data
                        '''
                        n_his, n_kp = args.n_his, args.n_kp
                        n_samples = args.n_identify + args.n_his + args.n_roll
                        n_identify = args.n_identify

                        '''
                        load data
                        '''
                        if args.env in ['Ball']:
                            # if using detected keypoints
                            if args.preload_kp == 1:
                                # if using preloaded keypoints
                                kps_gt, graph_gt = data

                            B = kps_gt.size(0)


                        '''
                        get detected keypoints -- kps
                        '''
                        # kps: B x (n_identify + n_his + n_roll) x n_kp x 2
                        kps = kps_gt
                        #print(kps.shape,'kps')

                        kps = kps.view(B, n_samples, n_kp, 2)
                        #print(kps.shape, '2')
                        kps_id, kps_dy = kps[:, :n_identify], kps[:, n_identify:]
                        #print(kps_id.shape, kps_dy.shape, 'kps') [24, 30, 5, 2]
                        # only train dynamics module
                        kps = kps.detach()

                        # step #2: dynamics prediction
                        eps = args.gauss_std
                        kp_cur = kps_dy[:, :n_his].view(B, n_his, n_kp, 2)
                        #print(kp_cur.shape,'2') [24, 10, 5, 2]
                        covar_gt = torch.FloatTensor(np.array([eps, 0., 0., eps]))
                        covar_gt = covar_gt.view(1, 1, 1, 4).repeat(B, n_his, n_kp, 1)
                        #print(covar_gt.shape,'cova') [24, 10, 5, 4]
                        kp_cur = torch.cat([kp_cur, covar_gt], 3)

                        loss_kp = 0.
                        loss_mse = 0.
                        # graph_gt:
                        #   edge_type_gt: B x n_kp x n_kp x edge_type_num
                        #   edge_attr_gt: B x n_kp x n_kp x edge_attr_dim

                        for j in range(args.n_roll):

                            kp_des = kps_dy[:, n_his + j]

                            if args.dy_model == 'gnn':
                                # kp_pred: B x n_kp x 2
                                kp_pred = model_dy.dynam_prediction(kp_cur, graph_gt, env=args.env)
                                mean_cur, covar_cur = kp_pred[:, :, :2], kp_pred[:, :, 2:].view(B, n_kp, 2, 2)

                                mean_des, covar_des = kp_des, covar_gt[:, 0].view(B, n_kp, 2, 2)

                                m_cur = MultivariateNormal(mean_cur, scale_tril=covar_cur)
                                m_des = MultivariateNormal(mean_des, scale_tril=covar_des)

                                log_prob = (m_cur.log_prob(kp_des) - m_des.log_prob(kp_des)).mean()
                                # log_prob = m_cur.log_prob(kp_des).mean()

                                loss_kp_cur = -log_prob * args.lam_kp
                                # loss_kp_cur = criterionMSE(mean_cur, mean_des) * args.lam_kp
                                # print(criterionMSE(mean_cur, mean_des) * args.lam_kp)
                                loss_kp += loss_kp_cur / args.n_roll

                                loss_mse_cur = criterionMSE(mean_cur, mean_des)
                                loss_mse += loss_mse_cur / args.n_roll

                            # update feat_cur and hmap_cur
                            kp_cur = torch.cat([kp_cur[:, 1:], kp_pred.unsqueeze(1)], 1)

                        # summarize the losses
                        loss = loss_kp

                        # update meter
                        meter_loss_rmse.update(np.sqrt(loss_mse.item()), B)
                        meter_loss_kp.update(loss_kp.item(), B)
                        meter_loss.update(loss.item(), B)
                        if i % 15 == 0:
                            writer.add_scalar("Loss/nll_train", loss_kp.item(), i + epoch * len(loader))
                            writer.add_scalar("Loss/mse_train", loss_mse.item(), i + epoch * len(loader))


                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if phase == 'train' and i % args.ckp_per_iter == 0:
                    if args.stage == 'dy':
                        torch.save(model_dy.state_dict(), '%s/net_dy_epoch_%d_iter_%d.pth' % (args.outf_dy, epoch, i))

            if phase == 'valid' and not args.eval:
                scheduler.step(meter_loss.avg)
                if meter_loss.avg < best_valid_loss:
                    best_valid_loss = meter_loss.avg

                    if args.stage == 'dy':
                        torch.save(model_dy.state_dict(), '%s/net_best_dy.pth' % (args.outf_dy))
                        print('saved model!')



