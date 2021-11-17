import os
import random
import itertools
import os
import tqdm
import random
import cv2
import sys
import numpy as np
import h5py
import torch
from torch.autograd import Variable
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from torch_geometric.data import Data
from progressbar import ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data import PhysicsDataset
from config import gen_args
from utils import load_data
from meshnet_modules import MeshGraph

trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
seed = 9
torch.manual_seed(seed)

args = gen_args()
print(args.dataf)
datasets = {}
data_n_batches = {}
dataloaders = {}

for phase in ['train', 'valid']:
    datasets[phase] = PhysicsDataset(args, phase=phase, trans_to_tensor=trans_to_tensor)

    if args.gen_data:
        datasets[phase].gen_data()
    else:
        datasets[phase].load_data()

    dataloaders[phase] = DataLoader(
        datasets[phase], batch_size=args.batch_size,
        shuffle=True if phase == 'train' else True,
        num_workers=args.num_workers)

    data_n_batches[phase] = len(dataloaders[phase])



if __name__ == '__main__':
    save_folder = args.log
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_path = save_folder + '/model'
    writer = SummaryWriter('{}/'.format(save_folder))
    phase = 'train'
    bar = ProgressBar(max_value=data_n_batches[phase])
    meshnet = MeshGraph(args=args, outputsize=2, latent_size=128, number_layers=2, message_passing_steps=args.num_message_passing)
    optimizer = optim.Adam(list(meshnet.parameters()), lr=1e-4)
    log_per_iter = 100
    val_iter = 500
    train_loader = dataloaders['train']
    val_loader = dataloaders['valid']
    print(len(val_loader),'val len',len(train_loader))
    best_nll = 1e20
    for phase in ['train']:
        loader = train_loader
        train_iter = -1
        for train_i, data in bar(enumerate(loader)):
            
            train_iter += 1
            kps_gt, edge_type, edge_attr = data
            meshnet.train()
            optimizer.zero_grad()
            pair = []
            filtered_edge_attr = []
            cnt = 0
            for x in range(args.n_kp):
                for y in range(x + 1, args.n_kp):
                    this_edge = edge_type[0, cnt]
                    this_attr = edge_attr[0, cnt]
                    if this_edge != 0:
                        pair.append([x, y])
                        filtered_edge_attr.append([this_edge, this_attr])
                cnt += 1

            pair = torch.tensor(pair)
            if len(filtered_edge_attr) < 2:
                continue
            filtered_edge_attr = torch.tensor(filtered_edge_attr)

            output, loss_nll, loss_mse, loss_acc = meshnet(kps_gt, pair, filtered_edge_attr)
            loss_acc.backward()
            optimizer.step()
            if train_iter % log_per_iter == 0:
                writer.add_scalar("Loss/nll_train", loss_nll.item(), train_iter)
                writer.add_scalar("Loss/mse_train", loss_mse.item(), train_iter)
            if train_iter % val_iter == 0:
                nll_eval = []
                mse_eval = []
                acc_mse_eval = []
                meshnet.eval()
                for eval_i, data in enumerate(val_loader):
                    if eval_i > 100:
                        break
                    kps_gt, edge_type, edge_attr = data
                    pair = []
                    filtered_edge_attr = []
                    cnt = 0
                    for x in range(args.n_kp):
                        for y in range(x + 1, args.n_kp):
                            this_edge = edge_type[0, cnt]
                            this_attr = edge_attr[0, cnt]
                            if this_edge != 0:
                                pair.append([x, y])
                                filtered_edge_attr.append([this_edge, this_attr])
                        cnt += 1
                    pair = torch.tensor(pair)
                    if len(filtered_edge_attr) < 2:
                        continue
                    filtered_edge_attr = torch.tensor(filtered_edge_attr)
                    output, loss_nll_eval, loss_mse_eval, loss_acc_eval = meshnet(kps_gt, pair, filtered_edge_attr)
                    nll_eval.append(loss_nll_eval.item())
                    mse_eval.append(loss_mse_eval.item())
                    acc_mse_eval.append(loss_acc_eval.item())
                writer.add_scalar("Eval/nll", np.mean(nll_eval), train_iter)
                writer.add_scalar("Eval/mse", np.mean(mse_eval), train_iter)
                writer.add_scalar("Eval/mse_acc", np.mean(acc_mse_eval), train_iter)
                print('iter:', train_iter, 'nll:', np.mean(nll_eval), 'mse:', np.mean(mse_eval), 'accmse:', np.mean(acc_mse_eval))
                if np.mean(nll_eval) < best_nll:
                    best_nll = np.mean(nll_eval)
                    torch.save(meshnet.state_dict(), model_path)
                    print('model saved!')

