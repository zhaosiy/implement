import os
import random
import itertools
import os
import random
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle, Polygon
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

for phase in ['valid']:
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


def render(states, rels, video=True, image=False, path=None, draw_edge=True,
           lim=(-80, 80, -80, 80), verbose=True, st_idx=0, image_prefix='fig'):
    # states: time_step x n_ball x 4
    # rel: (10) x 1
    # lim = (lim[0] - self.radius, lim[1] + self.radius, lim[2] - self.radius, lim[3] + self.radius)

    if video:
        video_path = path + '.avi'
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        if verbose:
            print('Save video as %s' % video_path)
        out = cv2.VideoWriter(video_path, fourcc, 25, (110, 110))

    if image:
        image_path = path
        if verbose:
            print('Save images to %s' % image_path)
        command = 'mkdir -p %s' % image_path
        os.system(command)

    c = ['royalblue', 'tomato', 'limegreen', 'orange', 'violet', 'chocolate', 'black', 'crimson']

    time_step = states.shape[0]
    n_ball = states.shape[1]
    print(time_step,'time steps', n_ball, 'number of balls')
    for i in range(0, time_step):
        fig, ax = plt.subplots(1)
        plt.xlim(lim[0], lim[1])
        plt.ylim(lim[2], lim[3])
        # plt.axis('off')

        fig.set_size_inches(1.5, 1.5)
        if draw_edge:
            param = rels

            # draw edge
            cnt = 0
            for x in range(n_ball):
                for y in range(x):
                    rel_type = int(param[cnt])
                    cnt += 1
                    if rel_type == 0:
                        continue

                    plt.plot([states[i, x, 0], states[i, y, 0]],
                             [states[i, x, 1], states[i, y, 1]],
                             '-', color=c[rel_type], lw=1, alpha=0.5)

        circles = []
        circles_color = []
        for j in range(n_ball):
            circle = Circle((states[i, j, 0], states[i, j, 1]), radius=6)
            circles.append(circle)
            circles_color.append(c[j % len(c)])

        pc = PatchCollection(circles, facecolor=circles_color, linewidth=0, alpha=0.5)
        ax.add_collection(pc)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.tight_layout()
        if video or image:
            fig.canvas.draw()
            frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = frame[21:-19, 21:-19]

        if video:
            out.write(frame)
            if i == time_step - 1:
                for _ in range(5):
                    out.write(frame)

        if image:
            plt.savefig(os.path.join(image_path, '%s_%s.png' % (image_prefix, i + st_idx)))
        plt.close()

    if video:
        out.release()

def render_overlay(pred, states, rels, video=True, image=False, path=None, draw_edge=True,
           lim=(-80, 80, -80, 80), verbose=True, st_idx=0, image_prefix='fig'):
    # states: time_step x n_ball x 4
    # rel: (10) x 1
    # lim = (lim[0] - self.radius, lim[1] + self.radius, lim[2] - self.radius, lim[3] + self.radius)
    draw_edge = 0
    if video:
        video_path = path + '.avi'
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        if verbose:
            print('Save video as %s' % video_path)
        out = cv2.VideoWriter(video_path, fourcc, 25, (110, 110))

    if image:
        image_path = path
        if verbose:
            print('Save images to %s' % image_path)
        command = 'mkdir -p %s' % image_path
        os.system(command)

    c = ['royalblue', 'tomato', 'limegreen', 'orange', 'violet', 'chocolate', 'black', 'crimson']

    time_step = states.shape[0]
    n_ball = states.shape[1]
    print(time_step,'time steps', n_ball, 'number of balls')
    for i in range(0, time_step):
        fig, ax = plt.subplots(1)
        plt.xlim(lim[0], lim[1])
        plt.ylim(lim[2], lim[3])
        # plt.axis('off')

        fig.set_size_inches(1.5, 1.5)
        if draw_edge:
            param = rels

            # draw edge
            cnt = 0
            for x in range(n_ball):
                for y in range(x):
                    rel_type = int(param[cnt])
                    cnt += 1
                    if rel_type == 0:
                        continue

                    plt.plot([states[i, x, 0], states[i, y, 0]],
                             [states[i, x, 1], states[i, y, 1]],
                             '-', color=c[rel_type], lw=1, alpha=0.5)

        circles_p = []
        circles_color_p = []
        circles_gt = []
        circles_color_gt = []
        for j in range(n_ball):
            circle_gt = Circle((states[i, j, 0], states[i, j, 1]), radius=6)
            circles_gt.append(circle_gt)
            circle = Circle((pred[i, j, 0], pred[i, j, 1]), radius=3)
            circles_p.append(circle)
            circles_color_p.append(c[j % len(c)])
            circles_color_gt.append(c[j % len(c)])

        pc = PatchCollection(circles_p, facecolor=circles_color_p, linewidth=0, alpha=0.5)
        ax.add_collection(pc)
        pc_gt = PatchCollection(circles_gt, facecolor=circles_color_gt, linewidth=0, alpha=0.5)
        ax.add_collection(pc_gt)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.tight_layout()
        if video or image:
            fig.canvas.draw()
            frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = frame[21:-19, 21:-19]

        if video:
            out.write(frame)
            if i == time_step - 1:
                for _ in range(5):
                    out.write(frame)

        if image:
            plt.savefig(os.path.join(image_path, '%s_%s.png' % (image_prefix, i + st_idx)))
        plt.close()

    if video:
        out.release()

if __name__ == '__main__':
    args.pred_roll = 20
    meshnet = MeshGraph(args=args, outputsize=2, latent_size=128, number_layers=2, message_passing_steps=2)
    model_path = '1_step_gtacc/model'
    eval_path = 'eval_gt'
    eval_pathp = 'eval_pred'
    meshnet.load_state_dict(torch.load(model_path))
    bar = ProgressBar(max_value=data_n_batches[phase])
    val_loader = dataloaders['valid']
    meshnet.eval()
    for eval_i, data in bar(enumerate(val_loader)):
        if eval_i >= 1:
            s = s
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

        output, loss_nll_eval, loss_mse_eval, gt_future = meshnet(kps_gt, pair, filtered_edge_attr)

        #render(gt_future.detach().numpy() * 80, edge_type[0].detach().numpy(), path=eval_path, video=False, image=True)
        #render(output[0].detach().numpy() * 80, edge_type[0].detach().numpy(), path=eval_pathp, video=False, image=True)
        render_overlay(output[0].detach().numpy() * 80, gt_future.detach().numpy() * 80, edge_type[0].detach().numpy(), path=eval_path, video=False, image=True)
        print(loss_mse_eval,'mse')