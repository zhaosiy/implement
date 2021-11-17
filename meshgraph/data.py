import multiprocessing as mp
import os
import time

from PIL import Image

import cv2
import numpy as np
import imageio
import scipy.misc
import torch
from progressbar import ProgressBar
from torch.autograd import Variable
from torch.utils.data import Dataset
#from torch_geometric.loader import DataLoader
from physics_engine import BallEngine, ClothEngine

from utils import rand_float, rand_int
from utils import init_stat, combine_stat, load_data, store_data
from utils import resize, crop
from utils import adjust_brightness, adjust_saturation, adjust_contrast, adjust_hue
#from torch_geometric.data import Data


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def normalize(data, stat, var=False):
    for i in range(len(stat)):
        stat[i][stat[i][:, 1] == 0, 1] = 1.0
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i]).cuda())
            data[i] = (data[i] - s[:, 0]) / s[:, 1]
    else:
        for i in range(len(stat)):
            data[i] = (data[i] - stat[i][:, 0]) / stat[i][:, 1]
    return data


def denormalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i]).cuda())
            data[i] = data[i] * s[:, 1] + s[:, 0]
    else:
        for i in range(len(stat)):
            data[i] = data[i] * stat[i][:, 1] + stat[i][:, 0]
    return data


def get_crop_params(phase, img, crop_size):
    w, h = img.size

    if w < h:
        tw = crop_size
        th = int(crop_size * h / w)
    else:
        th = crop_size
        tw = int(crop_size * w / h)

    if phase == 'train':
        if w == tw and h == th:
            return 0, 0, h, w
        assert False
        i = rand_int(0, h - th)
        j = rand_int(0, w - tw)

    else:
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

    return i, j, th, tw


def resize_and_crop(phase, src, scale_size, crop_size):
    # resize the images
    src = resize(src, scale_size)

    # crop the images
    crop_params = get_crop_params(phase, src, crop_size)
    src = crop(src, crop_params[0], crop_params[1], crop_params[2], crop_params[3])

    return src


def default_loader(path):
    return pil_loader(path)


def gen_Ball(info):
    thread_idx, data_dir, data_names = info['thread_idx'], info['data_dir'], info['data_names']
    n_rollout, time_step = info['n_rollout'], info['time_step']
    dt, video, args, phase = info['dt'], info['video'], info['args'], info['phase']
    n_ball = info['n_ball']

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    attr_dim = args.attr_dim  # radius
    state_dim = args.state_dim  # x, y, xdot, ydot
    action_dim = 2  # ddx, ddy

    stats = [init_stat(attr_dim), init_stat(state_dim), init_stat(action_dim)]

    engine = BallEngine(dt, state_dim, action_dim=2)

    bar = ProgressBar()
    for i in bar(range(n_rollout)):

        rollout_idx = thread_idx * n_rollout + i
        rollout_dir = os.path.join(data_dir, str(rollout_idx))
        os.system('mkdir -p ' + rollout_dir)

        engine.init(n_ball)

        n_obj = engine.num_obj
        attrs_all = np.zeros((time_step, n_obj, attr_dim))
        states_all = np.zeros((time_step, n_obj, state_dim))
        actions_all = np.zeros((time_step, n_obj, action_dim))
        rel_attrs_all = np.zeros((time_step, engine.param_dim, 2))

        act = np.zeros((n_obj, 2))
        for j in range(time_step):
            # print(j,'time step',rollout_idx,'rollout')
            state = engine.get_state()

            vel_dim = state_dim // 2
            pos = state[:, :vel_dim]
            vel = state[:, vel_dim:]

            if j > 0:
                vel = (pos - states_all[j - 1, :, :vel_dim]) / dt

            attrs = np.zeros((n_obj, attr_dim))
            attrs[:] = engine.radius

            attrs_all[j] = attrs
            states_all[j, :, :vel_dim] = pos
            states_all[j, :, vel_dim:] = vel
            # print(engine.param,'params')
            rel_attrs_all[j] = engine.param
            # if j!=0 and j % 88 == 0:
            #    print(engine.param[:,0])
            # print(pos.shape, vel.shape, rel_attrs_all[j].shape) (5, 2) (5, 2) (10, 2)
            # print(states_all.shape,'all', rel_attrs_all.shape) (500, 5, 4) (500, 10, 2)

            act += (np.random.rand(n_obj, 2) - 0.5) * 600 - act * 0.1 - state[:, 2:] * 0.1
            act = np.clip(act, -1000, 1000)
            engine.step(j, act)

            actions_all[j] = act.copy()

        datas = [attrs_all, states_all, actions_all, rel_attrs_all]
        # print(attrs_all.shape, states_all.shape, actions_all.shape, rel_attrs_all.shape)
        # (500, 5, 1)(500, 5, 4)(500, 5, 2)(500, 10, 2)

        store_data(data_names, datas, rollout_dir + '_new.h5')
        if 0:
            engine.render(states_all, actions_all, rel_attrs_all, video=False, image=True,
                          path='simple', draw_edge=True, verbose=True)

        datas = [datas[i].astype(np.float64) for i in range(len(datas))]

        for j in range(len(stats)):
            stat = init_stat(stats[j].shape[0])
            stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
            stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
            stat[:, 2] = datas[j].shape[0]
            stats[j] = combine_stat(stats[j], stat)

    return stats


class PhysicsDataset(Dataset):

    def __init__(self, args, phase, trans_to_tensor=None, loader=default_loader):
        self.args = args
        self.phase = phase
        self.trans_to_tensor = trans_to_tensor
        self.loader = loader

        self.data_dir = os.path.join(self.args.dataf, phase)
        # print(self.args.dataf,'dataf')
        self.stat_path = os.path.join(self.args.dataf, 'stat.h5')
        self.stat = None

        os.system('mkdir -p ' + self.data_dir)

        if args.env in ['Ball']:
            self.data_names = ['attrs', 'states', 'actions', 'rels']
        elif args.env in ['Cloth']:
            self.data_names = ['states', 'actions', 'scene_params']
        else:
            raise AssertionError("Unknown env")

        ratio = self.args.train_valid_ratio
        if phase in {'train'}:
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase in {'valid'}:
            self.n_rollout = self.args.n_rollout - int(self.args.n_rollout * ratio)
        else:
            raise AssertionError("Unknown phase")

        self.T = self.args.time_step
        self.scale_size = args.scale_size
        self.crop_size = args.crop_size

    def load_data(self):
        # print(self.stat_path,'path')
        self.stat = load_data(self.data_names, self.stat_path)

    def gen_data(self):
        # if the data hasn't been generated, generate the data
        n_rollout, time_step, dt = self.n_rollout, self.args.time_step, self.args.dt
        assert n_rollout % self.args.num_workers == 0

        print("Generating data ... n_rollout=%d, time_step=%d" % (n_rollout, time_step))

        infos = []
        for i in range(self.args.num_workers):
            info = {'thread_idx': i,
                    'data_dir': self.data_dir,
                    'data_names': self.data_names,
                    'n_rollout': n_rollout // self.args.num_workers,
                    'time_step': time_step,
                    'dt': dt,
                    'video': False,
                    'phase': self.phase,
                    'args': self.args,
                    'vis_height': self.args.height_raw,
                    'vis_width': self.args.width_raw}

            if self.args.env in ['Ball']:
                info['env'] = 'Ball'
                info['n_ball'] = self.args.n_ball
            elif self.args.env in ['Cloth']:
                info['env'] = 'Cloth'
                info['env_idx'] = 15

            infos.append(info)

        cores = self.args.num_workers
        pool = mp.Pool(processes=cores)

        env = self.args.env

        if env in ['Ball']:
            data = pool.map(gen_Ball, infos)

        print("Training data generated, warpping up stats ...")

        if self.phase == 'train':
            if env in ['Ball']:
                self.stat = [init_stat(self.args.attr_dim),
                             init_stat(self.args.state_dim),
                             init_stat(self.args.action_dim)]

            for i in range(len(data)):
                for j in range(len(self.stat)):
                    self.stat[j] = combine_stat(self.stat[j], data[i][j])

            store_data(self.data_names[:len(self.stat)], self.stat, self.stat_path)

        else:
            print("Loading stat from %s ..." % self.stat_path)
            self.stat = load_data(self.data_names, self.stat_path)

    def __len__(self):
        args = self.args
        length = self.n_rollout * int(args.time_step // (args.his + args.pred_roll))
        length = int(length)
        
        return length

    def __getitem__(self, idx):
        '''
        return: 1. key points for 130 time steps, only x, y, and y is multiplied by -1.
                2. ground truth edge type and length.
        '''
        args = self.args
        suffix = '.png' if args.env in ['Ball'] else '.jpg'
        window = args.his + args.pred_roll
        per_roll = int(args.time_step / (args.his + args.pred_roll))
        src_rollout = int(idx / per_roll)
        index_roll = int(idx % per_roll)
        '''
        used for dynamics modeling
        '''
        if args.stage in 'dy':

            # using detected keypoints
            if args.preload_kp == 1:
                # if using preload keypoints

                data_path = os.path.join(args.dataf, self.phase, str(src_rollout) + '_new.h5')
                metadata = load_data(self.data_names, data_path)

                edge_type = metadata[3][0, :, 0].astype(np.int)
                edge_attr = metadata[3][0, :, 1:]

                # get ground truth keypoint position
                states = metadata[1]
                states[:,:,:2] = states[:, :, :2] / 80.
                states[:,:,2:] = states[:, :, 2:] / 200. 
                #print(states[:,0,:],'states shape')
                totalt = args.his+args.pred_roll
                kps_gt = states[totalt * index_roll: totalt * (index_roll + 1), :, :]
                kps_gt = torch.FloatTensor(kps_gt)

                return kps_gt, torch.tensor(edge_type), torch.tensor(edge_attr)
