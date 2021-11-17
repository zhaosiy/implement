import os
import random
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

os.system('mkdir -p ' + args.outf_kp)
os.system('mkdir -p ' + args.dataf)

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
    datasets[phase].gen_data()

