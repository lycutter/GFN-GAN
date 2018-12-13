# Citation:
#     Gated Fusion Network for Joint Image Deblurring and Super-Resolution
#     The British Machine Vision Conference(BMVC2018 oral)
#     Xinyi Zhang, Hang Dong, Zhe Hu, Wei-Sheng Lai, Fei Wang and Ming-Hsuan Yang
# Contact:
#     cvxinyizhang@gmail.com
# Project Website:
#     http://xinyizhang.tech/bmvc2018
#     https://github.com/jacquelinelala/GFN

from __future__ import print_function
import torch.optim as optim
import argparse
from torch.autograd import Variable
import os
from os.path import join
import torch

import random
import re
from torchvision import transforms

from data.data_loader import CreateDataLoader
# from networks.Discriminator import Discriminator
from networks.Discriminator import Discriminator
from ESRGANLossPeception import GANLoss, VGGFeatureExtractor, TVLoss
from Motion.Motion import Motion

# Training settings
parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--start_training_step", type=int, default=1, help="Training step")
parser.add_argument("--nEpochs", type=int, default=60, help="Number of epochs to train")
parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate, default=1e-4")
parser.add_argument("--step", type=int, default=7, help="Change the learning rate for every 30 epochs")
parser.add_argument("--start-epoch", type=int, default=1, help="Start epoch from 1")
parser.add_argument("--lr_decay", type=float, default=0.5, help="Decay scale of learning rate, default=0.5")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--resumeD", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--scale", default=4, type=int, help="Scale factor, Default: 4")
parser.add_argument("--lambda_db", type=float, default=0.5, help="Weight of deblurring loss, default=0.5")
parser.add_argument("--gated", type=bool, default=False, help="Activated gate module")
parser.add_argument("--isTest", type=bool, default=False, help="Test or not")
# parser.add_argument('--dataset', required=True, help='Path of the training dataset(.h5)')


# add lately
parser.add_argument('--dataset_mode', type=str, default='aligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
parser.add_argument('--dataroot', default='D:/pythonWorkplace/Dataset/CelebA_Pair/combo/')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--loadSizeX', type=int, default=640, help='scale images to this size')
parser.add_argument('--loadSizeY', type=int, default=360, help='scale images to this size')
parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDA = 10
FilePath = './models/loss.txt'

FirstTrian = False

opt = parser.parse_args()


trainloader = CreateDataLoader(opt)

train_gen = trainloader.load_data() ###############
for iteration, batch in enumerate(train_gen):
    LR_Blur = batch['LR_Blur']
    LR_Deblur = batch['LR_Sharp']
    Blur = transforms.ToPILImage()(LR_Blur.cpu()[0])
    Blur.save(join('D:/pythonWorkplace/Dataset/CelebA_Pair/DeblurValid/blur/', '{0:04d}_GFN_4x.png'.format(iteration)))
    resultSRDeblur = transforms.ToPILImage()(LR_Deblur.cpu()[0])
    resultSRDeblur.save(join('D:/pythonWorkplace/Dataset/CelebA_Pair/DeblurValid/hr/', '{0:04d}_GFN_4x.png'.format(iteration)))