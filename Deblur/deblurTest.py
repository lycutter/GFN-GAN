from __future__ import print_function
import argparse
import os
import time
from math import log10
from os.path import join
from torchvision import transforms
from torchvision import utils as utils
import torch
from torch.utils.data import DataLoader
from datasets.dataset_hf5 import DataValSet
import statistics
import matplotlib.pyplot as plot
import re
import numpy as np
from PIL import Image
from ParametersDetect.GFN_deblur_model import _DeblurringMoudle
# from SRN.network import SRNDeblurNet
from Unet.Unet import Net

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Deblur_Path = 'SRN_Deblur'
# Deblur_Path = 'GFN_Deblur'
Deblur_Path = 'Unet_Deblur'
root_val_dir = 'D:/pythonWorkplace/GFN-GAN/datasets/LR-GOPRO/Validation_4'
model_path = './models/weights_backup/'

testloader = DataLoader(DataValSet(root_val_dir), batch_size=1, shuffle=False, pin_memory=False)

Deblur_dir = join(root_val_dir, Deblur_Path)

isexists = os.path.exists(Deblur_dir)
if not isexists:
    os.makedirs(Deblur_dir)
print("The results of testing images sotre in {}.".format(Deblur_dir))
# model = _DeblurringMoudle().to(device)
# model = SRNDeblurNet().to(device)
model = Net().to(device)
model = torch.load(model_path+'GFN_epoch_50_unetDeblur.pkl')
model.load_state_dict(model.state_dict())

for iteration, batch in enumerate(testloader):

    Blurx128 = batch[2].to(device)
    Blurx64 = batch[6].to(device)
    Blurx32 = batch[5].to(device)


    # Deblurx128, Deblurx64, Deblurx32 = model(Blurx128, Blurx64, Blurx32)
    Deblurx128 = model(Blurx128)
    Deblurx128 = torch.clamp(Deblurx128, min=0, max=1)

    resultDeblur = transforms.ToPILImage()(Deblurx128.cpu()[0])
    resultDeblur.save(join(Deblur_dir, '{0:04d}_GFN_4x.png'.format(iteration)))
