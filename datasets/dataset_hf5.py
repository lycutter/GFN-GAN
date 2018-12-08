import torch.utils.data as data
import torch
from skimage.io import imread, imsave
import numpy as np
import random
from os.path import join
import glob
import h5py
import sys
import os
from os.path import join
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Normalize
import torchvision.transforms as transforms
from PIL import Image
#=================== Utils ===================#

def is_image_file(filename):
    if filename.startswith('._'):
        return
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

#=================== Testing ===================#

def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(128 // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


class DataValSet(data.Dataset):
    def __init__(self, root_dir):
        #one input & ground truth
        self.input_dir = join(root_dir, 'LR_Blur')
        self.sr_dir = join(root_dir, 'HR')

        #Online Loading
        self.input_names = [x for x in sorted(os.listdir(self.input_dir)) if is_image_file(x)]
        self.hr_names = [x for x in sorted(os.listdir(self.sr_dir)) if is_image_file(x)]

    def __len__(self):
        return len(self.hr_names)

    def __getitem__(self, index):

        # input = np.asarray(imread(join(self.input_dir, self.input_names[index])).transpose((2, 0, 1)),
        #                    np.float32).copy() / 255
        # target = np.asarray(imread(join(self.sr_dir, self.hr_names[index])).transpose((2, 0, 1)),
        #                     np.float32).copy() / 255
        # return input, target


        #modify

        input_HR_Blur = np.asarray(imread(join(self.input_dir, self.input_names[index])).transpose((2, 0, 1)),
                           np.float32).copy() / 255
        input_HR_Blurx64 = np.asarray(imread(join(self.input_dir, self.input_names[index])),
                           np.float32).copy() / 255

        input     = np.asarray(imread(join(self.input_dir, self.input_names[index])), np.float32).copy() / 255
        target    = np.asarray(imread(join(self.sr_dir, self.hr_names[index])).transpose((2, 0, 1)), np.float32).copy() / 255
        transform_list = [ToTensor()]
        transform = transforms.Compose(transform_list)


        lr_input = transform(input)
        lr_tranform = train_lr_transform(128, 4)
        lr_input = lr_tranform(lr_input)

        lrx64 = transform(input_HR_Blurx64)
        lr_tranformx64 = train_lr_transform(128, 2)
        lrx64 = lr_tranformx64(lrx64)

        lrx16 = transform(input)
        lr_tranformx16 = train_lr_transform(128, 8)
        lrx16 = lr_tranformx16(lrx16)

        lrx8 = transform(input)
        lr_tranformx8 = train_lr_transform(128, 16)
        lrx8 = lr_tranformx8(lrx8)


        return input, target, lr_input, input_HR_Blur, lrx64, lrx8, lrx16

#=================== Training ===================#

class DataSet(data.Dataset):
    def __init__(self, h5py_file_path):
        super(DataSet, self).__init__()
        self.hdf5_file  = h5py_file_path

        self.file    = h5py.File(self.hdf5_file, 'r')
        #self.file.keys()
        self.inputs  = self.file.get("data")
        self.deblurs = self.file.get("label_db")
        self.hrs     = self.file.get("label")

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        #print(index)
        # numpy
        input_patch  = np.asarray(self.inputs[index, :, :, :], np.float32)
        deblur_patch = np.asarray(self.deblurs[index, :, :, :], np.float32)
        hr_patch     = np.asarray(self.hrs[index, :, :, :], np.float32)
        # randomly flip
        if random.randint(0, 1) == 0:
            input_patch  = np.flip(input_patch, 2)
            deblur_patch = np.flip(deblur_patch, 2)
            hr_patch     = np.flip(hr_patch, 2)
        # randomly rotation
        rotation_times = random.randint(0, 3)
        input_patch    = np.rot90(input_patch, rotation_times, (1, 2))
        deblur_patch   = np.rot90(deblur_patch, rotation_times, (1, 2))
        hr_patch       = np.rot90(hr_patch, rotation_times, (1, 2))

        return input_patch.copy(),\
               deblur_patch.copy(),\
               hr_patch.copy()

#
#
# if __name__ == '__main__':
#     dataset = DataSet('D:/pythonWorkplace/GFN-master/datasets/LR-GOPRO_x4_Part1.h5')
#     loader = DataLoader(dataset)
#     iter = iter(loader)
#     _, _1, data = iter.next()
#     print("OK")
