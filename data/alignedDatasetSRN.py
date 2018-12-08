import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Normalize

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir_AB))

        # assert(opt.resize_or_crop == 'resize_and_crop')

        # transform_list = [transforms.ToTensor(),
        #                   transforms.Normalize((0.5, 0.5, 0.5),
        #                                        (0.5, 0.5, 0.5))]
        transform_list = [ToTensor()]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # AB = Image.open(AB_path)
        # AB = AB.resize((self.opt.loadSizeX * 2, self.opt.loadSizeY), Image.BICUBIC)
        AB = self.transform(AB)

        # w_total = AB.size(2)
        # w = int(w_total / 2)
        # h = AB.size(1)
        # w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        # h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
        #
        # A = AB[:, h_offset:h_offset + self.opt.fineSize, # blurred img
        #        w_offset:w_offset + self.opt.fineSize]
        # B = AB[:, h_offset:h_offset + self.opt.fineSize, # sharp img
        #        w + w_offset:w + w_offset + self.opt.fineSize]

        A = AB[:, 0:128, 0:128]
        B = AB[:, 0:128, 128:256]

        lr_tranform_x2 = train_lr_transform(self.opt.fineSize, 2)
        lr_tranform_x4 = train_lr_transform(self.opt.fineSize, 4)


        A1x128 = A
        A2x64  = lr_tranform_x2(A)
        A3x32  = lr_tranform_x4(A)

        B1x128 = B
        B2x64  = lr_tranform_x2(B)
        B3x32  = lr_tranform_x4(B)

        # D = A.clone()
        # A = lr_tranform(A)
        # C = lr_tranform(B)

        if (not self.opt.no_flip) and random.random() < 0.5:  # 翻转图片，扩展数据量
            # idx = [i for i in range(A.size(2) - 1, -1, -1)]
            # idx = torch.LongTensor(idx)
            # A = A.index_select(2, idx)
            # B = B.index_select(2, idx)
            idx_A1 = [i for i in range(A1x128.size(2) - 1, -1, -1)]
            idx_A2 = [i for i in range(A2x64.size(2) - 1, -1, -1)]
            idx_A3 = [i for i in range(A3x32.size(2) - 1, -1, -1)]
            idx_B1 = [i for i in range(B1x128.size(2) - 1, -1, -1)]
            idx_B2 = [i for i in range(B2x64.size(2) - 1, -1, -1)]
            idx_B3 = [i for i in range(B3x32.size(2) - 1, -1, -1)]
            idx_A1 = torch.LongTensor(idx_A1)
            idx_A2 = torch.LongTensor(idx_A2)
            idx_A3 = torch.LongTensor(idx_A3)
            idx_B1 = torch.LongTensor(idx_B1)
            idx_B2 = torch.LongTensor(idx_B2)
            idx_B3 = torch.LongTensor(idx_B3)
            A1x128 = A1x128.index_select(2, idx_A1)
            A2x64 = A2x64.index_select(2, idx_A2)
            A3x32 = A3x32.index_select(2, idx_A3)
            B1x128 = B1x128.index_select(2, idx_B1)
            B2x64 = B2x64.index_select(2, idx_B2)
            B3x32 = B3x32.index_select(2, idx_B3)

        return {'A1x128': A1x128, 'A2x64': A2x64, 'A3x32': A3x32, 'B1x128': B1x128, 'B2x64': B2x64, 'B3x32': B3x32} # A是LR_Blur， B是HR_Sharp, C是LR_Sharp

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(128 // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])

