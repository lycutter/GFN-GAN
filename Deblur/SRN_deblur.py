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
from MSRN.MSRN import MSRN
import random
import re
from torchvision import transforms

from data.data_loader import CreateDataLoader
# from networks.Discriminator import Discriminator
from networks.Discriminator import Discriminator
from ESRGANLoss import GANLoss, VGGFeatureExtractor
from SRN.network import SRNDeblurNet

# Training settings
parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--batchSize", type=int, default=32, help="Training batch size")
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
parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--loadSizeX', type=int, default=640, help='scale images to this size')
parser.add_argument('--loadSizeY', type=int, default=360, help='scale images to this size')
parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')



#RACM
parser.add_argument('--act', type=str, default='relu', help='activation function')
parser.add_argument('--n_resgroups', type=int, default=10, help='number of residual groups')
parser.add_argument('--n_resblocks', type=int, default=20, help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
parser.add_argument('--reduction', type=int, default=16, help='number of feature maps reduction')
parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')
parser.add_argument('--res_scale', type=float, default=1, help='residual scaling')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDA = 10
FilePath = './models/loss.txt'

FirstTrian = False


training_settings=[
    {'nEpochs': 25, 'lr': 1e-4, 'step':  7, 'lr_decay': 0.5, 'lambda_db': 0.5, 'gated': False},
    {'nEpochs': 60, 'lr': 1e-4, 'step': 30, 'lr_decay': 0.1, 'lambda_db': 0.5, 'gated': False},
    {'nEpochs': 55, 'lr': 5e-5, 'step': 25, 'lr_decay': 0.1, 'lambda_db':   0, 'gated': True}
]


def adjust_learning_rate(epoch):
    lr = opt.lr * (opt.lr_decay ** (epoch // opt.step))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def mkdir_steptraing():
    root_folder = os.path.abspath('.')
    models_folder = join(root_folder, 'models')
    step1_folder, step2_folder, step3_folder = join(models_folder,'1'), join(models_folder,'2'), join(models_folder, '3')
    isexists = os.path.exists(step1_folder) and os.path.exists(step2_folder) and os.path.exists(step3_folder)
    if not isexists:
        os.makedirs(step1_folder)
        os.makedirs(step2_folder)
        os.makedirs(step3_folder)
        print("===> Step training models store in models/1 & /2 & /3.")

def is_hdf5_file(filename):
    return any(filename.endswith(extension) for extension in [".h5"])




def checkpoint(epoch):
    model_out_path = "models/GFN_epoch_{}.pkl".format(epoch)
    torch.save(model_GFN_deblur, model_out_path)
    print("===>Checkpoint saved to {}".format(model_out_path))

def train(train_gen, model, criterion, optimizer, epoch):
    epoch_loss = 0
    train_gen = train_gen.load_data() ###############
    for iteration, batch in enumerate(train_gen):

        Sharpx128 = batch['B1x128']
        Sharpx64 = batch['B2x64']
        Sharpx32 = batch['B3x32']

        Blurx128 = batch['A1x128']
        Blurx64 = batch['A2x64']
        Blurx32 = batch['A3x32']

        Sharpx128 = Sharpx128.to(device)
        Sharpx64 = Sharpx64.to(device)
        Sharpx32 = Sharpx32.to(device)

        Blurx128 = Blurx128.to(device)
        Blurx64 = Blurx64.to(device)
        Blurx32 = Blurx32.to(device)

        # # show the pictures
        # shx128 = transforms.ToPILImage()(Sharpx128.cpu()[0])
        # shx128.save('./pictureShow/sharpX128.jpg')
        # shx64 = transforms.ToPILImage()(Sharpx64.cpu()[0])
        # shx64.save('./pictureShow/sharpX64.jpg')
        # shx32 = transforms.ToPILImage()(Sharpx32.cpu()[0])
        # shx32.save('./pictureShow/sharpX32.jpg')
        # Blx128 = transforms.ToPILImage()(Blurx128.cpu()[0])
        # Blx128.save('./pictureShow/Blurx128.jpg')
        # Blx64 = transforms.ToPILImage()(Blurx64.cpu()[0])
        # Blx64.save('./pictureShow/Blurx64.jpg')
        # Blx32 = transforms.ToPILImage()(Blurx32.cpu()[0])
        # Blx32.save('./pictureShow/Blurx32.jpg')

        outx128, outx64, outx32 = model(Blurx128, Blurx64, Blurx32)
        l1 = criterion(outx128, Sharpx128)
        l2 = criterion(outx64, Sharpx64)
        l3 = criterion(outx32, Sharpx32)

        loss = l1 + l2 + l3
        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_GFN_deblur.module.convlstm.parameters(), 3)
        optimizer.step()


        if iteration % 100 == 0:

            print("===> Epoch[{}]: loss:{:.4f}"
                  .format(epoch, loss))

            f = open(FilePath, 'a')
            f.write(
                "===> Epoch[{}]: loss:{:.4f}"
                .format(epoch, loss) + '\n')
            f.close()

            shx128 = transforms.ToPILImage()(Sharpx128.cpu()[0])
            shx128.save('./pictureShow/sharpX128.jpg')
            # shx64 = transforms.ToPILImage()(Sharpx64.cpu()[0])
            # shx64.save('./pictureShow/sharpX64.jpg')
            # shx32 = transforms.ToPILImage()(Sharpx32.cpu()[0])
            # shx32.save('./pictureShow/sharpX32.jpg')
            Blx128 = transforms.ToPILImage()(Blurx128.cpu()[0])
            Blx128.save('./pictureShow/Blurx128.jpg')
            # Blx64 = transforms.ToPILImage()(Blurx64.cpu()[0])
            # Blx64.save('./pictureShow/Blurx64.jpg')
            # Blx32 = transforms.ToPILImage()(Blurx32.cpu()[0])
            # Blx32.save('./pictureShow/Blurx32.jpg')
            outx128 = torch.clamp(outx128, min=0, max=1)
            outx128 = transforms.ToPILImage()(outx128.cpu()[0])
            outx128.save('./pictureShow/outx128.jpg')
            # outx64 = transforms.ToPILImage()(outx64.cpu()[0])
            # outx64.save('./pictureShow/outx64.jpg')
            # outx32 = transforms.ToPILImage()(outx32.cpu()[0])
            # outx32.save('./pictureShow/outx64.jpg')

    print("===>Epoch{} Complete: Avg loss is :{:4f}".format(epoch, epoch_loss / len(trainloader)))
    f = open(FilePath, 'a')
    f.write("===>Epoch{} Complete: Avg loss is :{:4f}\n".format(epoch, epoch_loss / len(trainloader)))
    f.close()


opt = parser.parse_args()
opt.seed = random.randint(1, 1200)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)



if opt.resume:
    if os.path.isfile(opt.resume):
        print("Loading from checkpoint {}".format(opt.resume))
        model_GFN_deblur = torch.load(opt.resume)
        model_GFN_deblur.load_state_dict(model_GFN_deblur.state_dict())


else:
    # model_GFN_deblur = SRNDeblurNet()
    model_GFN_deblur = torch.nn.DataParallel(SRNDeblurNet(xavier_init_all={'xavier_init_all':True})).cuda()

model_GFN_deblur.to(device)
print('# GFN_deblur parameters:', sum(param.numel() for param in model_GFN_deblur.parameters()))


criterion = torch.nn.L1Loss(size_average=True)
criterion = criterion.to(device)
cri_perception = VGGFeatureExtractor().to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_GFN_deblur.parameters()), 0.00001, [0.9, 0.999])

print()



opt.start_epoch = 116
opt.nEpochs = 1000
for epoch in range(opt.start_epoch, opt.nEpochs+1):
    trainloader = CreateDataLoader(opt)
    train(trainloader, model_GFN_deblur, criterion, optimizer, epoch)
    if epoch % 5 == 0:
        checkpoint(epoch)
