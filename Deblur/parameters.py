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
from ParametersDetect.GFN_deblur_model import _DeblurringMoudle
from ParametersDetect.GFN_sr_model import _SRMoudle
from ParametersDetect.DeblurGAN_model import ResnetGenerator
from ESRGAN.ESRGAN import RRDBNet

# Training settings
parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")
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


# if FirstTrian:
#
#     training_settings=[
#         {'nEpochs': 30, 'lr': 1e-4, 'step':  7, 'lr_decay': 0.95, 'lambda_db': 0.5, 'gated': False},
#         {'nEpochs': 60, 'lr': 5e-5, 'step': 30, 'lr_decay': 0.95, 'lambda_db': 0.5, 'gated': False},
#         {'nEpochs': 85, 'lr': 5e-5, 'step': 25, 'lr_decay': 0.9, 'lambda_db': 0, 'gated': True}
#     ]
# else:
#     training_settings=[
#         {'nEpochs': 30, 'lr': 5e-5, 'step':  7, 'lr_decay': 0.95, 'lambda_db': 0.5, 'gated': False},
#         {'nEpochs': 60, 'lr': 2e-5, 'step': 30, 'lr_decay': 0.95, 'lambda_db': 0.5, 'gated': False},
#         {'nEpochs': 100, 'lr': 2e-5, 'step': 25, 'lr_decay': 0.90, 'lambda_db': 0, 'gated': True}
#     ]

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

def which_trainingstep_epoch(resume):
    trainingstep = "".join(re.findall(r"\d", resume)[0])
    start_epoch = "".join(re.findall(r"\d", resume)[1:])
    return int(trainingstep), int(start_epoch)

# def adjust_learning_rate(epoch):
#         # lr = opt.lr * (opt.lr_decay ** (epoch // opt.step))
#         # lr = opt.lr
#         if (epoch-1) % 3 == 0:
#             opt.lr = opt.lr * opt.lr_decay
#         print("learning_rate:",opt.lr)
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = opt.lr




def checkpoint(epoch):
    model_out_path = "models/GFN_epoch_{}.pkl".format(epoch)
    model_out_path_D = "models/GFN_D_epoch_{}.pkl".format(epoch)
    torch.save(model_GFN_deblur, model_out_path)
    torch.save(netD, model_out_path_D)
    print("===>Checkpoint saved to {}".format(model_out_path))

def train(train_gen, model, netD, criterion, optimizer, epoch):
    epoch_loss = 0
    train_gen = train_gen.load_data() ###############
    for iteration, batch in enumerate(train_gen):
        #input, targetdeblur, targetsr
        # LR_Blur = batch[0]
        # LR_Deblur = batch[1]
        # HR = batch[2]
        #
        HR = batch['HR_Sharp']
        HR_Blur = batch['HR_Blur']

        HR = HR.to(device)
        HR_Blur = HR_Blur.to(device)

        # # show the pictures
        # LRB = transforms.ToPILImage()(LR_Blur.cpu()[0])
        # LRB.save('./pictureShow/LRB.jpg')
        # LRD = transforms.ToPILImage()(LR_Deblur.cpu()[0])
        # LRD.save('./pictureShow/LRD.jpg')
        # HRP = transforms.ToPILImage()(HR.cpu()[0])
        # HRP.save('./pictureShow/HRP.jpg')


        HR_deblur = model(HR_Blur)


        # calculate loss_D
        fake_sr = netD(HR_deblur)
        real_sr = netD(HR)

        d_loss_real = torch.mean(real_sr)
        d_loss_fake = torch.mean(fake_sr)



        # Compute gradient penalty of HR and sr
        alpha = torch.rand(HR.size(0), 1, 1, 1).cuda().expand_as(HR)
        interpolated = Variable(alpha * HR.data + (1 - alpha) * HR_deblur.data, requires_grad=True)
        disc_interpolates = netD(interpolated)

        grad = torch.autograd.grad(outputs=disc_interpolates,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)


        # Backward + Optimize
        gradient_penalty = LAMBDA * d_loss_gp
        # gradient_penalty_lr = LAMBDA * d_loss_gp_lr

        loss_D = d_loss_fake - d_loss_real + gradient_penalty

        optimizer_D.zero_grad()
        loss_D.backward(retain_graph=True)
        optimizer_D.step()

        # for p in netD.parameters():
        #     p.data.clamp_(-0.01, 0.01)


        # calculate loss_G
        loss_G_GAN = - netD(HR_deblur).mean()

        pix_loss = criterion(HR_deblur, HR)
        feature_loss = cri_perception(HR_deblur, HR)

        image_loss = pix_loss + feature_loss
        Loss_G = 1 * image_loss + loss_G_GAN
        epoch_loss += Loss_G
        optimizer.zero_grad()
        Loss_G.backward()
        optimizer.step()


        if iteration % 200 == 0:
            # print("===> Epoch[{}]: G_GAN:{:.4f}, LossG:{:.4f}, LossD:{:.4f}, gredient_penalty:{:.4f}, d_real_loss:{:.4f}, d_fake_loss:{:.4f}"
            #       .format(epoch, loss_G_GAN.cpu(), mse.cpu(), loss_D.cpu(), gradient_penalty.cpu(), d_loss_real.cpu(), d_loss_fake.cpu()))

            print("===> Epoch[{}]: G_GAN:{:.4f}, image_loss:{:.4f}, LossG:{:.4f}, LossD:{:.4f}, penalty:{:.4f}, d_real:{:.4f}, d_fake:{:.4f}"
                  .format(epoch, loss_G_GAN.cpu(), image_loss.cpu(), Loss_G.cpu(), loss_D.cpu(), gradient_penalty.cpu(), d_loss_real.cpu(), d_loss_fake.cpu()))

            f = open(FilePath, 'a')
            f.write(
                "===> Epoch[{}]: G_GAN:{:.4f}, image_loss:{:.4f}, LossG:{:.4f}, LossD:{:.4f}, d_real_loss:{:.6f}, d_fake_loss:{:.6f}, penalty:{:.4f}"
                .format(epoch, loss_G_GAN.cpu(), image_loss.cpu(), Loss_G.cpu(), loss_D.cpu(), d_loss_real.cpu(), d_loss_fake.cpu(),
                        gradient_penalty.cpu()) + '\n')
            f.close()


            hr_deblur_save = torch.clamp(HR_deblur, min=0, max=1)
            hr_deblur_save = transforms.ToPILImage()(hr_deblur_save.cpu()[0])
            hr_deblur_save.save('./pictureShow/hr_deblur_save.png')
            hr_save = transforms.ToPILImage()(HR.cpu()[0])
            hr_save.save('./pictureShow/hr_save.png')
            hr_blur = transforms.ToPILImage()(HR_Blur.cpu()[0])
            hr_blur.save('./pictureShow/hr_blur.png')


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
        netD = torch.load(opt.resumeD)
        netD.load_state_dict(netD.state_dict())
        opt.start_training_step, opt.start_epoch = which_trainingstep_epoch(opt.resume)

else:
    # model_msrn = MSRN(opt)
    model_GFN_deblur = _DeblurringMoudle()
    # model_GFN_sr = _SRMoudle()
    # model_DebulrGAN = ResnetGenerator()
    # model_ESR = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, norm_type=None)
    netD = Discriminator()
    mkdir_steptraing()

# model_msrn = model_msrn.to(device)
model_GFN_deblur.to(device)
# model_GFN_sr = model_GFN_sr.to(device)
# model_DebulrGAN = model_DebulrGAN.to(device)
# model_ESR = model_ESR.to(device)
netD = netD.to(device)
# print('# MSRN parameters:', sum(param.numel() for param in model_msrn.parameters()))
print('# GFN_deblur parameters:', sum(param.numel() for param in model_GFN_deblur.parameters()))
# print('# sr parameters:', sum(param.numel() for param in model_GFN_sr.parameters()))
# print('# deblurGAN parameters:', sum(param.numel() for param in model_GFN_deblur.parameters()))
# print('# ESR parameters:', sum(param.numel() for param in model_ESR.parameters()))
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

criterion = torch.nn.L1Loss(size_average=True)
criterion = criterion.to(device)
cri_perception = VGGFeatureExtractor().to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_GFN_deblur.parameters()), 0.0001, [0.9, 0.999])
optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), 0.0004, [0.9, 0.999])
print()


# for i in range(opt.start_training_step, 4):
#     opt.nEpochs   = training_settings[i-1]['nEpochs']
#     opt.lr        = training_settings[i-1]['lr']
#     opt.step      = training_settings[i-1]['step']
#     opt.lr_decay  = training_settings[i-1]['lr_decay']
#     opt.lambda_db = training_settings[i-1]['lambda_db']
#     opt.gated     = training_settings[i-1]['gated']
#     print(opt)
opt.start_epoch = 121
opt.nEpochs = 1000
for epoch in range(opt.start_epoch, opt.nEpochs+1):
    trainloader = CreateDataLoader(opt)
    train(trainloader, model_GFN_deblur, netD, criterion, optimizer, epoch)
    if epoch % 5 == 0:
        checkpoint(epoch)