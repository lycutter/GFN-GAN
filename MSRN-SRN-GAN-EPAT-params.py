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
from SAGFN_MSRN_SRN import Net
import random
import re
from torchvision import transforms
from math import log10
from data.data_loader import CreateDataLoader
# from networks.Discriminator import Discriminator
from networks.Discriminator import Discriminator
from ESRGANLossPeception import GANLoss, VGGFeatureExtractor
from TextualLoss.vgg import VGG, GramMatrix, GramMSELoss

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
# parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--dataroot', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)', default='D:\pythonWorkplace\Dataset\CelebA_Pair\combo')
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


training_settings=[
    {'nEpochs': 25, 'lr': 1e-4, 'step': 10, 'lr_decay': 0.9, 'lambda_db': 0.5, 'gated': False},
    {'nEpochs': 60, 'lr': 1e-4, 'step':  5, 'lr_decay': 0.9, 'lambda_db': 0.3, 'gated': False},
    {'nEpochs': 55, 'lr': 5e-5, 'step':  5, 'lr_decay': 0.8, 'lambda_db': 0.1, 'gated': True}
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




def checkpoint(step, epoch):
    model_out_path = "models/{}/GFN_epoch_{}.pkl".format(step, epoch)
    model_out_path_D = "models/{}/GFN_D_epoch_{}.pkl".format(step, epoch)
    torch.save(model, model_out_path)
    torch.save(netD, model_out_path_D)
    print("===>Checkpoint saved to {}".format(model_out_path))

def train(train_gen, model, criterion, optimizer, epoch, lr):
    epoch_loss = 0
    train_gen = train_gen.load_data() ###############
    for iteration, batch in enumerate(train_gen):
        #input, targetdeblur, targetsr

        HR = batch['HR_Sharp'].to(device)
        Sharpx32 = batch['B1x32'].to(device)
        Sharpx16 = batch['B2x16'].to(device)
        Sharpx8 = batch['B3x8'].to(device)

        Blurx32 = batch['A1x32'].to(device)
        Blurx16 = batch['A2x16'].to(device)
        Blurx8 = batch['A3x8'].to(device)

        LR_vgg = Sharpx32.clone()

        sr_real_style_loss_out = []
        sr_fake_style_loss_out = []

        deblur_real_style_loss_out = []
        deblur_fake_style_loss_out = []


        if opt.isTest == True:
            test_Tensor = torch.cuda.FloatTensor().resize_(1).zero_()+1.

        else:
            test_Tensor = torch.cuda.FloatTensor().resize_(1).zero_()

        if opt.gated == True:
            gated_Tensor = torch.cuda.FloatTensor().resize_(1).zero_()+1

        else:
            gated_Tensor = torch.cuda.FloatTensor().resize_(1).zero_()


        deblurx32, deblurx16, deblurx8, sr = model(Blurx32, Blurx16, Blurx8, gated_Tensor, test_Tensor)
        HR_vgg = HR.clone()
        sr_vgg = sr.clone()
        deblur_vgg = deblurx32.clone()

        # calculate textual loss
        # SR
        j = 0
        for index, layer in enumerate(vgg19):
            if index == 21:
                break
            HR_vgg = layer(HR_vgg)
            if index == loss_layer[j]:
                j += 1
                sr_real_style_loss_out.append(HR_vgg)
        sr_style_targets = [GramMatrix()(A).detach() for A in sr_real_style_loss_out]

        sr_targets = sr_style_targets

        j = 0
        for index, layer in enumerate(vgg19):
            if index == 21:
                break
            sr_vgg = layer(sr_vgg)
            if index == loss_layer[j]:
                j += 1
                sr_fake_style_loss_out.append(sr_vgg)
        sr_layer_losses = [style_weights[a] * loss_fns[a](A, sr_targets[a]) for a, A in
                           enumerate(sr_fake_style_loss_out)]
        sr_textual_loss = sum(sr_layer_losses)


        # deblur textual loss

        j = 0
        for index, layer in enumerate(vgg19):
            if index == 21:
                break
            LR_vgg = layer(LR_vgg)
            if index == loss_layer[j]:
                j += 1
                deblur_real_style_loss_out.append(LR_vgg)
        deblur_style_targets = [GramMatrix()(A).detach() for A in deblur_real_style_loss_out]
        deblur_targets = deblur_style_targets

        j = 0
        for index, layer in enumerate(vgg19):
            if index == 21:
                break
            deblur_vgg = layer(deblur_vgg)
            if index == loss_layer[j]:
                j += 1
                deblur_fake_style_loss_out.append(deblur_vgg)
        deblur_layer_losses = [style_weights[a] * loss_fns[a](A, deblur_targets[a]) for a, A in
                               enumerate(deblur_fake_style_loss_out)]
        deblur_textual_loss = sum(deblur_layer_losses)

        textual_loss = sr_textual_loss + opt.lambda_db * deblur_textual_loss



        fake_sr = netD(sr)
        real_sr = netD(HR)

        d_loss_real = torch.mean(real_sr)
        d_loss_fake = torch.mean(fake_sr)

        # Compute gradient penalty of HR and sr
        alpha = torch.rand(HR.size(0), 1, 1, 1).cuda().expand_as(HR)
        interpolated = Variable(alpha * HR.data + (1 - alpha) * sr.data, requires_grad=True)
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



        # calculate loss_G
        loss_G_GAN = - netD(sr).mean()

        l1 = criterion(deblurx32, Sharpx32)
        l2 = criterion(deblurx16, Sharpx16)
        l3 = criterion(deblurx8, Sharpx8)
        deblur_image = l1 + l2 + l3
        deblur_perception = cri_perception(deblurx32, Sharpx32)
        sr_perception = cri_perception(sr, HR)

        psnr_deblur = 10 * log10(1 / l1)
        sr_image = criterion(sr, HR)

        image_loss = sr_image + deblur_image * opt.lambda_db
        perceptionloss = sr_perception + deblur_perception * opt.lambda_db
        psnr_sr = 10 * log10(1 / sr_image)

        Loss_G = image_loss + 0.01 * loss_G_GAN + 0.01 * perceptionloss + textual_loss


        optimizer.zero_grad()
        Loss_G.backward()
        optimizer.step()


        epoch_loss += Loss_G

        if iteration % 200 == 0:

            print("===> Epoch[{}]: G_GAN:{:.4f}, image_loss:{:.4f}, text:{:.4f}, perception:{:.4f}, LossG:{:.4f}, d_real:{:.4f}, d_fake:{:.4f}, deblur:{:.4f}, sr:{:.4f}"
                  .format(epoch, loss_G_GAN.cpu(), image_loss.cpu(), textual_loss, perceptionloss, Loss_G.cpu(), d_loss_real.cpu(), d_loss_fake.cpu(), psnr_deblur, psnr_sr))

            f = open(FilePath, 'a')
            f.write(
                "===> Epoch[{}]: G_GAN:{:.4f}, image_loss:{:.4f}, text:{:.4f}, perception:{:.4f}, LossG:{:.4f}, d_real_loss:{:.6f}, d_fake_loss:{:.6f}, deblur:{:.4f}, sr:{:.4f}, lr:{:.8f}"
                .format(epoch, loss_G_GAN.cpu(), image_loss.cpu(), textual_loss, perceptionloss, Loss_G.cpu(), d_loss_real.cpu(), d_loss_fake.cpu(), psnr_deblur, psnr_sr, lr) + '\n')
            f.close()
            sr_save = torch.clamp(sr, min=0, max=1)
            sr_save = transforms.ToPILImage()(sr_save.cpu()[0])
            sr_save.save('./pictureShow/sr_save.png')
            deblur_lr_save = torch.clamp(deblurx32, min=0, max=1)
            deblur_lr_save = transforms.ToPILImage()(deblur_lr_save.cpu()[0])
            deblur_lr_save.save('./pictureShow/deblur_lr_save.png')
            hr_save = transforms.ToPILImage()(HR.cpu()[0])
            hr_save.save('./pictureShow/hr_save.png')
            deblur_sharp_save = transforms.ToPILImage()(Sharpx32.cpu()[0])
            deblur_sharp_save.save('./pictureShow/deblur_sharp_save.png')
            blur_lr_save = transforms.ToPILImage()(Blurx32.cpu()[0])
            blur_lr_save.save('./pictureShow/blur_lr_save.png')



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
        model = torch.load(opt.resume)
        netD = torch.load(opt.resumeD)
        model.load_state_dict(model.state_dict())
        netD.load_state_dict(netD.state_dict())
        opt.start_training_step, opt.start_epoch = which_trainingstep_epoch(opt.resume)

else:
    model = Net()
    netD = Discriminator()
    mkdir_steptraing()



# model = torch.load('models/1/GFN_epoch_1.pkl')
# model.load_state_dict(model.state_dict())
# netD = torch.load('models/1/GFN_D_epoch_1.pkl')
# netD.load_state_dict(netD.state_dict())

model = model.to(device)
netD = netD.to(device)
criterion = torch.nn.MSELoss(size_average=True)
criterion = criterion.to(device)
cri_perception = VGGFeatureExtractor().to(device)


# textual init
vgg19 = cri_perception.vggNet
loss_layer = [1, 6, 11, 20]
loss_fns = [GramMSELoss()] * len(loss_layer)
if torch.cuda.is_available():
    loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512]]

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 0.0001, [0.9, 0.999])
optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), 0.0002, [0.9, 0.999])
print('# generator parameters:', sum(param.numel() for param in model.parameters()))
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
print()


for i in range(opt.start_training_step, 4):
    opt.nEpochs   = training_settings[i-1]['nEpochs']
    opt.lr        = training_settings[i-1]['lr']
    opt.step      = training_settings[i-1]['step']
    opt.lr_decay  = training_settings[i-1]['lr_decay']
    opt.lambda_db = training_settings[i-1]['lambda_db']
    opt.gated     = training_settings[i-1]['gated']
    print(opt)
    for epoch in range(opt.start_epoch, opt.nEpochs+1):
        lr = adjust_learning_rate(epoch-1)
        trainloader = CreateDataLoader(opt)
        train(trainloader, model, criterion, optimizer, epoch, lr)
        if epoch % 5 == 0:
            checkpoint(i, epoch)
