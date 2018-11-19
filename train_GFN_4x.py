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
from torch.utils.data import DataLoader
from datasets.dataset_hf5 import DataSet
from networks.GFN_4x import Net
import random
import re
from torchvision import transforms

from data.data_loader import CreateDataLoader
# from networks.Discriminator import Discriminator
from Discriminator_ import Discriminator


# Training settings
parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--batchSize", type=int, default=4, help="Training batch size")
parser.add_argument("--start_training_step", type=int, default=1, help="Training step")
parser.add_argument("--nEpochs", type=int, default=60, help="Number of epochs to train")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate, default=1e-4")
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDA = 10

training_settings=[
    {'nEpochs': 25, 'lr': 1e-4, 'step':  7, 'lr_decay': 1, 'lambda_db': 0.5, 'gated': False},
    {'nEpochs': 60, 'lr': 1e-4, 'step': 30, 'lr_decay': 1, 'lambda_db': 0.5, 'gated': False},
    {'nEpochs': 550, 'lr': 1e-4, 'step': 25, 'lr_decay': 1, 'lambda_db': 0.2, 'gated': True}
]

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

def adjust_learning_rate(epoch):
        lr = opt.lr * (opt.lr_decay ** (epoch // opt.step))
        print("learning_rate:",lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def checkpoint(step, epoch):
    model_out_path = "models/{}/GFN_epoch_{}.pkl".format(step, epoch)
    model_out_path_D = "models/{}/GFN_D_epoch_{}.pkl".format(step, epoch)
    torch.save(model, model_out_path)
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

        LR_Blur = batch['A']
        LR_Deblur = batch['C']
        HR = batch['B']

        LR_Blur = LR_Blur.to(device)
        LR_Deblur = LR_Deblur.to(device)
        HR = HR.to(device)

        # show the pictures
        # LRB = transforms.ToPILImage()(LR_Blur.cpu()[0])
        # LRB.save('./pictureShow/LRB.jpg')
        # LRD = transforms.ToPILImage()(LR_Deblur.cpu()[0])
        # LRD.save('./pictureShow/LRD.jpg')
        # HRP = transforms.ToPILImage()(HR.cpu()[0])
        # HRP.save('./pictureShow/HRP.jpg')

        if opt.isTest == True:
            test_Tensor = torch.cuda.FloatTensor().resize_(1).zero_()+1.

        else:
            test_Tensor = torch.cuda.FloatTensor().resize_(1).zero_()

        if opt.gated == True:
            gated_Tensor = torch.cuda.FloatTensor().resize_(1).zero_()+1

        else:
            gated_Tensor = torch.cuda.FloatTensor().resize_(1).zero_()


        [lr_deblur, sr] = model(LR_Blur, gated_Tensor, test_Tensor)


        # calculate loss_D
        fake_sr = netD(sr)
        real_sr = netD(HR)

        fake_lr_deblur = netD(lr_deblur)
        real_lr_deblur = netD(LR_Deblur)

        d_loss_real = torch.mean(real_sr)
        d_loss_fake = torch.mean(fake_sr)

        d_loss_real_lr = torch.mean(real_lr_deblur)
        d_loss_fake_lr = torch.mean(fake_lr_deblur)

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

        # Compute gradient penalty of deblur
        alpha = torch.rand(LR_Deblur.size(0), 1, 1, 1).cuda().expand_as(LR_Deblur)
        interpolated = Variable(alpha * LR_Deblur.data + (1 - alpha) * lr_deblur.data, requires_grad=True)
        disc_interpolates = netD(interpolated)

        grad = torch.autograd.grad(outputs=disc_interpolates,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp_lr = torch.mean((grad_l2norm - 1) ** 2)

        # Backward + Optimize
        gradient_penalty = LAMBDA * d_loss_gp
        gradient_penalty_lr = LAMBDA * d_loss_gp_lr

        loss_D = d_loss_fake + d_loss_fake_lr - (d_loss_real + d_loss_real_lr) + gradient_penalty + gradient_penalty_lr

        optimizer_D.zero_grad()
        loss_D.backward(retain_graph=True)
        optimizer_D.step()

        # for p in netD.parameters():
        #     p.data.clamp_(-0.01, 0.01)


        # calculate loss_G
        loss_G_GAN = - netD(sr).mean()
        loss_G_GAN_lr = - netD(lr_deblur).mean()
        loss1 = criterion(lr_deblur, LR_Deblur)
        loss2 = criterion(sr, HR)
        Loss_G = loss2 + opt.lambda_db * loss1 + loss_G_GAN + loss_G_GAN_lr
        epoch_loss += Loss_G
        optimizer.zero_grad()
        Loss_G.backward()
        optimizer.step()


        if iteration % 50 == 0:
            # print("===> Epoch[{}]: G_GAN:{:.4f}, LossG:{:.4f}, LossD:{:.4f}, gredient_penalty:{:.4f}, d_real_loss:{:.4f}, d_fake_loss:{:.4f}"
            #       .format(epoch, loss_G_GAN.cpu(), mse.cpu(), loss_D.cpu(), gradient_penalty.cpu(), d_loss_real.cpu(), d_loss_fake.cpu()))

            print("===> Epoch[{}]: G_GAN:{:.4f}, G_GAN_lr:{:.4f}, penalty_lr:{:.4f}, LossG:{:.4f}, LossD:{:.4f}, penalty:{:.4f}, d_real:{:.4f}, d_fake:{:.4f}, d_real_lr:{:.4f}, d_fake_lr:{:.4f}"
                  .format(epoch, loss_G_GAN.cpu(), loss_G_GAN_lr.cpu(), gradient_penalty_lr.cpu(), Loss_G.cpu(), loss_D.cpu(), gradient_penalty.cpu(), d_loss_real.cpu(), d_loss_fake.cpu(), d_loss_real_lr.cpu(), d_loss_fake_lr.cpu()))

            sr_save = transforms.ToPILImage()(sr.cpu()[0])
            sr_save.save('./pictureShow/sr_save.png')
            deblur_lr_save = transforms.ToPILImage()(lr_deblur.cpu()[0])
            deblur_lr_save.save('./pictureShow/deblur_lr_save.png')
            hr_save = transforms.ToPILImage()(HR.cpu()[0])
            hr_save.save('./pictureShow/hr_save.png')
            deblur_sharp_save = transforms.ToPILImage()(LR_Deblur.cpu()[0])
            deblur_sharp_save.save('./pictureShow/deblur_sharp_save.png')
            blur_lr_save = transforms.ToPILImage()(LR_Blur.cpu()[0])
            blur_lr_save.save('./pictureShow/blur_lr_save.png')

    print("===>Epoch{} Complete: Avg loss is :{:4f}".format(epoch, epoch_loss / len(trainloader)))

opt = parser.parse_args()
opt.seed = random.randint(1, 10000)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)



# train_dir = opt.dataset
# train_sets = [x for x in sorted(os.listdir(train_dir)) if is_hdf5_file(x)]
print("===> Loading model and criterion")



if opt.resume:
    if os.path.isfile(opt.resume):
        print("Loading from checkpoint {}".format(opt.resume))
        model = torch.load(opt.resume)
        model.load_state_dict(model.state_dict())
        netD = torch.load(opt.resumeD)
        netD.load_state_dict(netD.state_dict())
        opt.start_training_step, opt.start_epoch = which_trainingstep_epoch(opt.resume)

else:
    model = Net()
    netD = Discriminator()
    mkdir_steptraing()

model = model.to(device)
netD = netD.to(device)
criterion = torch.nn.MSELoss(size_average=True)
criterion = criterion.to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
optimizer_D = optim.Adam(netD.parameters(), lr = opt.lr*2)
print()



opt.start_training_step = 3
for i in range(opt.start_training_step, 4):
    opt.nEpochs   = training_settings[i-1]['nEpochs']
    opt.lr        = training_settings[i-1]['lr']
    opt.step      = training_settings[i-1]['step']
    opt.lr_decay  = training_settings[i-1]['lr_decay']
    opt.lambda_db = training_settings[i-1]['lambda_db']
    opt.gated     = training_settings[i-1]['gated']
    print(opt)
    trainloader = CreateDataLoader(opt)
    for epoch in range(opt.start_epoch, opt.nEpochs+1):
        adjust_learning_rate(epoch-1)
        # trainloader = CreateDataLoader(opt)
        # random.shuffle(train_sets)
        # for j in range(len(train_sets)):
        #     print("Step {}:Training folder is {}-------------------------------".format(i, train_sets[j]))
        #     train_set = DataSet(join(train_dir, train_sets[j]))
        #     trainloader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True, num_workers=0)
        train(trainloader, model, netD, criterion, optimizer, epoch)

        if epoch % 5 == 0:
            checkpoint(i, epoch)

