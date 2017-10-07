from __future__ import print_function
import argparse
from math import log10

import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Net,rec_inverse, ConvertNet
from SRdataset import DatasetFromHdf5
path = './SR_v0'
pathtest = 'testimg'
if not os.path.exists(path):
    os.mkdir(path)
if not os.path.exists(os.path.join(path, pathtest)):
    os.mkdir(os.path.join(path, pathtest))
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=2, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=256, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=100, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')

train_dataloader = DataLoader(dataset=DatasetFromHdf5(scale=2,train=True), num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
test_dataloader = DataLoader(dataset=DatasetFromHdf5(scale=2,train=False), num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model')
Hmodel = Net()
Lmodel = Net()
model = ConvertNet()
Hmodel.init_weight()
Lmodel.init_weight()
criterion = nn.MSELoss()

def mse_loss(input, target):
    return torch.sum((input - target)^2) / input.data.nelement()
    
if cuda:
    Hmodel = Hmodel.cuda()
    Lmodel = Lmodel.cuda()
    model = model.cuda()
    criterion = criterion.cuda()
optimizer = optim.Adam(model.parameters(),lr=opt.lr)    
#Hoptimizer = optim.Adam([{'params': Hmodel.conv1_b.parameters(),'lr':0},{'params': Hmodel.conv2_b.parameters(),'lr': 0},
#                          {'params': Hmodel.conv3_b.parameters(),'lr': 0}], lr=opt.lr)
#Loptimizer = optim.Adam([{'params': Lmodel.conv1_b.parameters(),'lr':0},{'params': Lmodel.conv2_b.parameters(),'lr': 0},
#                          {'params': Lmodel.conv3_b.parameters(),'lr': 0}], lr=opt.lr)


def to_img(x,depth):
    #x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), depth, 32, 32)
    return x
    
def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(train_dataloader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda()
            target = target.cuda()
    
        optimizer.zero_grad()
        #Loptimizer.zero_grad()
        f_H = Hmodel(target)
        f_L = Lmodel(input)
        f_H = Variable(f_H.data, requires_grad=False)
        f_L = Variable(f_L.data, requires_grad=False)     
        pre_f_H = model(f_L)        
        loss = criterion(pre_f_H, f_H)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        #Loptimizer.step()
        if iteration % 20 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(train_dataloader), loss.data[0]))
    
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(train_dataloader)))


def test():
    avg_psnr = 0
    for batch in test_dataloader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda()
            target = target.cuda()

        f_L = Lmodel(input)
        pre_f_H = model(f_L) 
        prediction = rec_inverse(pre_f_H,Hmodel)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_dataloader)))


def checkpoint(epoch):
#    a = rec_inverse(f,model)
#    pic = to_img(input.cpu().data,1)
#    save_image(pic, os.path.join(path, 'gt_{}.png').format(epoch))
#    pic = to_img(a.cpu().data,1)
#    save_image(pic, os.path.join(path, 'rec_{}.png').format(epoch))
    if epoch < 25:
        Hmodel_out_path = os.path.join(path, "Hmodel_epoch_{}.pth").format(epoch)
        torch.save(Hmodel, Hmodel_out_path)
        Lmodel_out_path = os.path.join(path, "Lmodel_epoch_{}.pth").format(epoch)
        torch.save(Lmodel, Lmodel_out_path)
    model_out_path = os.path.join(path, "model_epoch_{}.pth").format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved for epoch_{}".format(epoch))
#
for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    test()
    if epoch%20:
        checkpoint(epoch)
