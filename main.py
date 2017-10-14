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
from model import Net,rec_inverse, ConvertNet, VDSR
from SRdataset import DatasetFromHdf5
path = './SR_v3'
pathtest = 'testimg'
if not os.path.exists(path):
    os.mkdir(path)
if not os.path.exists(os.path.join(path, pathtest)):
    os.mkdir(os.path.join(path, pathtest))
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=512, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=100, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--level', type=int, default=1, help='feature level')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--f_size', type=int, default=64, help='VDSR feature size')
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

train_dataloader = DataLoader(dataset=DatasetFromHdf5(scale=4,train=True, filename = 'celebA_train_s4.h5'), num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
test_dataloader = DataLoader(dataset=DatasetFromHdf5(scale=4,train=False,filename = 'celebA_test2_s4.h5'), num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model')
Hmodel = Net(level = opt.level)
Lmodel = Net(level = opt.level)
if opt.level == 0:
    curL_in=1 #level 0: 1, level 1: 16, level 2: 496
if opt.level == 1:
    curL_in=16
if opt.level == 2:
    curL_in=496
#model_level1 = torch.load(os.path.join('./SR_v2', "model_epoch_200.pth"))
model = VDSR(curL_in=curL_in,filter_num = opt.f_size)#ConvertNet(curL_in=50,receptive_size=4)
#model = model.load_state_dict(os.path.join(path, "model_epoch_200.pth"))
Hmodel.init_weight_h()
Lmodel.init_weight_l()
criterion2 = nn.MSELoss()
criterion = nn.MSELoss(size_average=False)
    
if cuda:
    Hmodel = Hmodel.cuda()
    Lmodel = Lmodel.cuda()
    model = model.cuda()
    #model_level1 = model_level1.cuda()
    criterion = criterion.cuda()
    criterion2 = criterion2.cuda()
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
        if opt.level > 0:
            #input = model_level0(input)
            f_H = Hmodel(target)
            f_L = Lmodel(input)
            f_H = Variable(f_H.data, requires_grad=False)
            f_L = Variable(f_L.data, requires_grad=False)  
            pre_f_H = model(f_L)        
            #f_pre = model_level1(f_L)
            #pre = model(input)
            loss = criterion(pre_f_H,f_H)
        else:
            pre = model(input)
            loss = criterion(pre,target)
        epoch_loss += (loss.data[0])#torch.mean
        loss.backward()#torch.ones(loss.size()).cuda()
        optimizer.step()
        #Loptimizer.step()
        if iteration % 20 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(train_dataloader), (loss.data[0])))
    
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(train_dataloader)))


def test():
    avg_psnr = 0
    for batch in test_dataloader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda()
            target = target.cuda()
        #input = model_level0(input)
        if opt.level > 0:
            f_L = Lmodel(input)
            #f_pre = model_level1(f_L)
            #pre = rec_inverse(f_L,Hmodel)
            prediction = model(f_L) 
            #+ f_L[:,:curL_in,:,:].contiguous()
            #f_pre = torch.cat((f_pre,f_H[:,curL_in:,:,:],),1)
            prediction = rec_inverse(prediction,Hmodel)
        else:
            prediction = model(input)
        
        mse = criterion2(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_dataloader)))


def checkpoint(epoch):
#    a = rec_inverse(f,model)
#    pic = to_img(input.cpu().data,1)
#    save_image(pic, os.path.join(path, 'gt_{}.png').format(epoch))
#    pic = to_img(a.cpu().data,1)
#    save_image(pic, os.path.join(path, 'rec_{}.png').format(epoch))
    if epoch == 1:
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
    if epoch == 1:
        checkpoint(epoch)
    if epoch%50==0:
        checkpoint(epoch)
