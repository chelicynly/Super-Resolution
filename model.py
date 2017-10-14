import torch.nn as nn
import torch.nn.init as init
import torch
from math import sqrt
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

def CalcNextChannelNum(curL_out,keepComp=1):
    nextL_in = curL_out*2 + 1
    nextL_out = round(nextL_in*4*keepComp)-1
    return int(nextL_in),int(nextL_out)
    
class Net(nn.Module):
    def __init__(self, level = 1):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.level = level
        self.kernel_size = 4
        curL_in = 1; curL_out = 15 # initial
        self.conv1 = nn.Conv2d(curL_in, curL_out, self.kernel_size, stride=self.kernel_size, bias = False) 
        self.conv1_b = nn.Conv2d(curL_in, 1, self.kernel_size, stride=self.kernel_size, bias = False)
        f1 = np.ones([1,curL_in,self.kernel_size,self.kernel_size])
        self.f1=torch.from_numpy(f1/np.linalg.norm(f1))
        if self.level >= 2:
            curL_in,curL_out = CalcNextChannelNum(curL_out)
            self.conv2 = nn.Conv2d(curL_in, curL_out, self.kernel_size,stride=self.kernel_size, bias = False)
            self.conv2_b = nn.Conv2d(curL_in, 1, self.kernel_size,stride=self.kernel_size, bias = False)
            f1 = np.ones([1,curL_in,self.kernel_size,self.kernel_size])
            self.f2=torch.from_numpy(f1/np.linalg.norm(f1))
            curL_in,curL_out = CalcNextChannelNum(curL_out)
        if self.level >= 3:
            self.conv3 = nn.Conv2d(curL_in, curL_out, 2,stride=2, bias = False)
            self.conv3_b = nn.Conv2d(curL_in, 1, 2,stride=2, bias = False)
            f1 = np.ones([1,curL_in,2,2])
            self.f3=torch.from_numpy(f1/np.linalg.norm(f1))
            del f1
        self.relu = nn.ReLU()

    def Augment(self, ac, dc):
        return self.relu(torch.cat((ac,-ac, dc),1))
    
    def init_weight_h(self):
#        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
#        init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
#        init.orthogonal(self.conv3.weight, init.calculate_gain('relu'))
        init.readweight(self.conv1_b.weight, self.f1.float())
        if self.level >= 2:
            init.readweight(self.conv2_b.weight, self.f2.float())
        if self.level >= 3:
            init.readweight(self.conv3_b.weight, self.f3.float())
        w = np.load('/media/mcl418-2/New Volume/Yujian/PCA/h4_celebA_1_s8_centered.npy')
#        idx = w[:,0]>0
#        w[idx,:] = w[idx,:]*(-1) 
        w = w[0:w.shape[0]-1,:]
        W_pca = np.reshape(w,(w.shape[0],self.kernel_size,self.kernel_size,1))
        W_pca = np.transpose(W_pca,(0,3,1,2))
        init.readweight(self.conv1.weight, torch.from_numpy(W_pca).float())
        if self.level >= 2:
            w = np.load('/media/mcl418-2/New Volume/Yujian/PCA/h4_celebA_2_s8_centered.npy')
            w = w[0:w.shape[0]-1,:]
            W_pca = np.reshape(w,(w.shape[0],self.kernel_size,self.kernel_size,-1))
            W_pca = np.transpose(W_pca,(0,3,1,2))
            init.readweight(self.conv2.weight, torch.from_numpy(W_pca).float())
        if self.level >= 3:
            w = np.load('/media/mcl418-2/New Volume/Yujian/PCA/h4_celebA_3_s4_centered.npy')
            w = w[0:w.shape[0]-1,:]
            W_pca = np.reshape(w,(w.shape[0],self.kernel_size,self.kernel_size,-1))
            W_pca = np.transpose(W_pca,(0,3,1,2))
            init.readweight(self.conv3.weight, torch.from_numpy(W_pca).float())
        
    def init_weight_l(self):
        init.readweight(self.conv1_b.weight, self.f1.float())
        if self.level >= 2:
            init.readweight(self.conv2_b.weight, self.f2.float())
        if self.level >= 3:
            init.readweight(self.conv3_b.weight, self.f3.float())
            
        w = np.load('/media/mcl418-2/New Volume/Yujian/PCA/l4_celebA_1_s8_centered.npy')
        w = w[0:w.shape[0]-1,:]
        W_pca = np.reshape(w,(w.shape[0],4,4,1))
        W_pca = np.transpose(W_pca,(0,3,1,2))
        init.readweight(self.conv1.weight, torch.from_numpy(W_pca).float())
        if self.level >= 2:
            w = np.load('/media/mcl418-2/New Volume/Yujian/PCA/l4_celebA_2_s8_centered.npy')
            w = w[0:w.shape[0]-1,:]
            W_pca = np.reshape(w,(w.shape[0],self.kernel_size,self.kernel_size,-1))
            W_pca = np.transpose(W_pca,(0,3,1,2))
            init.readweight(self.conv2.weight, torch.from_numpy(W_pca).float())
        if self.level >= 3:
            w = np.load('/media/mcl418-2/New Volume/Yujian/PCA/l4_celebA_3_s4_centered.npy')
            w = w[0:w.shape[0]-1,:]
            W_pca = np.reshape(w,(w.shape[0],self.kernel_size,self.kernel_size,-1))
            W_pca = np.transpose(W_pca,(0,3,1,2))
            init.readweight(self.conv3.weight, torch.from_numpy(W_pca).float())

    def forward(self, f1):
        if self.level == 1:
            #f2 = self.Augment(self.conv1(f1),self.conv1_b(f1))
            f2 = torch.cat((self.conv1(f1),self.conv1_b(f1)),1)
        if self.level == 2:
            f1 = self.Augment(self.conv1(f1),self.conv1_b(f1))
            f2 = torch.cat((self.conv2(f1),self.conv2_b(f1)),1)
        if self.level == 3:
            f2 = self.Augment(self.conv1(f1),self.conv1_b(f1))
            f2 = self.Augment(self.conv2(f2),self.conv2_b(f2))
            f2 = torch.cat((self.conv3(f2),self.conv3_b(f2)),1)
        return f2
        
class ConvertNet(nn.Module):
    def __init__(self,curL_in=220,receptive_size=4):
        super(ConvertNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.curL_in = curL_in
        self.receptive_size = receptive_size # initial
        self.convert = nn.Sequential(
            
            nn.Linear(self.curL_in*self.receptive_size*self.receptive_size, 1000),  # b, 120
            nn.ReLU(True),
            nn.Linear(1000, 1000),  # b, 120
            nn.ReLU(True),
            nn.Linear(1000, 1000),  # b, 120
            nn.ReLU(True),
            nn.Linear(1000, 1000),  # b, 120
            nn.ReLU(True),
            nn.Linear(1000, 1000),  # b, 120
            nn.ReLU(True),
            nn.Linear(1000, 1000),  # b, 120
            nn.ReLU(True),
            nn.Linear(1000, 1000),  # b, 120
            nn.ReLU(True),
            nn.Linear(1000, 1000),  # b, 120
            nn.ReLU(True),
            nn.Linear(1000, 1000),  # b, 120
            nn.ReLU(True),
            nn.Linear(1000, self.curL_in*self.receptive_size*self.receptive_size),  # b, 84
            
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.curL_in != x.size(1):
            x = x[:,:self.curL_in,:,:].contiguous()

        f = self.convert(x.view(-1,self.curL_in*self.receptive_size*self.receptive_size))
        return f.view(-1,self.curL_in,self.receptive_size,self.receptive_size)

        
def Cal_inv_W(W,reception_size = 2, dc = True):
    # W has shape (outputChannel, inputChannel, receptionSize, receptionSize)
     
    assert len(W.size())==4, "input weight is not 4D"
    if dc==True:
        f1 = torch.ones([1,W.size()[1],reception_size,reception_size])
        f1 = Variable(f1/torch.norm(f1)).cuda()
        
        W = torch.cat((W,f1),0)
    assert W.size()[2] == W.size()[3], "reception window is not square"
    reception_size = W.size()[2]    
    n_input_channels = W.size()[1]
    n_output_channels = W.size()[0]
    
    # rearranged the weight
    # shape is (n_components, n_features)
    featureNum = reception_size**2*n_input_channels
    W_aranged = W.permute(0,2,3,1)
    W_aranged = W_aranged.resize(int(n_output_channels),int(featureNum))
 
    
    #######################################
    #calculate the inverse weight
    W_aranged_INV =  torch.inverse(W_aranged).t()
    W_INV = W_aranged_INV.resize(n_output_channels,reception_size,reception_size,n_input_channels)
    W_INV = W_INV.permute(0,3,1,2)
    return W_INV.contiguous()
    
def inverseAug(F):
    Cur_out = F.size()[1]
    assert Cur_out%2 == 1
    AC_len = (Cur_out - 1)/2
    DC = F[:,Cur_out-1:Cur_out,:,:]
    AC = F[:,0:AC_len,:,:] - F[:,AC_len:2*AC_len,:,:]
    return torch.cat((AC,DC),1)
    
def inverseconv(feature, W, stride = 2, padding=0, output_padding=0, Aug = True):
    if Aug: 
        feature = inverseAug(feature)   
    W = Cal_inv_W(W,reception_size = stride)
    #W = Variable(torch.from_numpy(W).cuda())    
    return F.conv_transpose2d(feature, W, bias = None, stride=stride, padding=padding, output_padding=output_padding)
def rec_inverse(feature,Net):
    level = Net.level
    if level == 1:
       W = Net.conv1.weight
       F1 = inverseconv(feature, W, stride = 4, padding=0, output_padding=0, Aug = False) 
    if level == 2:
        W = Net.conv2.weight
        F2 = inverseconv(feature, W, stride = 4, padding=0, output_padding=0, Aug = False) 
        W = Net.conv1.weight
        F1 = inverseconv(F2, W, stride = 4, padding=0, output_padding=0)
    if level == 3:           
        W = Net.conv3.weight
        F3 = inverseconv(feature, W, stride = 2, padding=0, output_padding=0, Aug = False)      
        W = Net.conv2.weight
        F2 = inverseconv(F3, W, stride = 2, padding=0, output_padding=0) 
        W = Net.conv1.weight
        F1 = inverseconv(F2, W, stride = 4, padding=0, output_padding=0) 
    return F1

class Conv_ReLU_Block(nn.Module):
    def __init__(self,filter_num = 64):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=filter_num, out_channels=filter_num, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))
        
class VDSR(nn.Module):
    def __init__(self,curL_in=220, filter_num = 64):
        super(VDSR, self).__init__()
        self.filter_num = filter_num
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 6)
        self.input = nn.Conv2d(in_channels=curL_in, out_channels=filter_num, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=filter_num, out_channels=curL_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(filter_num = self.filter_num))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out