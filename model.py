import torch.nn as nn
import torch.nn.init as init
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

def CalcNextChannelNum(curL_out,keepComp=1):
    nextL_in = curL_out*2 + 1
    nextL_out = round(nextL_in*4*keepComp)-1
    return int(nextL_in),int(nextL_out)
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        curL_in = 1; curL_out = 3 # initial
        self.conv1 = nn.Conv2d(curL_in, curL_out, 2,stride=2, bias = False) 
        self.conv1_b = nn.Conv2d(curL_in, 1, 2,stride=2, bias = False)
        f1 = np.ones([1,curL_in,2,2])
        self.f1=torch.from_numpy(f1/np.linalg.norm(f1))
        curL_in,curL_out = CalcNextChannelNum(curL_out)
        self.conv2 = nn.Conv2d(curL_in, curL_out, 2,stride=2, bias = False)
        self.conv2_b = nn.Conv2d(curL_in, 1, 2,stride=2, bias = False)
        f1 = np.ones([1,curL_in,2,2])
        self.f2=torch.from_numpy(f1/np.linalg.norm(f1))
        curL_in,curL_out = CalcNextChannelNum(curL_out)
        self.conv3 = nn.Conv2d(curL_in, curL_out, 2,stride=2, bias = False)
        self.conv3_b = nn.Conv2d(curL_in, 1, 2,stride=2, bias = False)
        f1 = np.ones([1,curL_in,2,2])
        self.f3=torch.from_numpy(f1/np.linalg.norm(f1))
        del f1
        self.relu = nn.ReLU()

    def Augment(self, ac, dc):
        return self.relu(torch.cat((ac,-ac, dc),1))
    
    def init_weight(self):
        init.orthogonal(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal(self.conv3.weight, init.calculate_gain('relu'))
        init.readweight(self.conv1_b.weight, self.f1.float())
        init.readweight(self.conv2_b.weight, self.f2.float())
        init.readweight(self.conv3_b.weight, self.f3.float())
#        w = torch.randn(10,1,5,5)
#        init.readweight(self.conv1.weight, w)
#        init.constant(self.conv1.bias, 0)
#        w = torch.randn(10,1,5,5)
#        init.readweight(self.conv1.weight, w)
#        init.constant(self.conv1.bias, 0)
    def forward(self, x):
        f = self.Augment(self.conv1(x),self.conv1_b(x))
        f = self.Augment(self.conv2(f),self.conv2_b(f))
        f = self.Augment(self.conv3(f),self.conv3_b(f))
        return f
        
class ConvertNet(nn.Module):
    def __init__(self,curL_in=439,receptive_size=4):
        super(ConvertNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.curL_in = curL_in
        self.receptive_size = receptive_size # initial
        self.convert = nn.Sequential(
            
            nn.Linear(self.curL_in*self.receptive_size*self.receptive_size, 2000),  # b, 120
            nn.ReLU(True),
            nn.Linear(2000, 4000),  # b, 120
            nn.ReLU(True),
            nn.Linear(4000, 2000),  # b, 120
            nn.ReLU(True),
            nn.Linear(2000, self.curL_in*self.receptive_size*self.receptive_size),  # b, 84
            
        )
        self.relu = nn.ReLU()

    def forward(self, x):
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
    
def inverseconv(feature, W, stride = 2, padding=0, output_padding=0):
    feature = inverseAug(feature)
    W = Cal_inv_W(W,reception_size = 2)
    #W = Variable(torch.from_numpy(W).cuda())    
    return F.conv_transpose2d(feature, W, bias = None, stride=stride, padding=padding, output_padding=output_padding)
def rec_inverse(feature,Net):
    
    W = Net.conv3.weight
    F3 = inverseconv(feature, W, stride = 2, padding=0, output_padding=0)      
    W = Net.conv2.weight
    F2 = inverseconv(F3, W, stride = 2, padding=0, output_padding=0) 
    W = Net.conv1.weight
    F1 = inverseconv(F2, W, stride = 2, padding=0, output_padding=0) 
    return F1
