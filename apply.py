from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from model import Net,rec_inverse, ConvertNet, VDSR
import numpy as np
from psnr import PSNR
from skimage.measure import compare_ssim as SSIM
from SRdataset import DatasetFromHdf5
from torch.utils.data import DataLoader
path = 'SR_v5'
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, default='./test/tests4r_101342.jpg', help='input image to use')
parser.add_argument('--gt_image', type=str, default='./test/test_101342.jpg', help='gt image to use')
parser.add_argument('--model', type=str, default='./'+path+'/model_epoch_300.pth', help='model file to use')
parser.add_argument('--Hmodel', type=str, default='./'+path+'/Hmodel_epoch_1.pth', help='model file to use')
parser.add_argument('--Lmodel', type=str, default='./'+path+'/Lmodel_epoch_1.pth', help='model file to use')
parser.add_argument('--output_filename', type=str,default='./'+path+'/output.png', help='where to save the output image')
parser.add_argument('--cuda', default = True, help='use cuda')
parser.add_argument('--level', type=int, default=2, help='network level')
opt = parser.parse_args()


print(opt)
#img = Image.open('./test/testsmall_101342_s8.jpg')
#gt_img = Image.open(opt.gt_image)
#plt.figure()    
#plt.imshow(gt_img,interpolation="nearest")
#plt.show()
#plt.figure()    
#plt.imshow(img,interpolation="nearest")
#plt.show()
img = Image.open(opt.input_image)
#half_the_width = img.size[0] / 2
#half_the_height = img.size[1] / 2
#img = img.crop(
#    (
#        half_the_width - 16,
#        half_the_height - 16,
#        half_the_width + 16,
#        half_the_height + 16
#    )
#)

img = img.convert('YCbCr')
y, cb, cr = img.split()
#y = np.transpose(y)
model = torch.load(opt.model)
Hmodel = torch.load(opt.Hmodel)
Lmodel = torch.load(opt.Lmodel)
input = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
input = torch.transpose(input,2,3)
if opt.cuda:
    model = model.cuda()
    Hmodel = Hmodel.cuda()
    Lmodel = Lmodel.cuda()
    input = input.cuda()
if opt.level == 0:
    out = model(input)
elif opt.level > 0:
    out = Lmodel(input)
    out = model(out)
    out = rec_inverse(out,Hmodel)
if opt.cuda:
    out = out.cpu()
out_img_y = out.data[0].numpy()
out_img_y = np.transpose(out.data[0].numpy(),(0,2,1))
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
plt.figure()    
plt.imshow(out_img,interpolation="nearest")
plt.show()
print ('ours:')
print (PSNR(np.asarray(out_img),np.asarray(gt_img)))
print (SSIM(np.asarray(out_img),np.asarray(gt_img),multichannel=True))
print ('Bicubic:')
print (PSNR(np.asarray(img),np.asarray(gt_img)))
print (SSIM(np.asarray(img),np.asarray(gt_img),multichannel=True))
out_img.save(opt.output_filename)
print('output image saved to ', opt.output_filename)
