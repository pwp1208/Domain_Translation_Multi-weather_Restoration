from __future__ import print_function
import argparse
import os
import cv2

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_test import get_test_set

from utils import is_image_file, load_img, save_img
torch.backends.cudnn.benchmark = True
# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', default='dataset', required=False, help='facades')
parser.add_argument('--save_path', default='results', required=False, help='facades')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--cuda', default= True, action='store_false',  help='use cuda')
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.save_path+'/'):
        os.makedirs(opt.save_path+'/')


if opt.cuda:
    torch.cuda.manual_seed(123)
device = torch.device("cuda:0" if opt.cuda else "cpu")


G_path = "./checkpoints/netG_model.pth"
my_net = torch.load(G_path).to(device) 

d1_path = "./checkpoints/domain1.pth"
d1_net = torch.load(d1_path).to(device)

d2_path = "./checkpoints/domain2.pth"
d2_net = torch.load(d2_path).to(device)

d3_path = "./checkpoints/domain3.pth"
d3_net = torch.load(d3_path).to(device)


image_dir = "{}/".format(opt.dataset)
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]
transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)

test_set = get_test_set(opt.dataset)
testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.test_batch_size, shuffle=False)


a = 0

for iteration_test, batch in enumerate(testing_data_loader,1):

    real_a, filename = batch[0].to(device), batch[1]

    fake_d1 = d1_net(real_a)
    fake_d2 = d2_net(real_a)
    fake_d3 = d3_net(real_a)

    fake_b = my_net(real_a, fake_d1, fake_d2, fake_d3)

    out = fake_b 
    out_img = out[0].detach().squeeze(0).cpu()

    save_img(out_img, "./{}/{}".format(opt.save_path+'/',filename[0]))

    a+=1

print('##################### Testing Successfully Completed###################################')    
print('Restored Results are saved in the results folder')
