from os import listdir
from os.path import join
import random

from PIL import Image
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file, load_img

class DatasetFromFolder_Test(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder_Test, self).__init__()
        self.path = image_dir
        self.image_filenames = [x for x in listdir(self.path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        image = cv2.imread(join(self.path, self.image_filenames[index]))
        width = image.shape[0]
        a = image[:,:width*1, :]
        a1 = cv2.resize(a, (256, 256),  interpolation = cv2.INTER_CUBIC)
        a1 = transforms.ToTensor()(a1)    
        a1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a1)

        return a1, self.image_filenames[index]

    def __len__(self):
        return len(self.image_filenames)
