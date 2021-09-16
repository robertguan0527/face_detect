import torch
from torch.utils.data import Dataset
import torchvision.transforms as transfroms
import matplotlib.pyplot as plt
import os
import PIL.Image as Image
import PIL
import cv2
import numpy as np

class WiderFaceDataset(Dataset):

    def __init__(self, images_folder, ground_truth_file, transform=None, target_transform=None):
        super(WiderFaceDataset, self).__init__()
        self.images_folder = images_folder
        self.ground_truth_file = ground_truth_file
        self.images_name_list = []
        self.ground_truth = []
        with open(ground_truth_file, 'r') as f:
            for i in f:
                self.images_name_list.append(i.rstrip())
                self.ground_truth.append(i.rstrip())

        self.images_name_list = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.bmp'),
                                       self.images_name_list))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images_name_list)

    def __getitem__(self, index):
        image_name = self.images_name_list[index]
        # 查找文件名
        loc = self._search(image_name)
        # 解析人脸个数
        face_nums = int(self.ground_truth[loc + 1])
        # 读取矩形框
        rects = []
        for i in range(loc + 2, loc + 2 + face_nums):
            line = self.ground_truth[i]
            x, y, w, h = line.split(' ')[:4]
            x, y, w, h = list(map(lambda k: int(k), [x, y, w, h]))
            rects.append([x, y, w, h])

        # 图像
        image = PIL.Image.open(os.path.join(self.images_folder, image_name))

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            rects = list(map(lambda x: self.target_transform(x), rects))
            # rects = torch.Tensor(rects)

        return image, rects

    def _search(self, image_name):
        for i, line in enumerate(self.ground_truth):
            if image_name == line:
                return i


if __name__ == '__main__':

    root_path = r'/userdir/guanyihua1993/tmp/pycharm_project_robert0806/ssd-pytorch/data'
    images_folder = os.path.join(root_path, 'widerface','WIDER_train','images')
    ground_truth_file = os.path.join(root_path,'widerface','wider_face_split','wider_face_train_bbx_gt.txt')

    dataset = WiderFaceDataset(images_folder,
                              ground_truth_file,
                              transform=transfroms.ToTensor(),
                              target_transform=lambda x: torch.tensor(x))
    print(dataset)
    var = next(iter(dataset))
    print(type(var))
    print(var[0].shape,type(var[1]))
    a = torch.tensor(var[1])


   

   
