# from layers.box_utils import point_form
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transfroms
import matplotlib.pyplot as plt
import os
import PIL.Image as Image
import PIL
import cv2
import numpy as np

FACE_CLASSES =('face',)




class WiderAnnotationTransformer(object):
    def __init__(self, class_to_ind = None):
        self.class_to_ind = class_to_ind or dict(zip(FACE_CLASSES,range(len(FACE_CLASSES))))
        
   
    def __call__(self,target,width,height):# target is rects
        
        target = self._change_point_form(target)
        res =[]
        for bbox in target:
            bndbox =[]
            for pos_index,cur_pt in enumerate(bbox) :
                if pos_index % 2 == 0 :
                    cur_pt = cur_pt/width 
                else:
                    cur_pt = cur_pt/height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind['face']
            bndbox.append(label_idx) #label_idx for face just 0
            res +=[bndbox]
        return res


    def _change_point_form(self,boxs):
        bboxs = []
        for b in boxs:
            if b[2]<2 or b[3]<2 or b[0]<0 or b[1]<0:
                continue
            b[0]=float(b[0]-b[2]/2)
            b[1]=float(b[1]-b[3]/2)
            b[2]=float(b[0]+b[2]/2)
            b[3]=float(b[1]+b[3]/2)
            bboxs.append(b)
        return bboxs



class WiderFaceDataset(Dataset):

    def __init__(self,root, image_sets='train', transform=None, target_transform=WiderAnnotationTransformer(),dataset_name='Widerface'):
# root is the path of data folder's location
        super(WiderFaceDataset, self).__init__()
        self.root = root
        self._images_folder = os.path.join(root,'widerface',f"WIDER_{image_sets}","images")
        self._gt_path = os.path.join(root, 'widerface','wider_face_split',f"wider_face_{image_sets}_bbx_gt.txt")
        self.images_name_list = []
        self.ground_truth = []
        self.name = dataset_name
        # print(self._gt_path)
        with open(self._gt_path, 'r') as f:
            for i in f:
                self.images_name_list.append(i.rstrip())
                self.ground_truth.append(i.rstrip())

        self.images_name_list = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.bmp'),
                                       self.images_name_list))
        # print(len(self.images_name_list))

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
        image = cv2.imread(os.path.join(self._images_folder, image_name))
        height, width, chanels = image.shape

        if self.target_transform is not None:
            
            target = self.target_transform(rects,width,height)  
            

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(image, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
 

# /userdir/guanyihua1993/tmp/pycharm_project_robert0806/ssd-pytorch/weights
        return torch.from_numpy(img).permute(2,0,1), target

    def _search(self, image_name):
        for i, line in enumerate(self.ground_truth):
            if image_name == line:
                return i



    def pull_image(self, index):
        img_id = self.images_name_list[index]
        return cv2.imread(os.path.join(self._images_folder, img_id),cv2.IMREAD_COLOR)



    
   


   

   
