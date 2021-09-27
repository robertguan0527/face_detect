# from layers.box_utils import point_form
import csv
import torch

from torch.utils.data import Dataset
from utils.csv_wr import csv_write, csv_writerows
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
        
        
        res =[]
        if not target:
            raise ValueError('target is None')
        for bbox in target:
            bndbox =self._change_point_form(bbox,width,height)
            for pos_index,cur_pt in enumerate(bndbox) :
                if pos_index % 2 == 0 :
                    bndbox[pos_index] = cur_pt/width 
                else:
                    bndbox[pos_index] = cur_pt/height
                
            label_idx = self.class_to_ind['face']
            bndbox.append(label_idx) #label_idx for face just 0
            res +=[bndbox]
        return res


    def _change_point_form(self,box,width,height):
       
        x,y,w,h = box
        # if b[2]<2 or b[3]<2 or b[0]<0 or b[1]<0:
        #     continue
        box[0]=x
        box[1]=y
        if width<30 or height<30:
            box[2]=x+w
            box[3]=y+h
        else:
            box[2]=min(x+w,width)
            box[3]=min(y+h,height)
            
        return box



class WiderFaceDataset(Dataset):

    def __init__(self,root, image_sets='train', transform=None, target_transform=WiderAnnotationTransformer(),dataset_name='Widerface'):
# root is the path of data folder's location
        super(WiderFaceDataset, self).__init__()
        self.root = root
        self.image_sets = image_sets
        self._images_folder = os.path.join(root,'widerface',f"WIDER_{image_sets}","images")
        self._gt_path = os.path.join(root, 'widerface','wider_face_split',f"wider_face_{image_sets}_bbx_gt.txt")
        self.images_name_list = []
        self.ground_truth = []
    
        self.name = dataset_name
        # print(self._gt_path)
        with open(self._gt_path, 'r') as f:
            for i in f:
                if len(i) == 0:
                    continue
                self.images_name_list.append(i.rstrip())
                self.ground_truth.append(i.rstrip())

        self.images_name_list = list(filter(lambda x: x.endswith('.jpg') or x.endswith('.bmp'),
                                       self.images_name_list))
        # print(len(self.images_name_list))
        # csv_writerows('annotation.csv',zip(self.images_name_list,self.ground_truth))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images_name_list)



    def __getitem__(self, index: int):
        im,gt,h,w = self.pull_item(index)
        return im, gt



    def pull_item(self, index):
        image_name = self.images_name_list[index]
        # print(image_name)
        # 查找文件名
        loc = self._search(image_name)
        # 解析人脸个数
        
        face_nums = int(self.ground_truth[loc + 1])
        if not face_nums:
            # face_nums+=1
            raise ValueError(f"{face_nums}, {image_name},{self.ground_truth[loc + 1]},{loc}") 
        # 读取矩形框
        rects = []
        for i in range(loc + 2, loc + 2 + face_nums):
            line = self.ground_truth[i]
            x, y, w, h = line.split(' ')[:4]
            if x =='' or y =='' or w == '' or h == '':
                raise ValueError('target maybe is empty')
            if self.image_sets =='train':
                if int(w)<=30 or int(h)<=30:
                    continue
            x, y, w, h = list(map(lambda k: int(k), [x, y, w, h]))                          
            rects.append([x, y, w, h])

        # 图像
        image = cv2.imread(os.path.join(self._images_folder, image_name))
        height, width, chanels = image.shape

        if self.target_transform is not None:
            if rects:
                target = self.target_transform(rects,width,height)  
            else:
                return None, None,height,width
        csv_write('pic_log.csv',[os.path.join(self._images_folder, image_name),rects])
        if self.transform is not None:
            target = np.array(target)
            try:
                img, boxes, labels = self.transform(image, target[:, :4], target[:, 4])
            except Exception as e:
                print(target,type(target),image_name)

            # img = transfroms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])(torch.from_numpy(img))
            # boxes_tensor = torch.from_numpy(boxes)
            
            # boxes = boxes_tensor.numpy()
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
 
        # print(image_name)
# /userdir/guanyihua1993/tmp/pycharm_project_robert0806/ssd-pytorch/weights
        return torch.from_numpy(img).permute(2,0,1), target, height, width

    def _search(self, image_name):
        for i, line in enumerate(self.ground_truth):
            if image_name == line:
                return i



    def pull_image(self, index):
        img_id = self.images_name_list[index]
        return cv2.imread(os.path.join(self._images_folder, img_id),cv2.IMREAD_COLOR)

      
    def pull_anno(self, index):
        img_id = self.images_name_list[index]
        # print(image_name)
        # 查找文件名
        loc = self._search(img_id)
        # 解析人脸个数
        
        face_nums = int(self.ground_truth[loc + 1])
   
        # 读取矩形框
        rects = []
        for i in range(loc + 2, loc + 2 + face_nums):
            line = self.ground_truth[i]
            x, y, w, h = line.split(' ')[:4]
            if x =='' or y =='' or w == '' or h == '':
                raise ValueError('target maybe is empty') 
         
            x, y, w, h = list(map(lambda k: int(k), [x, y, w, h]))                          
            rects.append([x, y, w, h])
        gt = self.target_transform(rects,1,1)
        return img_id,gt
    
    
      
       
    

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
 
   
if __name__ =="__main__":
    gt = [[620,103,12,18]]
    gt = WiderAnnotationTransformer()(gt,1024,800)
    print(gt)



   

   
