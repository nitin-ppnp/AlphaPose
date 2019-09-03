# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import os
import h5py
from functools import reduce
from pycocotools.coco import COCO

import torch.utils.data as data
from ..pose import generateSampleBox
from opt import opt
import numpy as np


class Mscoco(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '/ps/project/datasets/COCO/images'    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = 17
        self.nJoints = 17

        self.accIdxs = (1, 2, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.flipRef = ((2, 3), (4, 5), (6, 7),
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17))
        
        # create train/val split
        self.coco_train = COCO(os.path.join(self.img_folder,'annotations','person_keypoints_train2014.json'))
        catIds_train = self.coco_train.getCatIds(catNms=['person'])
        imgIds_train = self.coco_train.getImgIds(catIds=catIds_train)
        self.imgs_train = self.coco_train.loadImgs(imgIds_train)
        
        self.coco_val = COCO(os.path.join(self.img_folder,'annotations','person_keypoints_val2014.json'))
        catIds_val = self.coco_val.getCatIds(catNms=['person'])
        imgIds_val = self.coco_val.getImgIds(catIds=catIds_val)    
        self.imgs_val = self.coco_val.loadImgs(imgIds_val)      
        annIds = self.coco_val.getAnnIds(imgIds=imgIds_val, catIds=[1]) # catId is 1 for 'person'
        self.anns_val = self.coco_val.loadAnns(annIds)

#         with h5py.File('../data/coco/annot_coco.h5', 'r') as annot:
#             # train
#             self.imgname_coco_train = annot['imgname'][:-5887]
#             self.bndbox_coco_train = annot['bndbox'][:-5887]
#             self.part_coco_train = annot['part'][:-5887]
#             # val
#             self.imgname_coco_val = annot['imgname'][-5887:]
#             self.bndbox_coco_val = annot['bndbox'][-5887:]
#             self.part_coco_val = annot['part'][-5887:]

        # self.size_train = self.imgname_coco_train.shape[0]
        # self.size_val = self.imgname_coco_val.shape[0]

        self.size_train = len(self.imgs_train)
        self.size_val = len(self.imgs_val)

    def __getitem__(self, index):
        sf = self.scale_factor

#         if self.is_train:
#             part = self.part_coco_train[index]
#             bndbox = self.bndbox_coco_train[index]
#             imgname = self.imgname_coco_train[index]
#         else:
#             part = self.part_coco_val[index]
#             bndbox = self.bndbox_coco_val[index]
#             imgname = self.imgname_coco_val[index]
        
        if self.is_train:
            imgname = self.imgs_train[index]['file_name']
            annIds = self.coco_train.getAnnIds(imgIds=self.imgs_train[index]['id'], catIds=[1]) # catId is 1 for 'person'
            anns_train = self.coco_train.loadAnns(annIds)[0]
            part = np.array(anns_train['keypoints']).reshape(-1,3)
            bndbox = np.array(anns_train['bbox']).reshape(1,-1)
            bndbox[0,2:] += bndbox[0,:2]
            img_path = os.path.join(self.img_folder, 'train2014',imgname)
        else:
            imgname = self.imgs_val[index]['file_name']
            annIds = self.coco_val.getAnnIds(imgIds=self.imgs_val[index]['id'], catIds=[1]) # catId is 1 for 'person'
            anns_val = self.coco_val.loadAnns(annIds)[0]
            part = np.array(anns_val['keypoints']).reshape(-1,3)
            bndbox = np.array(anns_val['bbox']).reshape(1,-1)
            bndbox[0,2:] += bndbox[0,:2]
            img_path = os.path.join(self.img_folder, 'val2014',imgname)
            
        import ipdb;ipdb.set_trace()
            
        metaData = generateSampleBox(img_path, bndbox, part, self.nJoints,
                                     'coco', sf, self, train=self.is_train)

        inp, out, setMask = metaData

        return inp, out, setMask, 'coco'

    def __len__(self):
        if self.is_train:
            return self.size_train
        else:
            return self.size_val
