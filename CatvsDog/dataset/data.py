# -*- coding: utf-8 -*-
"""
@ project: CatvsDog
@ author: lzx
@ file: data.py
@ time: 2019/3/13 16:25
"""
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms

train_root = 'F:/数据/数据image/kaggle/train/train/'
test_root = 'F:/数据/数据image/kaggle/test1/test1/'
class CatDog(data.Dataset):
    def __init__(self,root,transform=None,train=True,test=False):
        '''
        :param root: 路径
        :param transform:
        :param train:训练集标注
        :param test:测试集标注
        '''
        self.test = test
        self.train = train
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        # print(imgs[1])
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        imgs_num = len(imgs)
        if self.test:
            self.imgs = imgs
        elif self.train:
            self.imgs = imgs[:int(0.8*imgs_num)]
        else:
            self.imgs = imgs[int(0.8*imgs_num):]
        if transform is None:
            transform = transforms.Normalize(mean=[0.485,0.456,0.406],
                                             std=[0.229,0.224,0.225])
            if self.test or not train:
                self.transforms = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transform
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),##随机图像水平对调
                    transforms.ToTensor(),
                    transform
                ])
    def __len__(self):
        '''
        :return:
        '''
        return len(self.imgs)

    def __getitem__(self, i):
        '''
        :param i: index
        :return:
        '''
        img_every = self.imgs[i]
        if self.test:
            label = int(self.imgs[i].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_every.split('/')[-1] else 0
        data = Image.open(img_every)
        data = self.transforms(data)
        return data,label


if __name__ == '__main__':
    pass
    # from tqdm import tqdm
    # train_dataset = CatDog(root=train_root,train=False,test=False)
    # train_dataloader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    # for ii, (data, label) in tqdm(enumerate(train_dataloader)):
    #     print(data.shape)
    # valid_dataset = CatDog(root=train_root,train=False,test=False)
    # test_dataset = CatDog(root=test_root, train=False, test=True)
    # print(len(train_dataset))
    # print(len(valid_dataset))
    # print(len(test_dataset))
    # imgs = [os.path.join(test_root, img) for img in os.listdir(test_root)]
    # print(imgs[1])
    # print(1)
    # # imgs = sorted(imgs, key=lambda x: x)
    # imgs = sorted(imgs, key=lambda x: x)
    # print(imgs[3])
