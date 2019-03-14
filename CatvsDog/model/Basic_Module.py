# -*- coding: utf-8 -*-
"""
@ project: CatvsDog
@ author: lzx
@ file: Basic_Module.py
@ time: 2019/3/13 19:37
"""
import torch
import time
import torch.nn as nn


class Basic_Module(nn.Module):
    def __init__(self):
        '''
        在nn.module的模型基础上再加模型，实现load和save功能
        '''
        super(Basic_Module,self).__init__()
        self.model_name = str(type(self))###确定模型的名字，str型

    def load(self,path):
        '''
        保存模型的参数
        :param path:
        :return:
        '''
        self.load_state_dict(torch.load(path))

    def save(self,name=None):
        '''
        保存模型
        :param name:
        :return:
        '''
        if name is None:
            pre = 'checkpoints/'+self.model_name+'_'
            name = time.strftime(pre+'%m%d_%H_%M_%S.pth')##按照时间来，月份+日期+时+分+秒
        torch.save(self.state_dict(),name)
        return name
    def get_optimizer(self,lr,weight_decay):
        '''
        :param lr: 学习率
        :param weight_decay:正则化参数，惩罚权重
        :return:
        '''
        return torch.optim.Adam(self.parameters(),lr=lr,weight_decay=weight_decay)

class Flatten(nn.Module):
    '''将数据平铺成一样'''
    def __init__(self):
        super(Flatten,self).__init__()

    def forward(self,x):
        return x.view(x.size(0),-1)