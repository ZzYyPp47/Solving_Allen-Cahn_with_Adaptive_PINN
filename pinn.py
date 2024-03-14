#-*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2024/1/25 15:26
@version: 1.0
@File: pinn.py
'''

# 导入必要的包
import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import datetime
from loss import *


class pinn(object):
    # 初始化
    def __init__(self,name,total_epochs,tor,loss_func,model,point,opt,weights_init,loss_weight,device):
        self.total_epochs = total_epochs
        self.path = 'save/'+name+'.pth'# '_'+datetime.date.today().strftime('%Y-%m-%d,%H:%M:%S')+'.pth'
        self.tor = tor # loss的停止阈值
        self.device = device # 使用的设备
        self.Epochs_loss = [] # 记录loss
        self.weights_init = weights_init # 确定神经网络初始化的方式
        self.loss_func = loss_func # 损失函数(接口)
        self.model = model # 神经网络(接口)
        self.point = point # 数据点(接口)
        self.opt = opt # 优化器(接口)
        if point is not None:
            self.loss_computer = LossCompute(model, loss_func, point, loss_weight)  # 损失计算(接口)
        self.model = self.model.to(self.device) # 将model移入相应设备
        self.model.apply(self.weights_initer) # 神经网络初始化
        print(self.model)

    # 参数初始化
    def weights_initer(self,model):
        if isinstance(model,nn.Conv2d):
            self.weights_init(model.weight.data)
            model.bias.data.zero_()
        elif isinstance(model,nn.Linear):
            self.weights_init(model.weight.data)
            model.bias.data.zero_()

    # 为LBFGS优化器准备的闭包函数
    def closure(self):
        self.opt.zero_grad()  # 清零梯度
        Loss = self.loss_computer.loss()  # 计算损失
        Loss.backward(retain_graph=True) # 反向计算出梯度
        return Loss.item()



    # 训练模型
    def train_all(self):
        print('now using device:', self. device, ':', torch.cuda.current_device())
        print('start training,using seed:',torch.initial_seed())
        self.model.train() # 启用训练模式
        start_time = time.time()
        for epoch in range(self.total_epochs):
            self.opt.zero_grad() # 清零梯度信息
            Loss = self.loss_computer.loss()
            self.Epochs_loss.append([epoch,Loss.cpu().detach().numpy()])
            Loss.backward(retain_graph=True) # 反向计算出梯度
            self.opt.step() # 更新参数
            if (epoch + 1)% 100 == 0:
                print('epoch:',epoch + 1,'Loss=',Loss.cpu().detach().numpy())
            if Loss <= self.tor:
                print('epoch:', epoch + 1,'loss_func =',Loss.cpu().detach().numpy(), '<=', self.tor, '(given tolerate loss_func)')
                self.Epochs_loss = np.array(self.Epochs_loss)
                break
        end_time =time.time()
        self.Epochs_loss = np.array(self.Epochs_loss)
        print('training terminated')
        print('using times:',end_time-start_time,'s')


    # 最基本的训练模块
    def train(self):
        self.opt.zero_grad()  # 清零梯度信息
        Loss = self.loss_computer.loss()
        Loss.backward(retain_graph=True)  # 反向计算出梯度
        self.opt.step()  # 更新参数
        return Loss.item()




     # 保存模型参数
    def save(self):
        state_dict = {"Arc": self.model,"seed": torch.initial_seed(), "model": self.model.state_dict(), "opt": self.opt.state_dict(), "loss": self.Epochs_loss}
        torch.save(state_dict,self.path)
        print('model saved to',self.path)
