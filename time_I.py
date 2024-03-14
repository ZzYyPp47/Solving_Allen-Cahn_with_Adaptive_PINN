# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2024/2/11 21:59
@version: 1.0
@File: test.py
'''

import torch
import torch.nn as nn
import matplotlib as plt
from scipy.io import loadmat
import random
from pinn import *
from data import *
from NN.FNN import FNN
from NN.ResFNN import ResFNN
from math import ceil
from loss import *


# 加载并测试...
def test():
    # 初始化
    seed = 0
    Arc = [2,128,128,128,128,128,128,128,128,1] # 神经网络架构
    func = nn.Tanh() # 确定激活函数
    N_pde = 1000
    N_bound = 200
    N_ini = 500
    N_real = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 自动选择可用设备,优先GPU
    learning_rate = 0.01
    init_method = torch.nn.init.kaiming_uniform_ # 设置神经网络参数初始化方法
    # total_epochs = 450
    delta_t = 0.01
    tor = 0.0001 # loss阈值
    loss_func = nn.MSELoss().to(device) # 确定损失计算函数
    loss_weight = [1,1,100,0] # loss各项权重(pde,bound,ini,real)
    file = 'data.mat'
    u = loadmat(file, mat_dtype=True)['u']
    u = torch.from_numpy(u).cuda()
    t = torch.arange(0, 1 + delta_t, delta_t, device=device)


    # 建立模型
    set_seed(seed)  # 设置确定的随机种子

    time_I = FNN(Arc, func, device)

    # 初始化L-BFGS优化器
    opt_time_I = torch.optim.LBFGS(time_I.parameters(), lr=0.01, max_iter=20,
                                        line_search_fn='strong_wolfe')

    # opt_time_I = torch.optim.Adam(params=time_I.parameters(), lr=learning_rate)

    point_time_I = create_point(0, 0, N_ini, 0, device)

    pinn_time_I = pinn('time_I', None, tor, loss_func, time_I, point_time_I, opt_time_I, init_method, loss_weight, device)
    # 务必确定model与point位于同一设备!!

    # 训练模型
    for num in range(ceil(1 / delta_t)):

        # 生成边界点
        start = t[num]
        end = t[num + 1]

        bound_x1 = torch.ones(N_bound // 2, 1, requires_grad=True, dtype=torch.float32, device=device)  # x = 1
        bound_xm1 = torch.full_like(bound_x1, -1, requires_grad=True, dtype=torch.float32, device=device)  # x = -1

        bound_tx1 = torch.rand_like(bound_x1, requires_grad=True, dtype=torch.float32, device=device) * (end - start) + start
        bound_txm1 = torch.rand_like(bound_x1, requires_grad=True, dtype=torch.float32, device=device) * (end - start) + start

        pde_x = 2 * torch.rand(N_ini,1,requires_grad=True, dtype=torch.float32,device=device) - 1  # -1 < x < 1
        pde_t = torch.rand_like(pde_x, requires_grad=True, dtype=torch.float32, device=device) * (end - start) + start

        point_all = PointContainer(pde_x,pde_t,None,None,bound_x1,bound_xm1,bound_tx1,bound_txm1)

        pinn_time_I.loss_computer.update_points(point_all)

        print('now training form {} to {}'.format(start,end))

        # for epochs in range(total_epochs):
        #
        #     计算损失并反向传播
        #     pinn_time_I.model.train()
        #     pinn_time_I.opt.zero_grad()  # 清零梯度信息
        #     Loss = pinn_time_I.loss_computer.loss()
        #     if (epochs + 1) % 50 == 0:
        #         print('epochs:{}/{},Loss:{}'.format(epochs + 1, total_epochs,Loss.item()))
        #     Loss.backward(retain_graph=True)  # 反向计算出梯度
        #     pinn_time_I.opt.step()  # 更新参数
        #     opt_save = pinn_time_I.opt

        num_epochs_lbfgs = 100
        print('now using L_BFGS...')
        for epoch in range(num_epochs_lbfgs):
            loss = pinn_time_I.opt.step(pinn_time_I.closure)  # 更新权重,注意不要加括号!因为传递的是函数本身而不是函数的返回值！
            if (epoch + 1) % 25==0:
                print('epoch:', epoch + 1)
                print('loss:', loss)
        # pinn_time_I.opt = opt_save # 还原Adam

    # 保存模型
    pinn_time_I.save()

    # 加载并测试
    draw(pinn_time_I, 'save/time_I.pth', u, device)
    plt.show()




# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # 为np设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        torch.backends.cudnn.deterministic = True  # 使用确定性算法
        torch.backends.cudnn.benchmark = True  # cudnn基准(使用卷积时可能影响结果)


# 画图辅助函数
def draw(pinn, load_path,u, device):
    # pinn.model.to('cpu') # 卸载到cpu
    checkpoint = torch.load(load_path)  # 加载模型
    print('loading from', load_path)
    pinn.model.load_state_dict(checkpoint['model'])
    pinn.opt.load_state_dict(checkpoint['opt'])
    pinn.Epochs_loss = checkpoint['loss']
    pinn.model.eval()  # 启用评估模式
    with torch.no_grad():
        x = torch.arange(-1, 1.002, 0.002, device=device)  # 不包含最后一项
        t = torch.arange(0, 1.001, 0.001, device=device)
        grid_x, grid_t = torch.meshgrid(x, t)
        mesh_x = grid_x.reshape(-1, 1)
        mesh_t = grid_t.reshape(-1, 1)
        pred = pinn.model(torch.cat([mesh_x, mesh_t], dim=1)).reshape(grid_x.shape)
        N = 900  # 等高线密集程度`

        total_relative_l2 = torch.norm(pred - u) / torch.norm(u)
        print('total_relative_l2 =',total_relative_l2.item())

        plt.figure()
        plt.contourf(grid_t.cpu(), grid_x.cpu(), pred.cpu(), N, cmap='jet')
        plt.colorbar()
        plt.title("pred")
        plt.xlabel("t")
        plt.ylabel("x")

        plt.figure()
        plt.contourf(grid_t.cpu(), grid_x.cpu(), u.cpu(), N, cmap='jet')
        plt.colorbar()
        plt.title("real")
        plt.xlabel("t")
        plt.ylabel("x")

        plt.figure()
        abs_error = torch.abs(pred - u)
        plt.contourf(grid_t.cpu(), grid_x.cpu(), abs_error.cpu(), N, cmap='jet')
        plt.colorbar()
        plt.title("Abs error")
        plt.xlabel("t")
        plt.ylabel("x")

        # plt.figure()
        # plt.plot(pinn.Epochs_loss[:, 0], pinn.Epochs_loss[:, 1])
        # plt.xlabel('epochs')
        # plt.ylabel('loss')
        # plt.title('losses with epochs')



if __name__ == '__main__':
    test()
