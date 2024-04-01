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


# 加载并测试...
def test():
    # 初始化
    seed = 0
    Arc = [2,128,128,128,128,1] # 神经网络架构
    func = nn.Tanh() # 确定激活函数
    N_pde = 10000
    N_bound = 200
    N_ini = 200
    N_real = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 自动选择可用设备,优先GPU
    learning_rate = 0.001
    init_method = torch.nn.init.kaiming_uniform_ # 设置神经网络参数初始化方法
    total_epochs = 40000
    tor = 0.0001 # loss阈值
    loss_func = nn.MSELoss().to(device) # 确定损失计算函数
    loss_weight = [1,1,100,0] # loss各项权重(pde,bound,ini,real)

    # 建立模型
    set_seed(seed)  # 设置确定的随机种子

    weighted = FNN(Arc, func, device)

    point_weighted = create_point(N_pde, N_bound, N_ini, N_real, device)

    opt_weighted = torch.optim.Adam(params = weighted.parameters(), lr=learning_rate)

    pinn_weighted = pinn('weighted', total_epochs, tor, loss_func, weighted, point_weighted, opt_weighted, init_method, loss_weight, device)
    # 务必确定model与point位于同一设备!!

    pinn_weighted.train_all()
    # 初始化L-BFGS优化器
    pinn_weighted.opt = torch.optim.LBFGS(pinn_weighted.model.parameters(), history_size=100, tolerance_change=0, tolerance_grad=1e-08, max_iter=40000, max_eval=50000)
    num_epochs_lbfgs = 1
    print('now using L_BFGS...')
    for epoch in range(num_epochs_lbfgs):
        pinn_weighted.opt.step(pinn_weighted.closure)  # 更新权重,注意不要加括号!因为传递的是函数本身而不是函数的返回值！
        # print('epoch:',epoch + 1)
        # print('loss:',loss)
    pinn_weighted.save()

    # 加载并测试
    draw(pinn_weighted,'save/weighted.pth',device)
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
def draw(pinn, load_path, device):
    # pinn.model.to('cpu') # 卸载到cpu
    checkpoint = torch.load(load_path)  # 加载模型
    print('loading from', load_path)
    pinn.model.load_state_dict(checkpoint['model'])
    pinn.opt.load_state_dict(checkpoint['opt'])
    pinn.Epochs_loss = checkpoint['loss']
    pinn.Epochs_loss = np.array(pinn.Epochs_loss)
    pinn.model.eval()  # 启用评估模式
    with torch.no_grad():
        x = torch.arange(-1, 1.002, 0.002, device=device)  # 不包含最后一项
        t = torch.arange(0, 1.001, 0.001, device=device)
        file = 'data.mat'
        u = loadmat(file, mat_dtype=True)['u']
        u = torch.from_numpy(u).cuda()
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

        plt.figure()
        plt.semilogy(pinn.Epochs_loss[:, 0], pinn.Epochs_loss[:, 1])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('losses with epochs')



if __name__ == '__main__':
    test()
