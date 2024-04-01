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
from torch.utils.data import DataLoader
from loss import *
from math import ceil



# 加载并测试...
def test():
    # 初始化
    seed = 0
    Arc = [2,30,30,30,30,30,30,30,30,1] # 神经网络架构
    func = nn.Tanh() # 确定激活函数
    N_pde = 10000
    N_bound = 100
    N_ini = 500
    N_real = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 自动选择可用设备,优先GPU
    learning_rate = 0.001
    init_method = torch.nn.init.kaiming_uniform_ # 设置神经网络参数初始化方法
    total_epochs = 1000
    tor = 0.0001 # loss阈值
    loss_func = nn.MSELoss().to(device) # 确定损失计算函数
    loss_weight = [1,1,100,0] # loss各项权重(pde,bound,ini,real)
    batch_size = 50
    file = 'data.mat'
    u = loadmat(file, mat_dtype=True)['u']
    u = torch.from_numpy(u).cuda()


    # 建立模型
    set_seed(seed)  # 设置确定的随机种子

    mini = FNN(Arc, func, device)

    point_mini = create_point(N_pde, N_bound, N_ini, N_real, device)

    opt_mini = torch.optim.Adam(params=mini.parameters(), lr=learning_rate)

    pinn_mini = pinn('mini', None, None, loss_func, mini, None, opt_mini, init_method, loss_weight, device)

    min_loss = float("inf")
    # 务必确定model与point位于同一设备!!



    # 训练模型(自动保存最佳模型:argmin loss)
    pinn_mini.model.train()
    for epochs in range(total_epochs):
        batchset = DataLoader(point_mini,batch_size= batch_size, shuffle= True,drop_last=False)
        for idx , batch_data in enumerate(batchset):
            pde_x, pde_t = batch_data['pde']
            bound_x1, bound_tx1, bound_xm1, bound_txm1 = batch_data['bound']
            ini_x, ini_t = batch_data['ini']

            # 初始化损失计算器
            loss_computer = LossCompute(mini, loss_func,None, loss_weight)
            loss_computer.point.pde_x = pde_x
            loss_computer.point.pde_t = pde_t
            loss_computer.point.bound_x1 = bound_x1
            loss_computer.point.bound_tx1 = bound_tx1
            loss_computer.point.bound_xm1 = bound_xm1
            loss_computer.point.bound_txm1 = bound_txm1
            loss_computer.point.ini_x = ini_x
            loss_computer.point.ini_t = ini_t
            pinn_mini.loss_computer = loss_computer

            # 计算损失并反向传播
            loss = pinn_mini.train()

    #         # 自动保存最优模型
    #         if loss < min_loss:
    #             min_loss = loss
    #             min_loss_epochs = epochs + 1
    #             min_loss_batch = idx + 1
    #             # 保存模型
    #             pinn_mini.save()
    #             # print('When epochs:{}/{},batch:{}/{},found min Loss:{}'.format(epochs + 1,total_epochs,idx + 1,ceil(len(point_mini)/batch_size),loss))
            if (idx + 1) % 100 == 0: # 每100个batch输出一次
                print('epochs:{}/{},batch:{}/{},Loss:{}'.format(epochs + 1,total_epochs,idx + 1,ceil(len(point_mini)/batch_size),loss))
            if idx + 1 == batch_size & len(pinn_mini.Epochs_loss) > 0:
                pinn_mini.Epochs_loss.append([pinn_mini.Epochs_loss[-1][0] + 1, loss])
            else:
                pinn_mini.Epochs_loss.append([1, loss])
    # print('IN EPOCHS:{},BATCH:{},SAVED BEST MODEL. (LOSS:{})'.format(min_loss_epochs,min_loss_batch,min_loss))

    # 初始化L-BFGS优化器
    pinn_mini.opt = torch.optim.LBFGS(pinn_mini.model.parameters(), history_size=100, tolerance_change=0, tolerance_grad=1e-08, max_iter=25000, max_eval=30000)
    num_epochs_lbfgs = 1
    print('now using L_BFGS...')
    for epoch in range(num_epochs_lbfgs):
        pinn_mini.opt.step(pinn_mini.closure)  # 更新权重,注意不要加括号!因为传递的是函数本身而不是函数的返回值！
        # print('epoch:',epoch + 1)
        # print('loss:',loss)
    pinn_mini.save()

    # 加载并测试
    draw(pinn_mini,'save/mini.pth',u,device)
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
    pinn.Epochs_loss = np.array(pinn.Epochs_loss)
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

        plt.figure()
        plt.semilogy(pinn.Epochs_loss[:, 0], pinn.Epochs_loss[:, 1])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('losses with epochs')

if __name__ == '__main__':
    test()
