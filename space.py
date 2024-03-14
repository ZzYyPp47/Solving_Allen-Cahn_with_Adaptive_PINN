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
    N_pde = 2000
    N_bound = 100
    N_ini = 500
    N_real = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 自动选择可用设备,优先GPU
    learning_rate = 0.001
    init_method = torch.nn.init.kaiming_uniform_ # 设置神经网络参数初始化方法
    # total_epochs = 400 # 运用Adam优化的次数
    total_space = 100 # 空间采样次数
    tor = 0.0001 # loss阈值
    loss_func = nn.MSELoss().to(device) # 确定损失计算函数
    loss_weight = [1,1,100,0] # loss各项权重(pde,bound,ini,real)

    file = 'data.mat'
    u = loadmat(file, mat_dtype=True)['u']
    u = torch.from_numpy(u).cuda()

    x = torch.arange(-1, 1.01, 0.01,requires_grad=True ,device=device)
    t = torch.arange(0, 1.01, 0.01,requires_grad=True ,device=device)
    grid_x, grid_t = torch.meshgrid(x, t)
    mesh_x = grid_x.reshape(-1, 1)
    mesh_t = grid_t.reshape(-1, 1)

    num = 50

    # 建立模型
    set_seed(seed)  # 设置确定的随机种子

    space = FNN(Arc, func, device)

    opt_space = torch.optim.LBFGS(space.model.parameters(), lr=0.01, max_iter=20, line_search_fn='strong_wolfe')

    point_space = create_point(N_pde, N_bound, N_ini, N_real, device)

    pinn_space = pinn('space', total_space, tor, loss_func, space, point_space, opt_space, init_method, loss_weight, device)
    # 务必确定model与point位于同一设备!!
    min_loss = float("inf")

    # 训练模型
    for space_epochs in range(total_space):
        # for epochs in range(total_epochs):
        #     # 计算损失并反向传播
        #     pinn_space.model.train()
        #     pinn_space.opt = torch.optim.Adam(params=space.parameters(), lr=learning_rate)
        #     pinn_space.opt.zero_grad()  # 清零梯度信息
        #     Loss = pinn_space.loss_computer.loss()
        #     if (epochs + 1) % 100 == 0:
        #         print('space_epoch:{}/{},epoch:{}/{},loss:{}'.format(space_epochs + 1,total_space,epochs + 1,total_epochs,Loss.item()))
        #     Loss.backward(retain_graph=True)  # 反向计算出梯度
        #     pinn_space.opt.step()  # 更新参数
        #
        # opt_save = pinn_space.opt

        num_epochs_lbfgs = 100
        # print('now using L_BFGS...')
        for epochs_lbfgs in range(num_epochs_lbfgs):
            loss = pinn_space.opt.step(pinn_space.closure)  # 更新权重,注意不要加括号!因为传递的是函数本身而不是函数的返回值！
            # 自动保存最优模型
            if loss < min_loss:
                min_loss = loss
                # min_loss_epochs = epochs + 1
                min_space_epochs = space_epochs + 1
                # 保存模型
                pinn_space.save()
                print('When space_epochs:{}/{},epochs_lbfgs:{}/{},found min Loss:{}'.format(space_epochs + 1,total_space,epochs_lbfgs + 1, num_epochs_lbfgs, loss))

            if (epochs_lbfgs + 1) % 25 == 0:
                print('space_epoch:{},epochs_lbfgs:{}'.format(space_epochs + 1,epochs_lbfgs + 1))
                print('loss:', loss)
        # pinn_space.opt = opt_save


        pinn_space.model.eval()
        # 计算pde损失
        pde_err = loss_pde(pinn_space,mesh_x,mesh_t).reshape(grid_x.shape)
        _, flat_indices = torch.topk(pde_err.view(-1), num, largest=True, sorted=True)
        flat_indices = flat_indices.reshape(-1, 1)

        # 将这些一维索引转换为二维索引，对应原始矩阵中的位置
        rows = flat_indices // pde_err.shape[0]
        cols = flat_indices % pde_err.shape[1]
        err_x = x[rows]
        err_t = t[cols]
        pinn_space.point.pde_x = torch.cat([pinn_space.point.pde_x,err_x],dim=0)
        pinn_space.point.pde_t = torch.cat([pinn_space.point.pde_t,err_t],dim=0)
        # print('ADDED {} POINT.'.format(num))

    print('IN SPACE_EPOCHS:{},SAVED BEST MODEL. (LOSS:{})'.format(space_epochs + 1,min_loss))
    # 保存模型
    # pinn_space.save()

    # 加载并测试
    draw(pinn_space,'save/space.pth',u,device)
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

# 计算求导
def gradient(func,var,order = 1):
    if order == 1:
        return torch.autograd.grad(func,var,grad_outputs=torch.ones_like(func),create_graph=True,only_inputs=True)[0]
    else:
        out = gradient(func,var)# 不要加order(以正常计算1阶导),否则会无限循环调用！
        return gradient(out,var,order - 1)

# 计算pde_loss
def loss_pde(pinn,x,t):
    u = pinn.model(torch.cat([x, t], dim=1))
    u_t = gradient(u, t)
    u_xx = gradient(u, x, 2)
    return u_t - 0.0001 * u_xx + 5 * (u ** 3) - 5 * u

if __name__ == '__main__':
    test()
