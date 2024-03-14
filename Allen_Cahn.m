% 计算Allen-Cahn方程：
% u_t - 0.0001*u_xx + 5*u^3 - 5*u = 0
% u(0,x) = x^2 * cos(pi * x)
% u(t,-1) = u(t,1)
% u_x(t,-1) = u_x(t,1)
%% 初始化
clc;clear
delta_t = 0.0001;
delta_x = 0.001;
x = -1:delta_x:1;
t = 0:delta_t:1;
[T,X] = meshgrid(t,x);%索引是(x,t)
u = zeros(size(X));
u(:,1) = X(:,1).^2.*cos(pi * X(:,1));
u(1,1) = 1;
u(end,1) = 1;
%% 迭代求解
for t = 2:1:size(u,2)
    for x = 2:1:size(u,1) - 1
        u(x,t) = delta_t * ((0.0001 * (u(x+1,t-1) - 2*u(x,t-1) + u(x-1,t-1))/(delta_x^2)) + 5*u(x,t-1) - 5*u(x,t-1)^3) + u(x,t-1);
    end
    u(end,t) = (u(end - 1,t - 1) + u(1,t - 1))/2;
    u(1,t) = u(end,t);
end
%% 画图
x = [0 1];
y = [-1 1];
imagesc(x,y,u)
colorbar
xlabel('t')
ylabel('x')
title('u(x,t)')