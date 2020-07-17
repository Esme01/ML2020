'''
source:
https://courses.cs.washington.edu/courses/cse446/19au/section9.html
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(446)#为CPU设置种子用于生成随机数，以使得结果是确定的
np.random.seed(446)


#################Tensor与Numpy##################
x_numpy = np.array([0.1,0.2,0.3])
x_torch = torch.tensor([0.1,0.2,0.3])
print('x_numpy, x_torch')
print(x_numpy, x_torch)
print()

# tensor与ndarray的相互转换
print('to and from numpy and pytorch')
print(torch.from_numpy(x_numpy), x_torch.numpy())
print()

# tensor与torch的基本运算
y_numpy= np.array([1,2,3])
y_torch = torch.tensor([1,2,3])
print("x+y")
print(x_numpy + y_numpy, x_torch + y_torch)
print()

# pytorch和numpy相似的函数
print("norm")
print(np.linalg.norm(x_numpy), torch.norm(x_torch))
print()

# 进行某一维度上的操作时
# numpy使用axis ; torch使用dim
print("mean along the 0th dimension")
x_numpy = np.array([[1,2],[3,4.]])
x_torch = torch.tensor([[1,2],[3,4.]])
print(np.mean(x_numpy, axis=0), torch.mean(x_torch, dim=0))
print()

# tensor.view()与np.reshape()作用类似
# -1依然被自动计算
N, C, W, H = 10000, 3, 28, 28
X = torch.randn((N, C, W, H))

print(X.shape)
print(X.view(N, C, 784).shape)
print(X.view(-1, C, 784).shape)
print()


#################计算图##################
# 计算图是一种将数学表达式以图的形式表示的方式
a = torch.tensor(2.0, requires_grad=True)
# 设置requires_grad=True使得pytorch知道要保存计算图
b = torch.tensor(1.0, requires_grad=True)
c = a + b
d = b + 1
e = c * d
print('c', c)
print('d', d)
print('e', e)
print()


#################pytorch自动计算梯度##################
def f(x):
    return (x-2)**2

def fp(x):
    return 2*(x-2)

x = torch.tensor([1.0], requires_grad=True)

y = f(x)
# 在计算图的结果y变量调用backward()，计算y的所有梯度
y.backward()

print('Analytical f\'(x):', fp(x))
# grad属性输出对应梯度
print('PyTorch\'s f\'(x):', x.grad)
print()

def g(w):
    return 2*w[0]*w[1] + w[1]*torch.cos(w[0])

def grad_g(w):
    return torch.tensor([2*w[1] - w[1]*torch.sin(w[0]), 2*w[0] + torch.cos(w[0])])

w = torch.tensor([np.pi, 1], requires_grad=True)

z = g(w)
z.backward()

print('Analytical grad g(w)', grad_g(w))
print('PyTorch\'s grad g(w)', w.grad)
print()


#################使用梯度##################

x = torch.tensor([5.0],requires_grad=True)
step_size = 0.25
print('iter,\tx,\tf(x),\tf\'(x),\tf\'(x) pytorch')
for i in range(15):
    y = f(x) # y = (x-2)^2
    y.backward()
    print('{},\t{:.3f},\t{:.3f},\t{:.3f},\t{:.3f}'.
          format(i, x.item(), f(x).item(), fp(x).item(), x.grad.item()))
    x.data -= step_size * x.grad

    #使用后将变量的梯度归零，因为backward()的结果是累加而不是覆盖
    x.grad.detach_()
    x.grad.zero_()


#################线性回归##################
# 构造一个带有噪声的数据集
d = 2
n = 50
X = torch.randn(n,d)
true_w = torch.tensor([[-1.0],[2.0]])
y = X.mm(true_w) + torch.randn(n,1) * 0.1 # 与使用X @ true_w相同
print('X shape', X.shape)
print('y shape', y.shape)
print('w shape', true_w.shape)

# 数据集可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0].numpy(), X[:,1].numpy(), y.numpy(), c='r', marker='o')
ax.set_xlabel('$X_1$')
ax.set_ylabel('$X_2$')
ax.set_zlabel('$Y$')

plt.title('Dataset')
plt.show()

# 合理性验证
def model(X, w):
    return X @ w

# the residual sum of squares loss function
def rss(y, y_hat):
    return torch.norm(y - y_hat)**2 / n

# analytical expression for the gradient
def grad_rss(X, y, w):
    return -2*X.t() @ (y - X @ w) / n

w = torch.tensor([[1.], [0]], requires_grad=True)
y_hat = model(X, w)

loss = rss(y, y_hat)
loss.backward()

print('Analytical gradient', grad_rss(X, y, w).detach().view(2).numpy())
print('PyTorch\'s gradient', w.grad.view(2).numpy())


#################使用梯度下降的线性回归##################
step_size = 0.1
print('iter,\tloss,\tw')
for i in range(20):
    y_hat = model(X,w)
    loss = rss(y,y_hat)
    loss.backward()
    w.data -= step_size * w.grad
    print('{},\t{:.2f},\t{}'.format(i, loss.item(), w.view(2).detach().numpy()))

    w.grad.detach_()
    w.grad.zero_()
print('\ntrue w\t\t', true_w.view(2).numpy())
print('estimated w\t', w.view(2).detach().numpy())

# 可视化
def visualize_fun(w, title, num_pts=20):
    x1, x2 = np.meshgrid(np.linspace(-2, 2, num_pts), np.linspace(-2, 2, num_pts))
    X_plane = torch.tensor(np.stack([np.reshape(x1, (num_pts ** 2)), np.reshape(x2, (num_pts ** 2))], axis=1)).float()
    y_plane = np.reshape((X_plane @ w).detach().numpy(), (num_pts, num_pts))

    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(x1, x2, y_plane, alpha=0.2)

    ax = plt.gca()
    ax.scatter(X[:, 0].numpy(), X[:, 1].numpy(), y.numpy(), c='r', marker='o')

    ax.set_xlabel('$X_1$')
    ax.set_ylabel('$X_2$')
    ax.set_zlabel('$Y$')

    plt.title(title)
    plt.show()

visualize_fun(true_w, 'Dataset and true $w$')


#################torch.nn##################