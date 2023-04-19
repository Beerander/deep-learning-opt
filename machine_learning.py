import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, transforms
from d2l import torch as d2l


from optim_scratch import *


# 使用Fashion-MNIST数据集
def load_Fashion_MNIST(data_dir, batch_size, bgd=False):
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    train_ds = datasets.FashionMNIST(data_dir, train=True, transform=trans, download=False)
    test_ds = datasets.FashionMNIST(data_dir, train=False, transform=trans, download=False)
    train_dataloader = data.DataLoader(train_ds, batch_size, shuffle=True)
    test_dataloader = data.DataLoader(test_ds, batch_size if not bgd else 256, shuffle=False)
    return train_dataloader, test_dataloader


# 网络结构
class Zynet(nn.Module):
    def __init__(self):
        super(Zynet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 72, (3, 3)), nn.BatchNorm2d(72), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(72, 256, (3, 3)), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 128, (3, 3)), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 84), nn.BatchNorm1d(84), nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_opt(net, train_iter, test_iter, num_epochs, hyperparameters, device, optimizer):
    net.to(device)
    # 记录训练过程
    metrics_train = [[], [], []]
    metrics_test = [[], []]
    # 优化算法
    optimizer = optimizer(net.parameters(), hyperparams=hyperparameters)
    # optimizer = optim.SGD(net.parameters(), lr=hyperparameters['lr'])
    loss = nn.CrossEntropyLoss()  # 损失函数
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # 累加器，用于累积训练损失与准确率
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y).mean()
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                # 每个epoch保存5个点，使图像更加细节
                if i % (len(train_iter) // 5) == 0:
                    train_loss = metric[0] / metric[2]
                    train_acc = metric[1] / metric[2]
                    metrics_train[0].append(epoch + (i/len(train_iter)))
                    metrics_train[1].append(train_loss)
                    metrics_train[2].append(train_acc)
            timer.stop()
        # 测试
        test_acc = evaluate(net, test_iter)
        metrics_test[0].append(epoch)
        metrics_test[1].append(test_acc)
        print(f"epoch {epoch+1}: {sum(timer.times[-len(train_iter):]):.2f}s")
    print(f"loss:{train_loss:.3f}, train_acc:{train_acc:.3f}, test_acc:{test_acc:.3f}\n"
          f"{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}\n"
          f"{timer.sum():.2f}s total")
    # 作图并保存
    plot_result(optimizer.name, metrics_train, metrics_test)


# 测试函数
def evaluate(n, t_iter):
    n.eval()
    device = next(iter(n.parameters())).device
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in t_iter:
            X, y = X.to(device), y.to(device)
            metric.add(d2l.accuracy(n(X), y), y.numel())
    return metric[0] / metric[1]


# 将结果转换为图像并保存
def plot_result(opt_name, trains, tests):
    font1 = {'family': 'Times New Roman', 'size': 28}
    font2 = {'family': 'Times New Roman', 'size': 24}
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.xlabel('epochs', font1)
    plt.ylabel('acc (%)', font1)
    plt.plot(tests[0], np.array(tests[1]) * 100, color='#BFBF00', label='test acc')
    plt.plot(trains[0], np.array(trains[2]) * 100, color='red', label='train acc')
    plt.legend(prop=font2)
    ax = plt.gca()  # gca:get current axis得到当前轴
    # 设置图片的右边框和上边框为不显示
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.subplot(1, 2, 2)
    plt.xlabel('epochs', font1)
    plt.ylabel('loss', font1)
    plt.plot(trains[0], trains[1], color='red', label='train loss')
    plt.legend(prop=font2)
    ax = plt.gca()  # gca:get current axis得到当前轴
    # 设置图片的右边框和上边框为不显示
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.savefig(f'./imgs_5epoch/{opt_name}.png')
    plt.show()


def train(optimizer, hyperparameters):
    """
    :param optimizer: 需要使用的优化器
    :param hyperparameters: 优化器对应的超参数
    :return: None
    """
    root = './'  # 设置数据集目录
    num_epoch = 5  # 训练周期数
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 在cuda上训练
    train_dl, test_dl = load_Fashion_MNIST(root, batch_size=256)  # 读取数据集，batch_size=256
    net = Zynet()  # 初始化网络
    train_opt(net, train_dl, test_dl, num_epoch, hyperparameters, device, optimizer)  # 调用训练函数


def main():
    # 列出所有实现的优化器并一起训练，整个过程大概225s
    # 如需单个训练将150行取消注释并注释151 152行 将x替换为想要使用的优化器即可
    op_dict = [SGD, SGD, SGD, AdaGrad, RMSProp, AdaDelta, Adam, AdaMax, NAdam]
    hyperparams_dict = [{'lr': 0.01}, {'lr': 0.01, 'momentum': 0.9}, {'lr': 0.01, 'momentum': 0.9, 'nesterov': True},
                        {'lr': 0.01}, {'lr': 0.01}, {'gamma': 0.9}, {'lr': 0.01}, {'lr': 0.01}, {'lr': 0.01}]
    # train(op_dict[x], hyperparams_dict[x])
    for op, hy in zip(op_dict, hyperparams_dict):
        train(op, hy)


if __name__ == '__main__':
    main()
