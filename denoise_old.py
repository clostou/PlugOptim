import os
import sys
from io import BytesIO
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.multiprocessing import Process, Queue
from torchinfo import summary

sys.path.append('/home/zhuofeng/lgq/python/')

from plugDesign import External, profile_to_msh
from bellDesign import CharacteristicsNozzle
from cfd_toolbox.submit import *
from cfd_toolbox.utils import *
from cfd_toolbox.gasdy import *
from cfd_toolbox.plot import *
from ML.regress import CurveFitting
from ML.reduce import PCA


class ResidualBlock(nn.Module):

    def __init__(self, input_channels, num_channels, drop_p=0.5):
        super(ResidualBlock, self).__init__()
        self.linear_up = nn.Linear(input_channels, num_channels)
        self.linear_down = nn.Linear(num_channels, input_channels)
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, X):
        Y = F.relu(self.linear_up(X))
        Y = self.linear_down(self.dropout(Y))
        return X + Y


class DenoiseNet(nn.Module):

    def __init__(self, input_channels, num_channels, out_channels, block_n=3):
        super(DenoiseNet, self).__init__()
        self.block_n = block_n
        self.residual = nn.Sequential(*[ResidualBlock(input_channels, num_channels, drop_p=0.2)
                                        for _ in range(block_n)])
        self.linear = nn.Linear(input_channels, out_channels)

    def forward(self, X):
        X = self.residual(X)
        return self.linear(F.tanh(X))


class FullConnectNet(nn.Module):

    def __init__(self, input_channels, num_channels, out_channels, layer_n=3):
        super(FullConnectNet, self).__init__()
        self.layer_n = layer_n
        self.linear_up = nn.Linear(input_channels, num_channels)
        hidden = []
        for _ in range(layer_n):
            hidden.extend([nn.Linear(num_channels, num_channels), nn.ReLU()])
        self.hidden = nn.Sequential(*hidden)
        self.linear_down = nn.Linear(num_channels, out_channels)

    def forward(self, X):
        X = F.tanh(self.linear_up(X))
        X = self.hidden(X)
        return self.linear_down(X)


def normalize(data, type):
    """归一化给定的数据集（按行排列），返回归一化函数及反函数"""
    if type == 'maxmin':
        _min = data.min(axis=0)
        _max = data.max(axis=0)
        if isinstance(data, torch.Tensor):
            _min = _min[0]
            _max = _max[0]
        _mean = 0.5 * (_max + _min)
        _std = 0.5 * (_max - _min)
    elif type == 'zscore':
        _mean = data.mean(axis=0)
        _std = data.std(axis=0)
    else:
        print("Unknown type of normalization: %s (Supported type: 'maxmin', 'zscore')" % type)
        _mean = 0.
        _std = 1.
    f = lambda x: (x - _mean) / _std
    f_inv = lambda x: _std * x + _mean
    return f, f_inv


class Normalize:
    """
    归一化给定的数据集（按行排列），返回包含归一化函数及反函数的实例
    """

    def __init__(self, data, type):
        _data = np.array(data, dtype=np.float64)
        if type == 'maxmin':
            _min = np.min(_data, axis=0)
            _max = np.max(_data, axis=0)
            _mean = 0.5 * (_max + _min)
            _std = 0.5 * (_max - _min)
        elif type == 'zscore':
            _mean = np.mean(_data, axis=0)
            _std = np.std(_data, axis=0)
        else:
            print("Unknown type of normalization: %s (Supported type: 'maxmin', 'zscore')" % type)
            _mean = np.zeros(_data.shape[1], dtype=np.float64)
            _std = np.ones(_data.shape[1], dtype=np.float64)

        self.mean = torch.from_numpy(_mean)
        self.std = torch.from_numpy(_std)
        self._mean = _mean
        self._std = _std

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return (x - self.mean) / self.std
        else:
            return (x - self._mean) / self._std

    def inv(self, x):
        if isinstance(x, torch.Tensor):
            return self.std * x + self.mean
        else:
            return self._std * x + self._mean


class Denoise:
    """
    使用神经网络或者局部加权线性回归对噪声数据集进行重构以去除误差

    注：仅支持小误差修正，因此默认会将误差较大的输入数据提前剔除
    """

    def __init__(self, data_in: np.ndarray, data_out: np.ndarray, data_noise: np.ndarray,
                 noise_threshold: float = 0.1, noise_deprecated: float = 10.):
        self.noise_deprecated = noise_deprecated
        self.noise_threshold = noise_threshold

        # 输入数据对齐
        data_in = np.array(data_in)
        data_out = np.array(data_out)
        data_noise = np.array(data_noise)
        self.m = min(data_in.shape[0], data_noise.shape[0], data_out.shape[0])
        self.n_in = data_in.shape[1]
        self.data_in = data_in[: self.m].astype(np.float64)
        self.n_noise = data_noise.shape[1]
        self.data_noise = data_noise[: self.m].astype(np.float64)
        self.n_out = data_out.shape[1]
        self.data_out = data_out[: self.m].astype(np.float64)

        # 剔除大误差输入
        if noise_deprecated > noise_threshold:
            retain_row = np.prod(self.data_noise <= noise_deprecated, axis=1, dtype=bool)
            self.data_in = self.data_in[retain_row]
            self.data_noise = self.data_noise[retain_row]
            self.data_out = self.data_out[retain_row]
            if len(retain_row) < self.m:
                print("%d samples have been deprecated")
                self.m = len(retain_row)

        self.index_in = []
        self.index_out = []
        self.index_noise = []
        self.valid_row = np.ones(self.m)
        self.data_in_scale = None
        self.data_out_scale = None
        self.model = None

        self.data_denoise = None
        self.error = float('inf')

    def by_net(self, lr: float = 5e-3, num_epochs: int = 200,
               in_i: Union[None, List[int]] = None,
               out_i: Union[None, List[int]] = None,
               noise_i: Union[None, List[int]] = None,
               print_detail: bool = True):

        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('linear'))

        self.index_in = list(range(self.n_in) if in_i is None else in_i)
        self.index_out = list(range(self.n_out) if out_i is None else out_i)
        self.index_noise = list(range(self.n_noise) if noise_i is None else noise_i)

        data_in = self.data_in[: , self.index_in]
        data_out = self.data_out[: , self.index_out]
        data_noise = self.data_noise[: , self.index_noise]
        self.data_in_scale = normalize(data_in, type='maxmin')
        self.data_out_scale = normalize(data_out, type='zscore')
        valid_row = np.prod(data_noise <= self.noise_threshold, axis=1, dtype=bool)

        dataloader = DataLoader(dataset=TensorDataset(torch.from_numpy(np.hstack([self.data_in_scale[0](data_in),
                                                                                  data_noise])),
                                                      torch.from_numpy(self.data_out_scale[0](data_out))),
                                batch_size=1, shuffle=True)
        input_n = data_in.shape[1] + data_noise.shape[1]
        # 设定网络架构
        net = DenoiseNet(input_channels=input_n, num_channels=10 * input_n,
                         out_channels=data_out.shape[1], block_n=3)
        # net = FullConnectNet(input_channels=input_n, num_channels=5 * input_n,
        #                      out_channels=data_out.shape[1], layer_n=3)
        if print_detail:
            summary(net, (1, input_n), device='cpu')
        net.double()
        net.apply(init_weights)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.00005)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.2)
        loss_f = nn.SmoothL1Loss()    # 使用平滑L1损失函数以减小离群样本点的影响

        net.train()
        for epoch in range(num_epochs):
            total_loss, n = 0., 0
            for i, (X, Y) in enumerate(dataloader):
                optimizer.zero_grad()
                Y_hat = net(X)
                loss = loss_f(Y_hat, Y)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    total_loss += float(loss) * X.shape[0]
                    n += X.shape[0]
            epoch_loss = total_loss / n
            scheduler.step()
            if print_detail:
                print("Epoch %i: loss %.4f" % (epoch + 1, epoch_loss))
        net.eval()
        total_loss, n = 0., 0
        for i, (X, Y) in enumerate(dataloader):
            Y_hat = net(X)
            loss = loss_f(Y_hat, Y)
            with torch.no_grad():
                total_loss += float(loss) * X.shape[0]
                n += X.shape[0]
        print("Final average loss = %.3e" % (total_loss / n))
        self.model = net
        self.valid_row = valid_row

        X = np.hstack([self.data_in_scale[0](data_in), data_noise])
        Y = net(torch.from_numpy(X)).detach().numpy()
        error = np.mean(np.abs((self.data_out_scale[1](Y) - data_out) / data_out), axis=0)
        print("Mean reconstruction error (with noise): " + ', '.join(map(lambda x: '{:.3f}%'.format(x), error * 100)))
        data_denoise = self.sample(data_in)
        error = np.mean(np.abs(((data_denoise - data_out) / data_out)[valid_row]), axis=0)
        print("Mean reconstruction error: " + ', '.join(map(lambda x: '{:.3f}%'.format(x), error * 100)))
        self.data_denoise = np.hstack([data_in, np.zeros(data_noise.shape, dtype=np.float64), data_denoise])
        self.error = error

    def by_lwlr(self, k: float = 1.0,
                in_i: Union[None, List[int]] = None,
                out_i: int = 0,
                noise_i: int = 0):
        self.index_in = list(range(self.n_in) if in_i is None else in_i)
        self.index_out = [out_i]
        self.index_noise = [noise_i]

        valid_row = self.data_noise[: , noise_i] <= self.noise_threshold
        data_in = self.data_in[: , self.index_in]
        self.data_in_scale = normalize(data_in[valid_row], type='zscore')
        data_noise = self.data_noise[: , self.index_noise]
        data_noise[~ valid_row] = 0.
        data_out = self.data_out[: , self.index_out]
        curve = CurveFitting(data_out[valid_row, 0], *self.data_in_scale[0](data_in[valid_row]).T)
        curve.Regress(k=k, show=False)
        self.model = curve
        self.valid_row = valid_row

        data_denoise = self.sample(data_in)
        error = np.mean(np.abs(((data_denoise - data_out) / data_out)[valid_row]))
        print("Mean reconstruction error: %.3f%%" % (error * 100))
        self.data_denoise = np.hstack([data_in, data_noise, data_denoise])
        self.error = error

    def sample(self, data_in: np.ndarray, input_scale: bool = True):
        data_in = np.array(data_in)
        if input_scale:
            data_in = self.data_in_scale[0](data_in)
        if isinstance(self.model, DenoiseNet) or isinstance(self.model, FullConnectNet):
            X = np.hstack([data_in,
                           np.zeros((data_in.shape[0], len(self.index_noise)), dtype=np.float64)])
            Y = self.model(torch.from_numpy(X)).detach().numpy()
            return self.data_out_scale[1](Y)
        elif isinstance(self.model, CurveFitting):
            y = []
            for p in data_in:
                y.append(self.model.Estimate(list(p)))
            return np.array([y]).T
        else:
            raise ValueError("Model is not initialized or has unknown type")

    def plot(self, n: int = 100,
             x_label: Union[None, List[str]] = None,
             y_label: Union[None, List[str]] = None):
        n_in = len(self.index_in)
        n_out = len(self.index_out)
        fig, axes = plt.subplots(n_out, n_in, sharex='col', sharey='row', figsize=(10, 8))
        axes = np.reshape(axes, (n_out, -1))
        fig.suptitle(f"Origin cross section of fitting hyperplane (dim: {n_in} -> {n_out})")
        for x_i, in_i in enumerate(self.index_in):
            data_in_std = np.zeros((n, n_in))
            data_in_std[:, x_i] = np.linspace(-2, 2, n)
            data_in = self.data_in_scale[1](data_in_std)
            data_out = self.sample(data_in_std, input_scale=False)
            arr = self.data_in[:, self.index_in]
            arr = self.data_in_scale[0](arr)
            arr = np.delete(arr, x_i, axis=1)
            distance = np.linalg.norm(arr, axis=1)
            for y_i, out_i in enumerate(self.index_out):
                ax = axes[y_i, x_i]
                ax.plot(data_in[:, x_i], data_out[:, y_i])
                ax.scatter(self.data_in[:, in_i], self.data_out[:, out_i], s=12, c=distance, cmap='viridis')
                if y_i + 1 == n_out:
                    if isinstance(x_label, list):
                        ax.set_xlabel(x_label[x_i], fontsize=20)
                    else:
                        ax.set_xlabel("${in}_{%d}$" % in_i)
                if x_i == 0:
                    if isinstance(y_label, list):
                        ax.set_ylabel(y_label[y_i], fontsize=20)
                    else:
                        ax.set_ylabel("${out}_{%d}$" % out_i)
                ax.autoscale()
                ax.grid()
        fig.show()


def test_denoise():

    def by_net(denoise, in_idx=None, out_idx=None, n_test=100, n_point=100, x_label=None, y_label=None):
        n_in = denoise.n_in if in_idx is None else len(in_idx)
        n_out = denoise.n_out if out_idx is None else len(out_idx)
        fig, axes = plt.subplots(n_out, n_in, sharex='col', sharey='row', figsize=(10, 8))
        axes = np.reshape(axes, (n_out, -1))
        fig.suptitle(f"Origin cross section of fitting hyperplane (dim: {n_in} -> {n_out})")
        # 使用网络多次拟合，记录每次的投影曲线
        error_record = np.zeros((n_test, n_out))
        curve_x = np.zeros(axes.shape + (n_test, n_point))
        curve_y = np.zeros(axes.shape + (n_test, n_point))
        for z_i in range(n_test):
            print("[%d]" % (z_i + 1), end=' ')
            denoise.by_net(lr=0.005, num_epochs=400, in_i=in_idx, out_i=out_idx, print_detail=False)
            error_record[z_i, :] += denoise.error
            for x_i, in_i in enumerate(denoise.index_in):
                data_in_std = np.zeros((n_point, n_in))
                data_in_std[:, x_i] = np.linspace(-2, 2, n_point)
                data_in = denoise.data_in_scale[1](data_in_std)
                data_out = denoise.sample(data_in_std, input_scale=False)
                for y_i, out_i in enumerate(denoise.index_out):
                    curve_x[y_i, x_i, z_i] = data_in[:, x_i]
                    curve_y[y_i, x_i, z_i] = data_out[:, y_i]
        # 绘制多次采样结果的均值和方差
        error = np.mean(error_record, axis=0)
        error_var = np.sqrt(np.var(error_record, axis=0))
        data_in = np.mean(curve_x, axis=2)
        data_out = np.mean(curve_y, axis=2)
        data_out_var = np.sqrt(np.var(curve_y, axis=2))
        for x_i, in_i in enumerate(denoise.index_in):
            arr = denoise.data_in[:, denoise.index_in]
            arr = denoise.data_in_scale[0](arr)
            arr = np.delete(arr, x_i, axis=1)
            distance = np.linalg.norm(arr, axis=1)
            for y_i, out_i in enumerate(denoise.index_out):
                ax = axes[y_i, x_i]
                ax.scatter(denoise.data_in[:, in_i], denoise.data_out[:, out_i], s=12, c=distance, cmap='viridis')
                ax.plot(data_in[y_i, x_i, :], data_out[y_i, x_i, :])
                ax.fill_between(data_in[y_i, x_i, :],
                                data_out[y_i, x_i, :] - 2 * data_out_var[y_i, x_i, :],
                                data_out[y_i, x_i, :] + 2 * data_out_var[y_i, x_i, :],
                                facecolor='blue', alpha=0.2)  # 95.4%置信区间
                if y_i + 1 == n_out:
                    if isinstance(x_label, list):
                        ax.set_xlabel(x_label[x_i], fontsize=20)
                    else:
                        ax.set_xlabel("${in}_{%d}$" % in_i)
                if x_i == 0:
                    if isinstance(y_label, list):
                        ax.set_ylabel(y_label[y_i], fontsize=20)
                    else:
                        ax.set_ylabel("${out}_{%d}$" % out_i)
                ax.autoscale()
                ax.grid()
        axes[0, -1].legend(["Observation", "Prediction", "95% confidence interval"])
        print("-" * 16)
        print("Mean denoise error of %d tests:" % n_test,
              ', '.join(map(lambda x: '{:.3f}±{:.3f}%'.format(*x), zip(error * 100, error_var * 300))))
        fig.show()

    data = pd.read_csv(r'.\plug_result.csv')
    print(data.columns)
    denoise = Denoise(data[['inlet_p', 'atmo_p', 'Qm_max', 'Cf_max']],
                      data[['report-def-massflow', 'report-def-thrust', 'Cf', 'SpecImpulse']],
                      data[['report-def-continuity']])
    #denoise.by_net(lr=0.005, num_epochs=400, in_i=[0, 1], out_i=[2])
    #denoise.by_lwlr(k=0.2, in_i=[0, 1], out_i=2)
    #denoise.plot(x_label=["$p_0$ / $Pa$", "$p_e$ / $Pa$"], y_label=["$C_f$"])
    by_net(denoise, in_idx=[0, 1], out_idx=[2], x_label=["$p_0$ / $Pa$", "$p_e$ / $Pa$"], y_label=["$C_f$"])


class EarlyStopping:

    def __init__(self, patience=10, verbose=False, delta=0, skip_epoch=0):
        """
        EarlyStopping初始化，支持根据最小损失创建检查点.
        Args:
            patience (int): 当验证集损失在指定的epoch数内没有减少时触发早停.
            verbose (bool): 如果为True，则每次验证集损失改进时会打印一条消息.
            delta (float): 验证集损失改进的最小变化.
            skip_epoch (int): 跳过初始epoch数，不触发早停.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.skip_epoch = skip_epoch
        self.best_loss = None
        self.iter = 0
        self.counter = 0
        self.early_stop = False
        self.net_cache = None

    def __bool__(self):
        return self.early_stop

    def __call__(self, val_loss, checkpoint=None):
        if self.iter <= self.skip_epoch:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f'Validation loss decreased to {self.best_loss:.6f}. Resetting counter.')
            if checkpoint is not None:
                self.net_cache = BytesIO()
                torch.save(checkpoint.state_dict(), self.net_cache)
                self.net_cache.seek(0)
        self.iter += 1

    def resume(self, net):
        if self.net_cache is not None:
            net.load_state_dict(torch.load(self.net_cache))


class NetBagging:
    """
    基于降噪网络的集成学习器

    使用并行式集成学习算法Bagging，有利于降低神经网络这类非稳定基学习器的方差
    """

    def __init__(self, data_in, data_out, data_noise, model_n=10,
                 noise_threshold=0.1, max_thread=4):
        self.data_in = np.array(data_in)
        self.data_out = np.array(data_out)
        self.data_noise = np.array(data_noise)
        self.m = min(self.data_in.shape[0], self.data_noise.shape[0], self.data_out.shape[0])
        self.n_input = self.data_in.shape[1] + self.data_noise.shape[1]
        self.data_in_scale = normalize(self.data_in, type='zscore')
        self.data_out_scale = normalize(self.data_out, type='zscore')
        self.T = model_n
        self.max_thread = max_thread
        self.noise_threshold = noise_threshold
        self.net = None

    class Trainer(Process):
        """
        用于训练神经网络的子进程，不包含输入输出的归一化
        """

        def __init__(self, queue, data_in, data_out, n=1, lr=5e-3, max_epochs=500, verbose=False):
            super(NetBagging.Trainer, self).__init__(name='NetTrainer', daemon=True)
            self.queue = queue
            self.data_in = data_in
            self.data_out = data_out
            self.n = n
            self.lr = lr
            self.max_epochs = max_epochs
            self.verbose = verbose

        def bootstrap(self):
            """通过自助采样引入数据样本扰动"""
            m = self.data_in.shape[0]
            D = np.arange(m)
            D_bs = np.random.choice(D, size=m)
            D_oob = np.setdiff1d(D, D_bs)
            return D_bs, D_oob

        def init_weights(self, m):
            """初始化网络权重"""
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('linear'))

        def run(self):
            """使用预设超参数，在给定的数据集上训练单个基学习器"""
            for _ in range(self.n):
                train_row, test_row = self.bootstrap()
                dataloader_train = DataLoader(dataset=TensorDataset(torch.from_numpy(self.data_in[train_row]),
                                                                    torch.from_numpy(self.data_out[train_row])),
                                              batch_size=1, shuffle=True)
                dataloader_test = DataLoader(dataset=TensorDataset(torch.from_numpy(self.data_in[test_row]),
                                                                   torch.from_numpy(self.data_out[test_row])),
                                             batch_size=8, shuffle=False)
                net = DenoiseNet(input_channels=self.data_in.shape[1], num_channels=10 * self.data_in.shape[1],
                                 out_channels=self.data_out.shape[1], block_n=3)
                #net = FullConnectNet(input_channels=self.data_in.shape[1], num_channels=5 * self.data_in.shape[1],
                #                     out_channels=self.data_out.shape[1], layer_n=3)
                #summary(net, (1, self.data_in.shape[1]), device='cpu')
                net.double()
                net.apply(self.init_weights)
                optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.00005)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.2)
                early_stopping = EarlyStopping(patience=100, skip_epoch=200, verbose=False)
                loss_f = nn.SmoothL1Loss()

                epoch = 1
                while epoch <= self.max_epochs:
                    net.train()
                    total_loss, n = 0., 0
                    for i, (X, Y) in enumerate(dataloader_train):
                        optimizer.zero_grad()
                        Y_hat = net(X)
                        loss = loss_f(Y_hat, Y)
                        loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                            total_loss += float(loss) * X.shape[0]
                            n += X.shape[0]
                    train_loss = total_loss / n
                    net.eval()
                    total_loss, n = 0., 0
                    for i, (X, Y) in enumerate(dataloader_test):
                        Y_hat = net(X)
                        loss = loss_f(Y_hat, Y)
                        with torch.no_grad():
                            total_loss += float(loss) * X.shape[0]
                            n += X.shape[0]
                    test_loss = total_loss / n
                    if self.verbose:
                        print("Epoch %i: train loss %.4f , test loss %.3e" % (epoch, train_loss, test_loss))
                    scheduler.step()
                    early_stopping(test_loss)
                    if early_stopping:
                        if self.verbose:
                            print("Early stopping triggered")
                        break
                    epoch += 1
                self.queue.put((net, epoch))

    def train_net(self, thread_n=1, n_per_thread=1):
        """批量训练多个同质学习器用于集成，可指定多进程"""
        queue = Queue()
        data_in = np.hstack([self.data_in_scale[0](self.data_in), self.data_noise])
        data_out = self.data_out_scale[0](self.data_out)
        worker = []
        for _ in range(thread_n):
            p = NetBagging.Trainer(queue, data_in, data_out, n_per_thread)
            p.start()
            worker.append(p)
        net_list = []
        error_list = []
        for i in range(thread_n * n_per_thread):
            net, epoch = queue.get()
            error, _ = self.score(net)
            print(f"[Net {i}] Train epochs: {epoch}; Mean reconstruction error:",
                  ', '.join(map(lambda x: '{:.3f}%'.format(x), error * 100)))
            net_list.append(net)
            error_list.append(error)
        return net_list, error_list

    def eval_net(self, *nets, data_in=None, input_scale=True):
        """从给定的网络集成采样"""
        if data_in is None:
            # 若不给定输入数据，计算训练数据点处的原始拟合值
            data_in = self.data_in
            data_noise = self.data_noise
        else:
            # 若给定输入数据，计算去噪后的拟合值
            data_in = np.array(data_in)
            data_noise = np.zeros((data_in.shape[0], self.data_noise.shape[1]), dtype=np.float64)
        if input_scale:
            data_in = self.data_in_scale[0](data_in)
        X = np.hstack([data_in, data_noise])
        Y_list = []
        if len(nets) == 0 and self.net is not None:
            nets = self.net
        for net in nets:
            Y_list.append(net(torch.from_numpy(X)).detach().numpy())
        Y = np.mean(np.array(Y_list), axis=0)
        return self.data_out_scale[1](Y)

    def score(self, *nets):
        """计算网络集成的去噪后误差和原始拟合误差"""
        data_denoise = self.eval_net(*nets)
        error_with_noise = np.mean(np.abs(data_denoise - self.data_out) / self.data_out, axis=0)

        valid_row = np.prod(self.data_noise <= self.noise_threshold, axis=1, dtype=bool)
        data_denoise = self.eval_net(*nets, data_in=self.data_in)
        error = np.mean((np.abs(data_denoise - self.data_out) / self.data_out)[valid_row], axis=0)
        return error, error_with_noise

    def bagging(self, search_domain=2.0):
        """使用Bagging算法训练神经网络集成"""
        n = int(search_domain * self.T)
        thread_n = min(n, self.max_thread)
        n_per_thread = int(np.ceil(n / thread_n))
        net_list, error_list = self.train_net(thread_n, n_per_thread)
        if n == 1:
            self.net = net_list
        else:
            net_list = np.array(net_list)
            error_list = np.array(error_list)
            f, _ = normalize(error_list, type='maxmin')
            error = np.linalg.norm(f(error_list) + 1., axis=1)
            ind = np.argsort(error)[: self.T]
            self.net = net_list[ind].tolist()
        final_error = self.score(*self.net)
        print("Final error (with noise): " + ', '.join(map(lambda x: '{:.3f}%'.format(x), final_error[1] * 100)))
        print("Final error: " + ', '.join(map(lambda x: '{:.3f}%'.format(x), final_error[0] * 100)))

    def plot(self, n=100, x_label=None, y_label=None):
        """在坐标截面（白化后）上绘制拟合曲面"""
        n_in = self.data_in.shape[1]
        n_out = self.data_out.shape[1]
        fig, axes = plt.subplots(n_out, n_in, sharex='col', sharey='row', figsize=(10, 8))
        axes = np.reshape(axes, (n_out, -1))
        fig.suptitle(f"Origin cross section of fitting hyperplane (dim: {n_in} -> {n_out})")
        for in_i in range(n_in):
            data_in_std = np.zeros((n, n_in))
            data_in_std[:, in_i] = np.linspace(-2, 2, n)
            data_in = self.data_in_scale[1](data_in_std)
            data_out = self.eval_net(*self.net, data_in=data_in_std, input_scale=False)
            arr = self.data_in_scale[0](self.data_in.copy())
            arr = np.delete(arr, in_i, axis=1)
            distance = np.linalg.norm(arr, axis=1)
            for out_i in range(n_out):
                ax = axes[out_i, in_i]
                ax.plot(data_in[:, in_i], data_out[:, out_i])
                ax.scatter(self.data_in[:, in_i], self.data_out[:, out_i], s=12, c=distance, cmap='viridis')
                if out_i + 1 == n_out:
                    if isinstance(x_label, list):
                        ax.set_xlabel(x_label[in_i], fontsize=20)
                    else:
                        ax.set_xlabel("${in}_{%d}$" % in_i)
                if in_i == 0:
                    if isinstance(y_label, list):
                        ax.set_ylabel(y_label[out_i], fontsize=20)
                    else:
                        ax.set_ylabel("${out}_{%d}$" % out_i)
                ax.autoscale()
                ax.grid()
        fig.show()

    def plot3d(self, in_i=0, in_j=1, out_i=0, n=20):
        """指定两个输入维度和一个输出维度绘制拟合曲面的三维投影"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        n_in = self.data_in.shape[1]
        data_in_std = np.zeros((n**2, n_in))
        divide = np.linspace(-2, 2, n)
        X, Y = np.meshgrid(divide, divide)
        data_in_std[:, in_i] = X.flatten()
        data_in_std[:, in_j] = Y.flatten()
        data_in = self.data_in_scale[1](data_in_std)
        data_out = self.eval_net(*self.net, data_in=data_in_std, input_scale=False)
        surf = ax.plot_surface(data_in[:, in_i].reshape((n, n)),
                               data_in[:, in_j].reshape((n, n)),
                               data_out[:, out_i].reshape((n, n)),
                               cmap='coolwarm', linewidth=0, alpha=0.6)
        fig.colorbar(surf, shrink=0.5, aspect=6)
        '''ax.plot_wireframe(data_in[:, in_i].reshape((n, n)),
                          data_in[:, in_j].reshape((n, n)),
                          data_out[:, out_i].reshape((n, n)),
                          rstride=2, cstride=2, alpha=0.6)'''
        arr = self.data_in_scale[0](self.data_in.copy())
        arr = np.delete(arr, [in_i, in_j], axis=1)
        distance = np.linalg.norm(arr, axis=1)
        ax.scatter(self.data_in[:, in_i], self.data_in[:, in_j], self.data_out[:, out_i],
                   s=16, c=distance, cmap='viridis')
        plt.show()


def performance(csv_file, n=20, Cp=2837.76, K=0.242, M=20.9, inlet_t=3500, r_t=0.2):
    gas_prop = thermo(Cp, M * 1e-3)

    data = pd.read_csv(csv_file)
    net_bagging = NetBagging(data[['inlet_p', 'atmo_p', 'Qm_max', 'Cf_max']],
                             data[['report-def-massflow', 'report-def-thrust', 'Cf', 'SpecImpulse']],
                             data[['report-def-continuity']],
                             model_n=20)
    net_bagging.bagging(search_domain=2)
    net_bagging.plot(x_label=["$p_0$ / $Pa$", "$p_e$ / $Pa$", "$Q_{m,max}$ / $kg·s^{-1}$", "$C_{f,max}$"],
                     y_label=["$Q_m$ / $kg·s^{-1}$", "$F$ / $N$", "$C_f$", "$I_s$"])
    net_bagging.plot3d(in_i=0, in_j=1, out_i=2)
    '''denoise = Denoise(data[['inlet_p', 'atmo_p', 'Qm_max', 'Cf_max']],
                      data[['report-def-massflow', 'report-def-thrust', 'Cf', 'SpecImpulse']],
                      data[['report-def-continuity']])
    denoise.by_net(lr=0.005, num_epochs=400)
    denoise.plot()
    net_bagging = denoise'''

    def net_surface(X, Y):
        x, y = X.flatten(), Y.flatten()
        data_in = np.vstack([x, y,
                             Qm_max(x, inlet_t, np.pi * r_t ** 2, gas_prop['gamma'], gas_prop['R']),
                             Cf_max(x, y, gas_prop['gamma'])]).T
        data_out = net_bagging.eval_net(data_in=data_in)
        #data_out = net_bagging.sample(data_in=data_in)
        return data_out[:, 2].reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_range = net_bagging.data_in[:, 0].min(), net_bagging.data_in[:, 0].max()
    y_range = net_bagging.data_in[:, 1].min(), net_bagging.data_in[:, 1].max()
    X, Y = np.meshgrid(np.linspace(*x_range, n), np.linspace(*y_range, n))
    surf = ax.plot_surface(X, Y, net_surface(X, Y), cmap='coolwarm', linewidth=0, alpha=0.6)
    fig.colorbar(surf, shrink=0.5, aspect=6)
    distance = np.linalg.norm(net_bagging.data_noise, axis=1)
    ax.scatter(net_bagging.data_in[:, 0], net_bagging.data_in[:, 1], net_bagging.data_out[:, 2],
               s=16, c=distance, cmap='viridis')
    ax.set_xlabel("$p_0$ / $Pa$")
    ax.set_ylabel("$p_e$ / $Pa$")
    ax.set_zlabel("$C_f$")
    plt.show()

    def trapezoid(n_int=100):
        # 梯形公式（1阶），计算点：(n+1)^2
        X, Y = np.meshgrid(np.linspace(*x_range, n_int + 1), np.linspace(*y_range, n_int + 1))
        Z = net_surface(X, Y)
        Z[:, [0, -1]] *= 0.5
        Z[[0, -1], :] *= 0.5
        return Z.sum() / n_int ** 2

    def midpoint(n_int=100):
        # 中点公式（1阶），计算点：n^2
        dx = (x_range[1] - x_range[0]) / n_int
        dy = (y_range[1] - y_range[0]) / n_int
        X, Y = np.meshgrid(np.arange(x_range[0] + 0.5 * dx, x_range[1], dx),
                           np.arange(y_range[0] + 0.5 * dy, y_range[1], dy))
        Z = net_surface(X, Y)
        return Z.sum() / n_int**2

    def simpson(n_int=100):
        # 中点公式+辛普森公式（<3阶），计算点：(n+1)^2+n^2
        X, Y = np.meshgrid(np.linspace(*x_range, n_int + 1), np.linspace(*y_range, n_int + 1))
        Z = net_surface(X, Y)
        Z[:, [0, -1]] *= 0.5
        Z[[0, -1], :] *= 0.5
        dx = (x_range[1] - x_range[0]) / n_int
        dy = (y_range[1] - y_range[0]) / n_int
        Z_mid = net_surface(X[: -1, : -1] + 0.5 * dx, Y[: -1, : -1] + 0.5 * dy)
        return (Z.sum() + 2 * Z_mid.sum()) / (3 * n_int ** 2)

    def gauss_2(n_int=100):
        # 两点高斯积分（3阶），计算点：4*n^2
        dx = (x_range[1] - x_range[0]) / n_int
        dy = (y_range[1] - y_range[0]) / n_int
        X, Y = np.meshgrid(np.arange(x_range[0] + 0.5 * dx, x_range[1], dx),
                           np.arange(y_range[0] + 0.5 * dy, y_range[1], dy))
        k_x = 0.5 * dx * np.sqrt(1 / 3)
        k_y = 0.5 * dy * np.sqrt(1 / 3)
        Z_11 = net_surface(X - k_x, Y - k_y)
        Z_12 = net_surface(X - k_x, Y + k_y)
        Z_21 = net_surface(X + k_x, Y - k_y)
        Z_22 = net_surface(X + k_x, Y + k_y)
        return (Z_11.sum() + Z_12.sum() + Z_21.sum() + Z_22.sum()) / (4 * n_int**2)

    def gauss_3(n_int=100):
        # 中点公式+三点高斯积分（<5阶），计算点：5*n^2
        dx = (x_range[1] - x_range[0]) / n_int
        dy = (y_range[1] - y_range[0]) / n_int
        X, Y = np.meshgrid(np.arange(x_range[0] + 0.5 * dx, x_range[1], dx),
                           np.arange(y_range[0] + 0.5 * dy, y_range[1], dy))
        k_x = 0.5 * dx * np.sqrt(3 / 5)
        k_y = 0.5 * dy * np.sqrt(3 / 5)
        Z = net_surface(X, Y)
        Z_11 = net_surface(X - k_x, Y - k_y)
        Z_12 = net_surface(X - k_x, Y + k_y)
        Z_21 = net_surface(X + k_x, Y - k_y)
        Z_22 = net_surface(X + k_x, Y + k_y)
        return (16 * Z.sum() + 5 * (Z_11.sum() + Z_12.sum() + Z_21.sum() + Z_22.sum())) / (36 * n_int**2)

    def test(p_list):
        a = []
        b = []
        c = []
        d = []
        e = []
        for p in p_list:
            a.append(trapezoid(p))
            b.append(midpoint(p))
            c.append(simpson(p))
            d.append(gauss_2(p))
            e.append(gauss_3(p))
        std = gauss_3(1000)
        print(std, a, b, c, d, e, sep='\n')
        f = lambda l: list(map(lambda x: x - std, l))
        plot(np.log10(p_list), f(a), f(b), f(c), f(d), f(e),
             legend=['midpoint', 'trapezoid', 'simpson', 'gauss_2', 'gauss_3'],
             x_label='$lg$ $N$', y_label='$Error_{abs}$')
    #test([25, 50, 100, 200, 400, 800])
    test([20, 30, 50, 80, 120, 170, 230, 300])

    return gauss_3(400)    # 误差小于1e-8


class NozzleCFD:

    def __init__(self, jet_type, r_t, epsilon, mesh_n=None, thread=4,
                 script_path=r'./', work_path=r'./',
                 fluent_path=r'/public/software/ansys_inc211/v211/fluent/bin/fluent'):
        self.r_t = r_t
        self.epsilon = epsilon
        self.params = {
            'inlet_p': [3e6, 6e6, 12e6, 20e6],
            'atmo_p': [p + 325 for p in [101e3, 60e3, 36e3, 20e3, 8e3, 0]],
            'inlet_t': 3500,
            'Cp': 2837.76,    # 1006.43
            'K': 0.242,    # 0.0242
            'M': 20.9,    # 28.966
        }
        self.params.update(thermo(self.params['Cp'], self.params['M'] * 1e-3))

        if jet_type == 'bell':
            if not mesh_n:
                mesh_n = 20
            self.base_path = os.path.join(work_path, f'bell_Rt{r_t:.1e}_eps{epsilon:.1f}_n{mesh_n:d}')
            os.makedirs(self.base_path, exist_ok=True)
            jou_path = os.path.join(self.base_path, 'bell.jou')
            copy_file(jou_path, os.path.join(script_path, 'bell.jou'))
            # 计算喷管构型
            bell = CharacteristicsNozzle(r_t=r_t, rho_t=10 * r_t, axial_sym=True)
            bell.derive(epsilon=epsilon, throat_theta=40)
            bell.plot_field()
            # 模型生成及网格划分
            bell.generate(size=r_t / mesh_n, factor=3)
            bell.plot_profile()
            profile, tag = bell.get_profile()
            profile_to_msh(profile, tag, lc=0., planner=True,
                           save_path=os.path.join(self.base_path, 'bell.bdf'))
            self.A_inlet = np.pi * profile[tag['inlet'][0]][1] ** 2
            # 创建Fluent任务
            self.model = bell
            self.task = FluentQuest(fluent_path, os.path.abspath(jou_path), planar_geom=True, thread_n=thread)
        elif jet_type == 'plug':
            if not mesh_n:
                mesh_n = 150
            self.base_path = os.path.join(work_path, f'plug_Rt{r_t:.2e}_eps{epsilon:.1f}_n{mesh_n:d}')
            os.makedirs(self.base_path, exist_ok=True)
            jou_path = os.path.join(self.base_path, 'plug.jou')
            copy_file(jou_path, os.path.join(script_path, 'plug.jou'))
            # 计算喷管构型
            plug = External(epsilon=epsilon, r_t=r_t)
            plug.derive()
            # 模型生成及网格划分
            plug.generate(n=mesh_n, factor=6)
            plug.plot()
            profile, tag = plug.get_profile()
            profile_to_msh(profile, tag, lc=0., planner=True,
                           save_path=os.path.join(self.base_path, 'plug.bdf'))
            self.A_inlet = np.pi * (profile[tag['inlet'][0]][1] ** 2 - profile[tag['inlet'][-1]][1] ** 2)
            # 创建Fluent任务
            self.model = plug
            self.task = FluentQuest(fluent_path, os.path.abspath(jou_path), planar_geom=True, thread_n=thread)
        else:
            raise ValueError("Unknown type of nozzle. (Supported type: bell, plug)")

        # 需要求解的参数组
        self.task.add_params(
            Cp=[self.params['Cp']], K=[self.params['K']], M=[self.params['M']],
            inlet_p=self.params['inlet_p'], inlet_t=[self.params['inlet_t']],
            atmo_p=self.params['atmo_p'], inlet_area=[self.A_inlet])

        self.data = None

    def postproc(self):
        self.task.get_result('report-def-0-rfile.out')
        result_file = os.path.join(self.base_path, 'fluent_result.txt')
        with open(result_file, 'r', encoding='utf-8') as fr:
            header = fr.readline().strip().split(',')
            data = pd.read_csv(fr, delimiter=' ', index_col=0, names=header)
        gas_prop = thermo(data['Cp'], data['M'] * 1e-3)
        data['Qm_max'] = Qm_max(data['inlet_p'], data['inlet_t'], np.pi * self.r_t ** 2,
                                gas_prop['gamma'], gas_prop['R'])
        data['Cf'] = data['report-def-thrust'] / (np.pi * self.r_t ** 2 * data['inlet_p'])
        data['Cf_max'] = Cf_max(data['inlet_p'], data['atmo_p'], gas_prop['gamma'])
        data['SpecImpulse'] = data['report-def-thrust'] / (9.80665 * data['report-def-massflow'])

        data.to_csv(os.path.join(self.base_path, 'result.csv'))
        self.data = data
        return data

    def calc_cf(self, continuity_limit=10, n_plot=20, n_int=400):
        # 拟合cfd计算结果
        data = self.data[self.data['report-def-continuity'] < continuity_limit]    # 未收敛（发散）的结果直接丢弃
        net_bagging = NetBagging(data[['inlet_p', 'atmo_p', 'Qm_max', 'Cf_max']],
                                 data[['report-def-massflow', 'report-def-thrust', 'Cf', 'SpecImpulse']],
                                 data[['report-def-continuity']],
                                 model_n=20)
        net_bagging.bagging(search_domain=2)
        net_bagging.plot(x_label=["$p_0$ / $Pa$", "$p_e$ / $Pa$", "$Q_{m,max}$ / $kg·s^{-1}$", "$C_{f,max}$"],
                         y_label=["$Q_m$ / $kg·s^{-1}$", "$F$ / $N$", "$C_f$", "$I_s$"])

        # 假定部分参数不变，并简化去噪网络的输入和输出
        def net_surface(X, Y):
            x, y = X.flatten(), Y.flatten()
            data_in = np.vstack([x, y,
                                 Qm_max(x, self.params['inlet_t'], np.pi * self.r_t ** 2,
                                        self.params['gamma'], self.params['R']),
                                 Cf_max(x, y, self.params['gamma'])]).T
            data_out = net_bagging.eval_net(data_in=data_in)
            return data_out[:, 2].reshape(X.shape)
        x_range = net_bagging.data_in[:, 0].min(), net_bagging.data_in[:, 0].max()
        y_range = net_bagging.data_in[:, 1].min(), net_bagging.data_in[:, 1].max()

        # 绘制推力系数Cf关于燃烧室压强p0和环境压强pe的拟合曲面
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(np.linspace(*x_range, n_plot), np.linspace(*y_range, n_plot))
        surf = ax.plot_surface(X, Y, net_surface(X, Y), cmap='coolwarm', linewidth=0, alpha=0.6)
        fig.colorbar(surf, shrink=0.5, aspect=6)
        distance = np.linalg.norm(net_bagging.data_noise, axis=1)
        ax.scatter(net_bagging.data_in[:, 0], net_bagging.data_in[:, 1], net_bagging.data_out[:, 2],
                   s=16, c=distance, cmap='viridis')
        ax.set_xlabel("$p_0$ / $Pa$")
        ax.set_ylabel("$p_e$ / $Pa$")
        ax.set_zlabel("$C_f$")
        plt.show()

        # 使用二维三点高斯积分计算当前参数域下Cf的平均值
        dx = (x_range[1] - x_range[0]) / n_int
        dy = (y_range[1] - y_range[0]) / n_int
        X, Y = np.meshgrid(np.arange(x_range[0] + 0.5 * dx, x_range[1], dx),
                           np.arange(y_range[0] + 0.5 * dy, y_range[1], dy))
        k_x = 0.5 * dx * np.sqrt(3 / 5)
        k_y = 0.5 * dy * np.sqrt(3 / 5)
        Z = net_surface(X, Y)
        Z_11 = net_surface(X - k_x, Y - k_y)
        Z_12 = net_surface(X - k_x, Y + k_y)
        Z_21 = net_surface(X + k_x, Y - k_y)
        Z_22 = net_surface(X + k_x, Y + k_y)
        return (16 * Z.sum() + 5 * (Z_11.sum() + Z_12.sum() + Z_21.sum() + Z_22.sum())) / (36 * n_int ** 2)


class PiDenoise:
    """
    针对特定问题，使用物理嵌入的神经网络对噪声数据集进行重构以去除误差
    """

    def __init__(self, data_in: np.ndarray, data_out: np.ndarray, data_noise: np.ndarray,
                 noise_threshold: float = 0.1, noise_deprecated: float = 10.,
                 n_extra_in: int = 0, n_extra_out: int = 0, weight: float = 0.5):
        self.noise_deprecated = noise_deprecated
        self.noise_threshold = noise_threshold
        self.n_extra_in = n_extra_in
        self.n_extra_out = n_extra_out
        self.weight = weight

        # 输入数据对齐
        data_in = np.array(data_in)
        data_out = np.array(data_out)
        data_noise = np.array(data_noise)
        self.m = min(data_in.shape[0], data_noise.shape[0], data_out.shape[0])
        self.n_in = data_in.shape[1]
        self.data_in = data_in[: self.m].astype(np.float64)
        self.n_noise = data_noise.shape[1]
        self.data_noise = data_noise[: self.m].astype(np.float64)
        self.n_out = data_out.shape[1]
        self.data_out = data_out[: self.m].astype(np.float64)

        # 剔除大误差输入
        if noise_deprecated > noise_threshold:
            retain_row = np.prod(self.data_noise <= noise_deprecated, axis=1, dtype=bool)
            self.data_in = self.data_in[retain_row]
            self.data_noise = self.data_noise[retain_row]
            self.data_out = self.data_out[retain_row]
            if len(retain_row) < self.m:
                print("%d samples have been deprecated")
                self.m = len(retain_row)

        self.index_in = []
        self.index_out = []
        self.index_noise = []
        self.valid_row = np.ones(self.m)
        self.data_in_scale = None
        self.data_extra_in_scale = None
        self.data_out_scale = None
        self.model = None

        self.data_denoise = None
        self.error = float('inf')

    def extra_input(self, data_in):
        """根据输入数据或常量计算额外输入，用于指导网络训练"""
        return torch.zeros((self.m, 0))

    def extra_output(self, data_in, data_out, extra_in, extra_out):
        """根据输入输出数据或常量计算额外输出（守恒方程残差），用于指导网络训练
        注意：输入参数中的extra_in和extra_out是网络在数据集之外的输入和输出"""
        return torch.zeros((self.m, 0))

    def train(self, lr: float = 5e-3, num_epochs: int = 200,
              in_i: Union[None, List[int]] = None,
              out_i: Union[None, List[int]] = None,
              noise_i: Union[None, List[int]] = None,
              print_detail: bool = True):

        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('linear'))

        self.index_in = list(range(self.n_in) if in_i is None else in_i)
        self.index_out = list(range(self.n_out) if out_i is None else out_i)
        self.index_noise = list(range(self.n_noise) if noise_i is None else noise_i)

        data_in = torch.from_numpy(self.data_in[:, self.index_in])
        data_extra_in = self.extra_input(data_in)
        data_out = torch.from_numpy(self.data_out[:, self.index_out])
        data_noise = torch.from_numpy(self.data_noise[:, self.index_noise])
        self.data_in_scale = Normalize(data_in, type='maxmin')
        self.data_extra_in_scale = Normalize(data_extra_in, type='maxmin')
        self.data_out_scale = Normalize(data_out, type='zscore')
        valid_row = np.prod(data_noise.numpy() <= self.noise_threshold, axis=1, dtype=bool)

        dataloader = DataLoader(dataset=TensorDataset(
            torch.arange(self.m),
            torch.hstack([self.data_in_scale(data_in),
                          self.data_extra_in_scale(data_extra_in),
                          data_noise]),
            self.data_out_scale(data_out)),
            batch_size=4, shuffle=True)
        # 设定网络架构
        input_n = data_in.shape[1] + self.n_extra_in + data_noise.shape[1]
        n_data_out = data_out.shape[1]
        output_n = n_data_out + self.n_extra_out
        net = DenoiseNet(input_channels=input_n, num_channels=20 * input_n, out_channels=output_n, block_n=3)
        # net = FullConnectNet(input_channels=input_n, num_channels=5 * input_n, out_channels=output_n, layer_n=3)
        if print_detail:
            summary(net, (1, input_n), device='cpu')
        net.double()
        net.apply(init_weights)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.00005)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400], gamma=0.2)
        loss_d_f = nn.SmoothL1Loss()    # 使用平滑L1损失函数以减小离群样本点的影响
        loss_p_f = nn.SmoothL1Loss()    # 物理损失项

        net.train()
        for epoch in range(num_epochs):
            total_loss_d, total_loss_p, n = 0., 0., 0
            for i, (ind, X, Y) in enumerate(dataloader):
                optimizer.zero_grad()
                Y_hat = net(X)
                loss_d = loss_d_f(Y_hat, Y)
                extra_out = self.extra_output(data_in[ind],
                                              self.data_out_scale.inv(Y_hat[:, : n_data_out]),
                                              data_extra_in[ind],
                                              Y_hat[:, n_data_out:])
                loss_p = loss_p_f(extra_out, torch.zeros(extra_out.shape, dtype=torch.float64))
                loss = (1. - self.weight) * loss_d + self.weight * loss_p
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    total_loss_d += float(loss_d) * X.shape[0]
                    total_loss_p += float(loss_p) * X.shape[0]
                    n += X.shape[0]
            scheduler.step()
            if print_detail:
                print("Epoch %i: loss_d %.4f, loss_p %.4f" % (epoch + 1, total_loss_d / n, total_loss_p / n))
        net.eval()
        total_loss_d, total_loss_p, n = 0., 0., 0
        for i, (ind, X, Y) in enumerate(dataloader):
            Y_hat = net(X)
            loss_d = loss_d_f(Y_hat, Y)
            extra_out = self.extra_output(data_in[ind],
                                          self.data_out_scale.inv(Y_hat[:, : n_data_out]),
                                          data_extra_in[ind],
                                          Y_hat[:, n_data_out:])
            loss_p = loss_p_f(extra_out, torch.zeros(extra_out.shape))
            with torch.no_grad():
                total_loss_d += float(loss_d) * X.shape[0]
                total_loss_p += float(loss_p) * X.shape[0]
                n += X.shape[0]
        total_loss = ((1. - self.weight) * total_loss_d + self.weight * total_loss_p) / n
        print("Final average loss = %.3e (ratio %.3f)" % (total_loss, total_loss_p / total_loss_d))
        self.model = net
        self.valid_row = valid_row

        X = torch.hstack([self.data_in_scale(data_in), self.data_extra_in_scale(data_extra_in), data_noise])
        Y = self.data_out_scale.inv(net(X)[:, : n_data_out])
        error = np.mean(torch.abs((Y - data_out) / data_out).detach().numpy(), axis=0)
        print("Mean reconstruction error (with noise): " + ', '.join(map(lambda x: '{:.3f}%'.format(x), error * 100)))
        data_denoise = self.sample(data_in)
        error = np.mean(torch.abs((data_denoise - data_out) / data_out).detach().numpy()[valid_row], axis=0)
        print("Mean reconstruction error: " + ', '.join(map(lambda x: '{:.3f}%'.format(x), error * 100)))
        self.data_denoise = torch.hstack([data_in,
                                          torch.zeros(data_noise.shape, dtype=torch.float64),
                                          data_denoise]).detach().numpy()
        self.error = error

    def sample(self, data_in: torch.Tensor, input_scale: bool = True) -> torch.Tensor:
        if input_scale:
            data_extra_in = self.extra_input(data_in)
            data_in = self.data_in_scale(data_in)
        else:
            data_extra_in = self.extra_input(self.data_in_scale.inv(data_in))
        if isinstance(self.model, DenoiseNet) or isinstance(self.model, FullConnectNet):
            X = torch.hstack([data_in,
                              self.data_extra_in_scale(data_extra_in),
                              torch.zeros((data_in.shape[0], len(self.index_noise)), dtype=torch.float64)])
            Y = self.data_out_scale.inv(self.model(X)[:, : len(self.index_out)])
            return Y.detach()
        else:
            raise ValueError("Model is not initialized or has unknown type")

    def plot(self, n: int = 100,
             x_label: Union[None, List[str]] = None,
             y_label: Union[None, List[str]] = None):
        n_in = len(self.index_in)
        n_out = len(self.index_out)
        fig, axes = plt.subplots(n_out, n_in, sharex='col', sharey='row', figsize=(10, 8))
        axes = np.reshape(axes, (n_out, -1))
        fig.suptitle(f"Origin cross section of fitting hyperplane (dim: {n_in} -> {n_out})")
        for x_i, in_i in enumerate(self.index_in):
            data_in_std = torch.zeros((n, n_in))
            data_in_std[:, x_i] = torch.linspace(-2, 2, n)
            data_in = self.data_in_scale.inv(data_in_std)
            data_out = self.sample(data_in_std, input_scale=False)
            arr = self.data_in[:, self.index_in]
            arr = self.data_in_scale(arr)
            arr = np.delete(arr, x_i, axis=1)
            distance = np.linalg.norm(arr, axis=1)
            for y_i, out_i in enumerate(self.index_out):
                ax = axes[y_i, x_i]
                ax.plot(data_in[:, x_i], data_out[:, y_i])
                ax.scatter(self.data_in[:, in_i], self.data_out[:, out_i], s=12, c=distance, cmap='viridis')
                if y_i + 1 == n_out:
                    if isinstance(x_label, list):
                        ax.set_xlabel(x_label[x_i], fontsize=20)
                    else:
                        ax.set_xlabel("${in}_{%d}$" % in_i)
                if x_i == 0:
                    if isinstance(y_label, list):
                        ax.set_ylabel(y_label[y_i], fontsize=20)
                    else:
                        ax.set_ylabel("${out}_{%d}$" % out_i)
                ax.autoscale()
                ax.grid()
        fig.show()


class MyPiDenoise(PiDenoise):

    def __init__(self, data_in, data_out, data_noise, weight=0.5):
        super(MyPiDenoise, self).__init__(data_in, data_out, data_noise, noise_threshold=0.1, noise_deprecated=10,
                                          n_extra_in=2, n_extra_out=0, weight=weight)
        # 定义常量
        self.gas_prop = {
            'Cp': 2837.76,  # 1006.43
            'K': 0.242,  # 0.0242
            'M': 20.9,  # 28.966
        }
        self.gas_prop.update(thermo(self.gas_prop['Cp'], self.gas_prop['M'] * 1e-3))
        self.inlet_t = 3500
        self.area_t = np.pi * 0.2 ** 2

    def extra_input(self, data_in):
        inlet_p = data_in[:, 0]
        atmo_p = data_in[:, 1]
        return torch.vstack([Qm_max(inlet_p, self.inlet_t, self.area_t, self.gas_prop['gamma'], self.gas_prop['R']),
                             Cf_max(inlet_p, atmo_p, self.gas_prop['gamma'])]).T

    def extra_output(self, data_in, data_out, extra_in, extra_out):
        inlet_p = data_in[:, 0]
        mass_flow = data_out[:, 0]
        thrust = data_out[:, 1]
        c_f = data_out[:, 2]
        spec_imp = data_out[:, 3]
        return torch.vstack([(c_f - thrust / (self.area_t * inlet_p)) / self.data_out_scale.mean[2],
                             (spec_imp - thrust / (9.80665 * mass_flow)) / self.data_out_scale.mean[3]]).T


def test_pi_denoise():
    data = pd.read_csv(r'.\plug_result.csv')
    print(data.columns)

    denoise = MyPiDenoise(data[['inlet_p', 'atmo_p', 'Qm_max', 'Cf_max']],
                          data[['report-def-massflow', 'report-def-thrust', 'Cf', 'SpecImpulse']],
                          data[['report-def-continuity']], weight=0.2)
    denoise.train(lr=0.005, num_epochs=500, in_i=[0, 1])
    denoise.plot(x_label=["$p_0$ / $Pa$", "$p_e$ / $Pa$"],
                 y_label=["$Q_m$ / $kg·s^{-1}$", "$F$ / $N$", "$C_f$", "$I_s$"])


def test_bagging():
    data = pd.read_csv(r'.\plug_result.csv')
    print(data.columns)

    # multiprocessing、pickle不支持和pycharm控制台一起使用
    net_bagging = NetBagging(data[['inlet_p', 'atmo_p', 'Qm_max', 'Cf_max']],
                             data[['report-def-massflow', 'report-def-thrust', 'Cf', 'SpecImpulse']],
                             data[['report-def-continuity']],
                             model_n=1)
    net_bagging.bagging(search_domain=1)
    net_bagging.plot(x_label=["$p_0$ / $Pa$", "$p_e$ / $Pa$", "$Q_{m,max}$ / $kg·s^{-1}$", "$C_{f,max}$"],
                     y_label=["$Q_m$ / $kg·s^{-1}$", "$F$ / $N$", "$C_f$", "$I_s$"])
    net_bagging.plot3d(out_i=2)
    '''net_bagging = NetBagging(data[['inlet_p', 'atmo_p']],
                             data[['Cf']],
                             data[['report-def-continuity']],
                             model_n=4)
    net_bagging.bagging(search_domain=2)
    #net_bagging.plot(x_label=["$p_0$ / $Pa$", "$p_e$ / $Pa$"], y_label=["$C_f$"])'''
    #net_bagging.plot3d(in_i=0, in_j=1, out_i=2)

    #performance(r'.\plug_result.csv')


if __name__ == '__main__':
    # test_denoise()
    # test_pi_denoise()
    pass

    data = pd.read_csv(r'.\plug_result.csv')
    print(data.columns)
    '''denoise = Denoise(data[['inlet_p', 'atmo_p', 'Qm_max', 'Cf_max']],
                      data[['report-def-massflow', 'report-def-thrust', 'Cf', 'SpecImpulse']],
                      data[['report-def-continuity']])
    #denoise.by_lwlr(k=0.2, in_i=[0, 1], out_i=2)
    #denoise.by_net(lr=0.005, num_epochs=400, in_i=[0, 1], out_i=[2])
    denoise.by_net(lr=0.005, num_epochs=400)
    #denoise.plot(x_label=["$p_0$ / $Pa$", "$p_e$ / $Pa$"], y_label=["$C_f$"])
    denoise.plot()'''

    # multiprocessing不支持和pycharm控制台一起使用
    net_bagging = NetBagging(data[['inlet_p', 'atmo_p', 'Qm_max', 'Cf_max']],
                             data[['report-def-massflow', 'report-def-thrust', 'Cf', 'SpecImpulse']],
                             data[['report-def-continuity']],
                             model_n=1)
    net_bagging.bagging(search_domain=1)
    net_bagging.plot(x_label=["$p_0$ / $Pa$", "$p_e$ / $Pa$", "$Q_{m,max}$ / $kg·s^{-1}$", "$C_{f,max}$"],
                     y_label=["$Q_m$ / $kg·s^{-1}$", "$F$ / $N$", "$C_f$", "$I_s$"])
    net_bagging.plot3d(out_i=2)
    '''net_bagging = NetBagging(data[['inlet_p', 'atmo_p']],
                             data[['Cf']],
                             data[['report-def-continuity']],
                             model_n=4)
    net_bagging.bagging(search_domain=2)
    #net_bagging.plot(x_label=["$p_0$ / $Pa$", "$p_e$ / $Pa$"], y_label=["$C_f$"])'''
    #net_bagging.plot3d(in_i=0, in_j=1, out_i=2)

    #performance(r'.\plug_result.csv')
    pass


