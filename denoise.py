import os
import random
import sys
import time
from io import BytesIO
from typing import Union, List, Iterable, Any, Type, Optional
from threading import Thread

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.multiprocessing import Process, Queue
from torchinfo import summary
from thop import profile

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

    def __init__(self, input_channels, num_channels, output_channels, block_n=3):
        super(DenoiseNet, self).__init__()
        self.block_n = block_n
        self.residual = nn.Sequential(*[ResidualBlock(num_channels, 4 * num_channels, drop_p=0.0)
                                        for _ in range(block_n)])
        self.linear_in = nn.Linear(input_channels, num_channels)
        self.linear_out = nn.Linear(num_channels, output_channels)

    def forward(self, X):
        X = F.tanh(self.linear_in(X))
        Y = self.residual(X)
        Y = self.linear_out(F.tanh(Y))
        return Y


class FullConnectNet(nn.Module):

    def __init__(self, input_channels, num_channels, output_channels, layer_n=3):
        super(FullConnectNet, self).__init__()
        self.layer_n = layer_n
        self.linear_up = nn.Linear(input_channels, num_channels)
        hidden = []
        for _ in range(layer_n):
            hidden.extend([nn.Linear(num_channels, num_channels), nn.ReLU()])
        self.hidden = nn.Sequential(*hidden)
        self.linear_down = nn.Linear(num_channels, output_channels)

    def forward(self, X):
        X = self.linear_up(X)
        X = self.hidden(X)
        return self.linear_down(F.tanh(X))


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

    def __init__(self, num_channels: int, data: torch.Tensor = None, reduce_type: str = 'zscore'):
        if data is not None:
            data_channels = data.shape[1]
            if reduce_type == 'maxmin':
                _min = torch.min(data, dim=0)[0]
                _max = torch.max(data, dim=0)[0]
                _mean = 0.5 * (_max + _min)
                _std = 0.5 * (_max - _min)
            elif reduce_type == 'zscore':
                _mean = torch.mean(data, dim=0)
                _std = torch.std(data, dim=0)
            else:
                print("Unknown type of normalization: %s (Supported type: 'maxmin', 'zscore')" % reduce_type)
                _mean = torch.zeros(data_channels, dtype=data.dtype)
                _std = torch.ones(data_channels, dtype=data.dtype)
            if num_channels <= data_channels:
                self.mean = _mean[: num_channels]
                self.std = _std[: num_channels]
            else:
                self.mean = torch.zeros(num_channels, dtype=data.dtype)
                self.mean[: data_channels] = _mean
                self.std = torch.ones(num_channels, dtype=data.dtype)
                self.std[: data_channels] = _std
            # 将计算为零的标准差设为1
            self.std[self.std == 0.] = 1.
        else:
            self.mean = torch.zeros(num_channels)
            self.std = torch.ones(num_channels)
        self.n = num_channels

    def _mean_like(self, x: torch.Tensor):
        n = x.shape[1]
        if self.n < n:
            mean = torch.zeros(n)
            mean[: self.n] = self.mean
        else:
            mean = self.mean[: n]
        return mean

    def _std_like(self, x: torch.Tensor):
        n = x.shape[1]
        if self.n < n:
            std = torch.ones(n)
            std[: self.n] = self.std
        else:
            std = self.std[: n]
        return std

    def __call__(self, x: torch.Tensor, strict=True):
        if strict:
            return (x - self.mean) / self.std
        else:
            return (x - self._mean_like(x)) / self._std_like(x)

    def inv(self, x: torch.Tensor, strict=True):
        if strict:
            return self.std * x + self.mean
        else:
            return self._std_like(x) * x + self._mean_like(x)


class BaseNetWrapper(nn.Module):
    """
    用于回归问题的神经网络，通过继承该类可以实现额外物理信息的嵌入

    需要提供网络的输入输出通道数，以及用于计算网络归一化参数的数据集
    """

    hidden_channels = 10  # 隐藏层神经元数量（宽度）
    layer_n = 3  # 神经网络层数（深度）
    weight = 0.0  # 物理损失权重

    def __init__(self, in_channels: int, out_channels: int,
                 data_in: torch.Tensor = None, data_out: torch.Tensor = None):
        super(BaseNetWrapper, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 计算包含额外输入的输入通道数，同时测试外部定义函数
        dummy_x = torch.rand(1, self.in_channels)
        dummy_y = self._extra_input(dummy_x)
        assert np.ndim(dummy_y) == 2
        in_channels = self.in_channels + dummy_y.shape[1]
        # 创建网络
        self.net = DenoiseNet(input_channels=in_channels,
                              num_channels=self.hidden_channels,
                              output_channels=self.out_channels,
                              block_n=self.layer_n)
        # 调整数据的通道数以符合网络要求
        if data_in is not None:
            if data_in.shape[1] < self.in_channels:
                data_in = torch.hstack([data_in,
                                        torch.zeros(data_in.shape[0],
                                                    self.in_channels - data_in.shape[1])])
            else:
                data_in = data_in[:, : self.in_channels]
            data_in = torch.hstack([data_in, self._extra_input(data_in)])
        # 根据给定数据计算网络的归一化层
        self.scale_in = Normalize(in_channels, data_in, reduce_type='maxmin')
        self.scale_out = Normalize(self.out_channels, data_out, reduce_type='zscore')
        # 设定基本损失函数
        self.loss_f = nn.SmoothL1Loss()
        self.net.apply(self.init_weights)
        # summary(self.net, (1000, in_channels), device='cpu')
        self.net.double()
        # 测试外部定义函数
        dummy_x = torch.rand(1, in_channels)
        dummy_y = torch.rand(1, self.out_channels)
        assert np.ndim(self._extra_loss(dummy_x, dummy_y)) == 0

    def check_data(self, data_in: torch.Tensor, data_out: torch.Tensor):
        """检测给定数据集是否符合网络要求"""
        if data_in.shape[1] != self.in_channels:
            raise ValueError("The size of data_in (%d) at dimension 1 must match the input channel (%d) of net" %
                             (data_in.shape[1], self.in_channels))
        if data_out.shape[1] != self.out_channels:
            raise ValueError("The size of data_out (%d) at dimension 1 must match the output channel (%d) of net" %
                             (data_out.shape[1], self.out_channels))

    def init_weights(self, m):
        """初始化网络权重"""
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('linear'))
            nn.init.zeros_(m.bias)

    def _extra_input(self, x: torch.Tensor) -> torch.Tensor:
        """可根据物理方程构建额外输入。注意函数的输入可能包含随机数"""
        return torch.zeros((x.shape[0], 0), dtype=x.dtype)

    def _extra_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """可根据物理方程构建额外损失项。注意函数的输入可能包含随机数"""
        return torch.tensor(.0, dtype=x.dtype)

    def forward(self, x: torch.Tensor, x_scale=True, y_scale=True):
        """执行网络，要求：
        x.shape[1]==self.in_channels"""
        if not x_scale:
            x = self.scale_in.inv(x, strict=False)
        x = torch.hstack([x, self._extra_input(x)])
        _x = self.scale_in(x)
        _y = self.net(_x)
        if y_scale:
            return self.scale_out.inv(_y)
        else:
            return _y

    def calc_loss(self, x: torch.Tensor, y: torch.Tensor = None):
        """计算给定数据集上的损失，要求：
        1. x.shape[1]==self.in_channels
        2. y.shape[1]<=self.out_channels
        3. y.shape[0]<=x.shape[0]"""
        x = torch.hstack([x, self._extra_input(x)])
        _x = self.scale_in(x)
        _y_hat = self.net(_x)
        y_hat = self.scale_out.inv(_y_hat)
        if y is None or len(y) == 0:
            # 若不提供目标函数值y，则仅计算附加损失项
            loss = self.weight * self._extra_loss(x, y_hat)
        else:
            _y = self.scale_out(y, strict=False)
            loss = (1. - self.weight) * self.loss_f(_y_hat[: _y.shape[0], : _y.shape[1]], _y) \
                   + self.weight * self._extra_loss(x, y_hat)
        return loss


class DenoiseWrapper(BaseNetWrapper):
    """
    用于回归问题的神经网络（含噪声数据），通过继承该类可以实现额外物理信息的嵌入
    """

    def __init__(self, data_in: torch.Tensor, data_out: torch.Tensor, noise_channels: int = 1):
        # 默认将网络的最后一个输入保留为噪声通道
        in_channels = data_in.shape[1]
        out_channels = data_out.shape[1]
        scale_channels = in_channels - max(min(noise_channels, in_channels), 0)
        _data_in = data_in[:, : scale_channels]  # 提取非噪声通道，即指定网络只对噪声通道以外的通道进行归一化
        super(DenoiseWrapper, self).__init__(in_channels, out_channels, _data_in, data_out)


class DenoisePhysInWrapper(DenoiseWrapper):
    """
    针对特定问题、包含物理信息的降噪网络

    Net Input (2 + 1 noise): p0, pe, Rm
    Phys. Input (2): Qm_max, Cf_max
    Net Output (4): Qm, F, Cf, Is
    """

    in_channels = 3
    noise_channels = 1
    out_channels = 4

    def __init__(self, data_in: torch.Tensor, data_out: torch.Tensor):
        # 定义常量
        self.gas_prop = {
            'Cp': 2837.76,  # 1006.43
            'K': 0.242,  # 0.0242
            'M': 20.9,  # 28.966
        }
        self.gas_prop.update(thermo(self.gas_prop['Cp'], self.gas_prop['M'] * 1e-3))
        self.inlet_t = 3500
        self.area_t = np.pi * 0.2 ** 2
        # 检查给定数据的通道数是否正确
        self.check_data(data_in, data_out)

        super(DenoisePhysInWrapper, self).__init__(data_in, data_out, noise_channels=1)

    def _extra_input(self, data_in: torch.Tensor):
        inlet_p = data_in[:, 0]
        atmo_p = data_in[:, 1]
        return torch.vstack([Qm_max(inlet_p, self.inlet_t, self.area_t, self.gas_prop['gamma'], self.gas_prop['R']),
                             Cf_max(inlet_p, atmo_p, self.gas_prop['gamma'])]).T


class DenoisePhysInStackWrapper(DenoisePhysInWrapper):
    """
    单通道物理内嵌网络（堆叠型）

    详见 DenoisePhysInWrapper
    """

    out_channels = 1

    def __init__(self, data_in: torch.Tensor, data_out: torch.Tensor):
        super(DenoisePhysInStackWrapper, self).__init__(data_in, data_out)


class DenoisePhysOutWrapper(DenoiseWrapper):
    """
    针对特定问题、包含物理信息的降噪网络

    Net Input (2 + 1 noise): p0, pe, Rm
    Net Output (4): Qm, F, Cf, Is
    Phys. Loss (2): Cf=F/(At*p0), Is=F/(g*Qm)
    """

    in_channels = 3
    noise_channels = 1
    out_channels = 4
    weight = 0.5

    def __init__(self, data_in: torch.Tensor, data_out: torch.Tensor):
        # 定义常量
        self.area_t = np.pi * 0.2 ** 2
        # 检查给定数据的通道数是否正确
        self.check_data(data_in, data_out)

        super(DenoisePhysOutWrapper, self).__init__(data_in, data_out, noise_channels=1)

    def _extra_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inlet_p = x[:, 0]
        mass_flow = y[:, 0]
        thrust = y[:, 1]
        c_f = y[:, 2]
        spec_imp = y[:, 3]
        loss_output = torch.vstack([(c_f - thrust / (self.area_t * inlet_p)) / self.scale_out.mean[2],
                                    (spec_imp - thrust / (9.80665 * mass_flow)) / self.scale_out.mean[3]]).T
        return self.loss_f(loss_output, torch.zeros_like(loss_output))


class DenoisePhysInOutWrapper(DenoiseWrapper):
    """
    针对特定问题、包含物理信息的降噪网络

    Net Input (2 + 1 noise): p0, pe, Rm
    Phys. Input (2): Qm_max, Cf_max
    Net Output (4): Qm, F, Cf, Is
    Phys. Loss (2): Cf=F/(At*p0), Is=F/(g*Qm)
    """

    in_channels = 3
    noise_channels = 1
    out_channels = 4
    weight = 0.5

    def __init__(self, data_in: torch.Tensor, data_out: torch.Tensor):
        # 定义常量
        self.gas_prop = {
            'Cp': 2837.76,  # 1006.43
            'K': 0.242,  # 0.0242
            'M': 20.9,  # 28.966
        }
        self.gas_prop.update(thermo(self.gas_prop['Cp'], self.gas_prop['M'] * 1e-3))
        self.inlet_t = 3500
        self.area_t = np.pi * 0.2 ** 2
        # 检查给定数据的通道数是否正确
        self.check_data(data_in, data_out)

        super(DenoisePhysInOutWrapper, self).__init__(data_in, data_out, noise_channels=1)

    def _extra_input(self, data_in: torch.Tensor):
        inlet_p = data_in[:, 0]
        atmo_p = data_in[:, 1]
        return torch.vstack([Qm_max(inlet_p, self.inlet_t, self.area_t, self.gas_prop['gamma'], self.gas_prop['R']),
                             Cf_max(inlet_p, atmo_p, self.gas_prop['gamma'])]).T

    def _extra_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inlet_p = x[:, 0]
        mass_flow = y[:, 0]
        thrust = y[:, 1]
        c_f = y[:, 2]
        spec_imp = y[:, 3]
        loss_output = torch.vstack([(c_f - thrust / (self.area_t * inlet_p)) / self.scale_out.mean[2],
                                    (spec_imp - thrust / (9.80665 * mass_flow)) / self.scale_out.mean[3]]).T
        return self.loss_f(loss_output, torch.zeros_like(loss_output))


class TrainDataLoader:
    """
    支持额外输入张量的TensorDataset/DataLoader
    """

    def __init__(self, data_in: torch.Tensor, data_out: torch.Tensor, data_in_extra: torch.Tensor = None,
                 batch_size: int = 1, shuffle: bool = False):
        assert data_in.shape[0] == data_out.shape[0], "Size mismatch between data_in and data_out"
        if data_in_extra is None:
            self.data_in = data_in
        else:
            self.data_in = torch.vstack([data_in, data_in_extra])
        self.data_out = data_out
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._i = 0
        self._m = data_in.shape[0]
        self._index = np.arange(self.data_in.shape[0])

    def __iter__(self):
        self._i = 0
        if self.shuffle:
            np.random.shuffle(self._index)
        return self

    def __next__(self):
        if self._i >= self.data_in.shape[0]:
            raise StopIteration
        index_batch = self._index[self._i: min(self._i + self.batch_size, self.data_in.shape[0])]
        not_extra = index_batch < self._m
        ind = index_batch[not_extra]
        ind_extra = index_batch[np.logical_not(not_extra)]
        data_in = torch.vstack([self.data_in[ind], self.data_in[ind_extra]])
        data_out = self.data_out[ind]
        self._i += self.batch_size
        return data_in, data_out

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, index):
        if index < self._m:
            return self.data_in[index], self.data_out[index]
        else:
            return self.data_in[index], None


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
        self.iter_checkpoint = 0

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
                self.iter_checkpoint = self.iter + 1
        self.iter += 1

    def resume(self, net):
        if self.net_cache is not None:
            net.load_state_dict(torch.load(self.net_cache))
        return self.iter_checkpoint


class Trainer(Process):
    """
    用于训练神经网络的子进程，不包含输入输出的归一化
    """

    def __init__(self, queue: Queue,
                 data_in: torch.Tensor, data_out: torch.Tensor,
                 net_type: Optional[Type] = DenoiseWrapper, net_args: dict = None,
                 test_row: Iterable[int] = None, data_in_extra: torch.Tensor = None,
                 net_n: int = 1, lr: float = 5e-3, max_epochs: int = 500, verbose: bool = False):
        super(Trainer, self).__init__(name='NetTrainer', daemon=True)
        self.queue = queue
        self.data_in = data_in
        self.data_out = data_out
        self.data_in_extra = data_in_extra  # 不带监督信息的训练样本
        self.test_row = test_row  # 手动指定数据集划分
        self.net_type = net_type
        self.net_args = {} if net_args is None else net_args
        self.net_n = net_n  # 需要训练的网络数量
        self.lr = lr
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.loss_history = []

    def redirect(self):
        """重定向子进程的标准输出"""
        if self.is_alive():
            sys.stderr.write = lambda s: self.queue.put(('error', s))
            sys.stdout.write = lambda s: self.queue.put(('info', s))

    def bootstrap(self, oob: Iterable[int] = None):
        """手动指定训练集/测试集，或通过自助采样划分（会引入数据样本扰动）"""
        m = self.data_in.shape[0]
        D = np.arange(m)
        if oob is None:
            D_bs = np.random.choice(D, size=m)
            D_oob = np.setdiff1d(D, D_bs)
        else:
            D_oob = np.array(oob)
            D_bs = np.setdiff1d(D, D_oob)
        return D_bs, D_oob

    def run(self):
        """使用预设超参数，在给定的数据集上训练单个基学习器"""
        self.redirect()
        if not issubclass(self.net_type, DenoiseWrapper):
            raise ValueError("Net for training should be same as or subclass of DenoiseWrapper (given %s)" \
                             % self.net_type)
        for _ in range(self.net_n):
            train_row, test_row = self.bootstrap(oob=self.test_row)
            if self.data_in_extra is None:
                dataloader_train = DataLoader(dataset=TensorDataset(self.data_in[train_row],
                                                                    self.data_out[train_row]),
                                              batch_size=4, shuffle=True)
            else:
                dataloader_train = TrainDataLoader(self.data_in[train_row],
                                                   self.data_out[train_row],
                                                   self.data_in_extra,
                                                   batch_size=4, shuffle=True)
            dataloader_test = DataLoader(dataset=TensorDataset(self.data_in[test_row],
                                                               self.data_out[test_row]),
                                         batch_size=8, shuffle=False)
            net = self.net_type(self.data_in[train_row], self.data_out[train_row], **self.net_args)
            if self.verbose:
                # 输出网络详细信息
                print(net)
                X = torch.rand(size=(1, net.in_channels), dtype=self.data_in.dtype)
                flops, params = profile(net, inputs=(X,), verbose=False)
                print("Arch: %s" % self.net_type.__name__,
                      "MACs: %s" % num2str(flops),
                      "Total params: %s" % num2str(params), sep='\n')
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.00005)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.2)
            early_stopping = EarlyStopping(patience=100, skip_epoch=200, verbose=False)

            epoch = 0
            self.loss_history = []
            while epoch < self.max_epochs:
                net.train()
                total_loss, n = 0., 0
                for i, (X, Y) in enumerate(dataloader_train):
                    optimizer.zero_grad()
                    loss = net.calc_loss(X, Y)
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        total_loss += float(loss) * X.shape[0]
                        n += X.shape[0]
                train_loss = total_loss / n
                net.eval()
                total_loss, n = 0., 0
                for i, (X, Y) in enumerate(dataloader_test):
                    loss = net.calc_loss(X, Y)
                    with torch.no_grad():
                        total_loss += float(loss) * X.shape[0]
                        n += X.shape[0]
                test_loss = train_loss if n == 0 else total_loss / n
                if self.verbose:
                    print("Epoch %i: train loss %.4f , test loss %.3e" % (epoch + 1, train_loss, test_loss))
                self.loss_history.append((epoch, train_loss, test_loss))
                scheduler.step()
                early_stopping(test_loss, checkpoint=net)
                if early_stopping:
                    if self.verbose:
                        current_lr = scheduler.get_last_lr()[0]
                        print("Early stopping triggered (lr: %.3e)" % current_lr)
                    break
                epoch += 1
            # 读取损失最小的历史检查点，并发送至主进程
            # epoch = early_stopping.resume(net)
            self.queue.put(('net', net, epoch))

    def plot_loss(self):
        if len(self.loss_history) == 0:
            return
        data = np.array(self.loss_history).T
        fig, ax = plt.subplots()
        ax.plot(data[0], data[1], label='$train$ $set$')
        ax.plot(data[0], data[2], label='$test$ $set$')
        ax.set_xlabel('$Episodes$')
        ax.set_ylabel('$Total$ $loss$')
        ax.legend()
        fig.show()


class FineTuner(Trainer):
    """
    用于微调神经网络的子进程，不包含输入输出的归一化
    """

    def __init__(self, queue: Queue, queue_recv: Queue, finetune_layer: Union[str, List[str]],
                 data_in: torch.Tensor, data_out: torch.Tensor,
                 test_row: Iterable[int] = None, data_in_extra: torch.Tensor = None,
                 lr: float = 5e-3, max_epochs: int = 200, verbose: bool = False):
        super(FineTuner, self).__init__(queue, data_in, data_out,
                                        net_type=None, test_row=test_row, data_in_extra=data_in_extra,
                                        lr=lr, max_epochs=max_epochs, verbose=verbose)
        self.queue_recv = queue_recv
        self.finetune_layer = [finetune_layer] if isinstance(finetune_layer, str) \
            else finetune_layer  # 需要微调的网络层的名称

    def freeze(self, net):
        """冻结部分网络，从而训练特定的网络层"""
        counter = 0
        for name, param in net.named_parameters():
            if np.any(list(map(name.startswith, self.finetune_layer))):
                counter += 1
            else:
                param.requires_grad = False
        if counter == 0:
            raise ValueError("No layer matches name '%s'" % self.finetune_layer)

    def run(self):
        """在给定的数据集上微调基学习器"""
        self.redirect()
        while True:
            net = self.queue_recv.get()
            if net is None:  # 如果收到None，则结束进程
                if self.verbose:
                    print("Received stopping signal, process finished")
                break
            train_row, test_row = self.bootstrap(oob=self.test_row)
            if self.data_in_extra is None:
                dataloader_train = DataLoader(dataset=TensorDataset(self.data_in[train_row],
                                                                    self.data_out[train_row]),
                                              batch_size=2, shuffle=True)
            else:
                dataloader_train = TrainDataLoader(self.data_in[train_row],
                                                   self.data_out[train_row],
                                                   self.data_in_extra,
                                                   batch_size=2, shuffle=True)
            dataloader_test = DataLoader(dataset=TensorDataset(self.data_in[test_row],
                                                               self.data_out[test_row]),
                                         batch_size=8, shuffle=False)
            self.freeze(net)
            if self.verbose:
                # 输出网络详细信息
                summary(net)
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.00005)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=0)
            early_stopping = EarlyStopping(patience=50, skip_epoch=0, verbose=False)

            epoch = 0
            self.loss_history = []
            while epoch < self.max_epochs:
                net.train()
                total_loss, n = 0., 0
                for i, (X, Y) in enumerate(dataloader_train):
                    optimizer.zero_grad()
                    loss = net.calc_loss(X, Y)
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        total_loss += float(loss) * X.shape[0]
                        n += X.shape[0]
                train_loss = total_loss / n
                net.eval()
                total_loss, n = 0., 0
                for i, (X, Y) in enumerate(dataloader_test):
                    loss = net.calc_loss(X, Y)
                    with torch.no_grad():
                        total_loss += float(loss) * X.shape[0]
                        n += X.shape[0]
                test_loss = train_loss if n == 0 else total_loss / n
                if self.verbose:
                    print("Epoch %i: train loss %.4f , test loss %.3e" % (epoch + 1, train_loss, test_loss))
                self.loss_history.append((epoch, train_loss, test_loss))
                scheduler.step()
                early_stopping(test_loss, checkpoint=net)
                if early_stopping:
                    if self.verbose:
                        current_lr = scheduler.get_last_lr()[0]
                        print("Early stopping triggered (lr: %.3e)" % current_lr)
                    break
                epoch += 1
            # 读取损失最小的历史检查点，并发送至主进程
            # epoch = early_stopping.resume(net)
            self.queue.put(('net', net, epoch))


class Plotter:
    """
    绘图工具类，用于在数据集上评估一个或多个网络
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, *nets: BaseNetWrapper):
        if len(nets) == 0:
            raise ValueError("At least one net should be provided")
        nets[0].check_data(x, y)
        self.nets = nets  # 网络集成
        self.data_in = x
        self.data_out = y
        self.latent_dist = self._latent_distribution()  # 原始输入至输入潜空间的归一化参数

    def _latent_distribution(self):
        """计算网络集成的平均输入变换"""
        n_in = self.nets[0].in_channels
        mean = torch.zeros_like(self.nets[0].scale_in.mean)
        std = torch.zeros_like(self.nets[0].scale_in.std)
        for net in self.nets:
            mean += net.scale_in.mean
            std += net.scale_in.std
        return mean[: n_in] / len(self.nets), std[: n_in] / len(self.nets)

    @torch.no_grad()
    def eval_net(self, data_in: torch.Tensor, scale: bool = True):
        """从给定的网络集成采样"""
        data_out_list = []
        for net in self.nets:
            data_out_list.append(net.forward(data_in, x_scale=scale))
        data_out = torch.stack(data_out_list, dim=0)
        mean = data_out.mean(dim=0)
        std = data_out.std(dim=0) if len(self.nets) > 1 else torch.zeros_like(mean)
        return mean, std

    def result(self, out_i: int = None, noise_i: List[int] = None):
        """计算网络集成在给定数据集上的预估值"""
        if noise_i is None:
            data_in = self.data_in
        else:
            data_in = self.data_in.clone()
            data_in[:, noise_i] = 0.
        mean, std = self.eval_net(data_in)
        if out_i is None:
            return mean, std
        else:
            return mean[:, out_i], std[:, out_i]

    @torch.no_grad()
    def score(self, noise_i: List[int] = None, noise_threshold: float = 0.1):
        """计算网络在给定数据集上的误差"""
        if noise_i is None:
            data_in = self.data_in
            data_out = self.data_out
        else:
            valid_row = np.prod(self.data_in[:, noise_i].numpy() <= noise_threshold, axis=1, dtype=bool)
            data_in = self.data_in[valid_row]
            data_in[:, noise_i] = 0.
            data_out = self.data_out[valid_row]
        error_list = []
        for net in self.nets:
            data_out_hat = net.forward(data_in)
            error_list.append(torch.mean(torch.abs(data_out_hat - data_out) / data_out, dim=0))
        error = torch.vstack(error_list)
        mean = error.mean(dim=0)
        std = error.std(dim=0) if len(self.nets) > 1 else torch.zeros_like(mean)
        return mean, std

    def plot_error(self, out_i: int = 0, noise_i: List[int] = None, noise_threshold: float = 0.1,
                   y_label: str = None):
        """绘制网络集成在给定数据集上的拟合曲线和误差分布"""
        x = np.arange(len(self.data_in))
        y = self.data_out[:, out_i]

        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(3, 1, figure=fig)
        ax1 = fig.add_subplot(gs[:2, 0])
        ax2 = fig.add_subplot(gs[2, 0])
        y_hat, y_hat_std = self.result(out_i)
        y_error = 100 * torch.abs(y_hat / y - 1)
        ax1.plot(x, y_hat, 'b.-', label="fitting")
        ax1.fill_between(x, y_hat - 2 * y_hat_std, y_hat + 2 * y_hat_std,
                         facecolor='blue', alpha=0.2)
        ax2.bar(x, y_error, color='dodgerblue')
        if noise_i is not None:
            _y_hat, _y_hat_std = self.result(out_i, noise_i)
            _y_error = 100 * torch.abs(_y_hat / y - 1)
            ax1.plot(x, _y_hat, 'r.-', label="denoise")
            ax1.fill_between(x, _y_hat - 2 * _y_hat_std, _y_hat + 2 * _y_hat_std,
                             facecolor='red', alpha=0.2)
            invalid_row = np.sum(self.data_in[:, noise_i].numpy() > noise_threshold, axis=1, dtype=bool)
            for _x in np.argwhere(invalid_row)[:, 0]:
                ax1.axvline(_x, c='gray', alpha=0.3)  # 标识超出噪声阈值的样本
            shift = _y_error - y_error
            shift[invalid_row] = 0.
            colors = np.array(['#FF5151'] * len(x))
            colors[shift < 0] = '#93FF93'
            ax2.bar(x, shift, color=colors, bottom=y_error)
            ax2.set_ylim(0, 1.05 * (torch.clamp(shift, min=0) + y_error).max().item())
        ax1.plot(x, y, 'k.-', label="origin")
        ax1.set_xticklabels([])
        ax2.set_xlabel("$case_i$", fontsize=20)
        if y_label:
            ax1.set_ylabel(y_label, fontsize=20)
            ax2.set_ylabel(f"$error$ $of$ {y_label} %", fontsize=20)
        else:
            ax1.set_ylabel("$value$", fontsize=20)
            ax2.set_ylabel("$error$ %", fontsize=20)
        ax1.legend(loc='lower right')
        fig.tight_layout()
        fig.show()

    def _input_interp(self, data_in: torch.Tensor, interp_i: int, k: int = 0.5):
        """基于给定的数据集，对输入数据的某一通道使用LWLR进行插值"""
        x_ind = np.ones(self.data_in.shape[1], dtype=bool)
        x_ind[interp_i] = False
        data = self.data_in.clone()
        data[:, interp_i] = 0.
        scale = Normalize(self.data_in.shape[1], data)
        data = scale(self.data_in).numpy()
        curve = CurveFitting(data[:, interp_i], *data[:, x_ind].T, k=k)
        y = []
        for p in scale(data_in)[:, x_ind]:
            y.append(curve.Estimate(p.tolist()))
        data_in[:, interp_i] = torch.tensor(y)
        return data_in

    def _pre_plot(self, *in_i: int, n: int = 10, margin: float = 0.,
                  section: Iterable[float] = None, interp_i: int = None):
        """生成特定坐标截面上的等间隔数据，并计算数据点至界面的距离"""
        if section is None:
            section = self.latent_dist[0]
        else:
            section = torch.tensor(section, dtype=self.data_in.dtype)
            if len(section) != self.latent_dist[0].shape[0]:
                raise ValueError("The length of section should be equal to input channels (%d) of net"
                                 % self.latent_dist[0].shape[0])
            ind = section.isnan()  # 将section中为nan的元素替换为默认值
            section[ind] = self.latent_dist[0][ind]
        _min = self.data_in.min(dim=0)[0]
        _max = self.data_in.max(dim=0)[0]
        margin = margin * (_max - _min)
        _min = _min - margin
        _max = _max + margin
        divide = [torch.linspace(_min[i], _max[i], n, dtype=self.data_in.dtype) for i in in_i]
        divide_mesh = torch.meshgrid(*divide, indexing='ij')
        data_in = torch.tile(section, (n ** len(in_i), 1))
        data_in[:, in_i] = torch.stack(divide_mesh, dim=0).flatten(1).T
        if interp_i is not None:
            # 对某一输入通道（如噪声通道）进行插值
            data_in = self._input_interp(data_in, interp_i)
        arr = (self.data_in - section) / self.latent_dist[1]
        arr = np.delete(arr.numpy(), in_i, axis=1)
        distance = np.linalg.norm(arr, axis=1)
        return data_in, distance

    def plot(self, in_i: int = 0, out_i: int = 0, n: int = 100, margin: float = 0.,
             interp_i: int = None, line_dim: int = None, section: Iterable[Any] = None,
             x_label: str = None, y_label: str = None):
        """在特定投影坐标面上绘制网络的拟合曲线"""
        fig, ax = plt.subplots()
        if section is None or (line_dim is None and np.ndim(section) == 1):
            # 在section定义的投影面上绘制
            data_in, distance = self._pre_plot(in_i, n=n, margin=margin, section=section)
            data_out, data_out_std = self.eval_net(data_in)
            ax.scatter(self.data_in[:, in_i], self.data_out[:, out_i],
                       s=16, c=distance, cmap='viridis', label="Observation")
            ax.plot(data_in[:, in_i], data_out[:, out_i], label="Prediction")
            ax.fill_between(data_in[:, in_i],
                            data_out[:, out_i] - 2 * data_out_std[:, out_i],
                            data_out[:, out_i] + 2 * data_out_std[:, out_i],
                            facecolor='blue', alpha=0.2, label="95% confidence interval")  # 95.4%置信区间
            if interp_i is not None:
                # 当只有一个投影面时，可以绘制额外绘制一条包含插值的曲线
                if interp_i == in_i:
                    raise ValueError("Dimension of interpolation should be different from input")
                if self.data_in.shape[1] < 3:
                    raise ValueError("Extra 2 dimensions except input channel are needed to interpolate")
                data_in, distance = self._pre_plot(in_i, n=n, margin=margin, section=section, interp_i=interp_i)
                data_out, data_out_std = self.eval_net(data_in)
                line, = ax.plot(data_in[:, in_i], data_out[:, out_i], '--')
                ax.fill_between(data_in[:, in_i],
                                data_out[:, out_i] - 2 * data_out_std[:, out_i],
                                data_out[:, out_i] + 2 * data_out_std[:, out_i],
                                facecolor=line.get_color(), alpha=0.2)
        else:
            # 在多个投影面上分别绘制
            if line_dim is None:
                # 若未指定line_dim参数，则假定section为按行排列的二维数组
                sections = list(section)
            else:
                # 若指定line_dim参数，按section改变特定维度来定义多个投影面
                sections = []
                for value in section:
                    _section = [np.nan] * self.data_in.shape[1]
                    _section[line_dim] = value
                    sections.append(_section)
            for i, _section in enumerate(sections, start=1):
                data_in, distance = self._pre_plot(in_i, n=n, margin=margin, section=_section)
                if i == 1:
                    # 样本点的距离基于第一个投影面计算
                    ax.scatter(self.data_in[:, in_i], self.data_out[:, out_i],
                               s=16, c=distance, cmap='viridis', label="Observation")
                data_out, data_out_std = self.eval_net(data_in)
                ax.plot(data_in[:, in_i], data_out[:, out_i], label="Prediction-%d" % i)
        if x_label:
            ax.set_xlabel(x_label, fontsize=20)
        else:
            ax.set_xlabel("${in}_{%d}$" % in_i)
        if y_label:
            ax.set_ylabel(y_label, fontsize=20)
        else:
            ax.set_ylabel("${out}_{%d}$" % out_i)
        ax.autoscale()
        ax.legend()
        ax.grid()
        fig.show()

    def plot_all(self, n: int = 100, margin: float = 0., section: Iterable[float] = None,
                 x_label: List[str] = None, y_label: List[str] = None):
        """在潜空间的原点截面上，绘制网络各个通道间的拟合曲线"""
        n_in = self.nets[0].in_channels
        n_out = self.nets[0].out_channels
        fig, axes = plt.subplots(n_out, n_in, sharex='col', sharey='row', figsize=(10, 8))
        axes = np.reshape(axes, (n_out, -1))
        fig.suptitle(f"Origin cross section of fitting hyperplane (dim: {n_in} -> {n_out})")
        for in_i in range(n_in):
            data_in, distance = self._pre_plot(in_i, n=n, margin=margin, section=section)
            data_out, data_out_std = self.eval_net(data_in)
            for out_i in range(n_out):
                ax = axes[out_i, in_i]
                ax.scatter(self.data_in[:, in_i], self.data_out[:, out_i], s=16, c=distance, cmap='viridis')
                ax.plot(data_in[:, in_i], data_out[:, out_i])
                ax.fill_between(data_in[:, in_i],
                                data_out[:, out_i] - 2 * data_out_std[:, out_i],
                                data_out[:, out_i] + 2 * data_out_std[:, out_i],
                                facecolor='blue', alpha=0.2)  # 95.4%置信区间
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

    def plot3d(self, in_i: int = 0, in_j: int = 1, out_i: int = 0, n: int = 20,
               margin: float = 0., section: Iterable[float] = None):
        """指定两个输入维度和一个输出维度绘制拟合曲面的三维投影"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        data_in, distance = self._pre_plot(in_i, in_j, n=n, margin=margin, section=section)
        data_out, data_out_var = self.eval_net(data_in)
        surf = ax.plot_surface(data_in[:, in_i].reshape((n, n)),
                               data_in[:, in_j].reshape((n, n)),
                               data_out[:, out_i].reshape((n, n)),
                               cmap='coolwarm', linewidth=0, alpha=0.6)
        fig.colorbar(surf, shrink=0.5, aspect=6)
        '''ax.plot_wireframe(data_in[:, in_i].reshape((n, n)),
                          data_in[:, in_j].reshape((n, n)),
                          data_out[:, out_i].reshape((n, n)),
                          rstride=2, cstride=2, alpha=0.6)'''
        ax.scatter(self.data_in[:, in_i], self.data_in[:, in_j], self.data_out[:, out_i],
                   s=16, c=distance, cmap='viridis')
        plt.show()


def train_net(net_type, data_in, data_out, test_row, data_in_extra=None, thread_n=1, n_per_thread=1) -> list:
    """批量训练多个同质学习器用于集成，可指定多进程"""
    net_n = thread_n * n_per_thread
    queue = Queue()
    worker = []
    for _ in range(thread_n):
        p = Trainer(queue, data_in, data_out, net_type=net_type,
                    test_row=test_row, data_in_extra=data_in_extra,
                    net_n=n_per_thread, lr=5e-3, max_epochs=500, verbose=False)
        p.start()
        worker.append(p)
    net_list = []
    while True:
        ret = queue.get()
        if ret[0] == 'error':
            _, string = ret
            print(string, file=sys.stderr)
        elif ret[0] == 'net':
            _, net, epoch = ret
            net_list.append(net)
            print(f"[Net {len(net_list)}/{net_n}] Train epochs: {epoch}")
            if len(net_list) == net_n:
                break
        else:
            pass
    return net_list


def finetune_net(finetune_layer, data_in, data_out, test_row, *nets, data_in_extra=None, thread_n=1) -> list:

    class SendNet(Thread):

        def __init__(self, queue, *nets):
            super(SendNet, self).__init__(daemon=True)
            self.queue = queue
            self.nets = nets

        def run(self):
            i = 0
            n = len(self.nets)
            while True:
                if self.queue.empty():
                    self.queue.put(self.nets[i])
                    i += 1
                if i >= n:
                    break

    q_left = Queue()
    q_right = Queue()
    worker = []
    for _ in range(min(len(nets), thread_n)):
        p = FineTuner(q_right, q_left, finetune_layer, data_in, data_out, test_row=test_row,
                      data_in_extra=data_in_extra, lr=5e-3, max_epochs=200, verbose=False)
        p.start()
        worker.append(p)
    # 发送待微调的网络
    sender = SendNet(q_left, *nets)
    sender.start()
    # 接送worker信息
    net_list = []
    while True:
        ret = q_right.get()
        if ret[0] == 'error':
            _, string = ret
            print(string, file=sys.stderr)
        elif ret[0] == 'net':
            _, net, epoch = ret
            net_list.append(net)
            print(f"[Net {len(net_list)}/{len(nets)}] Train epochs: {epoch}")
            if len(net_list) == len(nets):
                break
        else:
            pass
    # 向worker发送结束信号
    for _ in range(min(len(nets), thread_n)):
        q_left.put(None)
    return net_list


def save_net(path: str, *nets: nn.Module, desc: Any = None) -> None:
    """保存网络至文件"""
    obj = {'nets': list(nets),
           'desc': os.path.basename(path) if desc is None else desc,
           'date': time.asctime()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)
    if os.path.exists(path):
        print(f"{len(nets)} net(s) of {int(os.path.getsize(path) / 1024)} KB saved to '{path}'")


def load_net(path: str) -> list:
    """从文件中载入网络"""
    obj = torch.load(path)
    print(f"Load net: '{obj['desc']}' (created at {obj['date']})")
    return obj['nets']


def test_data(mesh='150'):
    """读取等效喉径0.2的塞式喷管数据集"""
    # 读取特定网格下的计算结果
    data = pd.read_excel(r".\plug_result_mesh.xlsx", sheet_name=mesh)
    print(data.columns)
    data_in = torch.tensor(data[['inlet_p', 'atmo_p', 'report-def-continuity']].to_numpy())
    data_out = torch.tensor(data[['Cf']].to_numpy())  # 单输出用于堆叠网络
    data_out_all = torch.tensor(data[['report-def-massflow', 'report-def-thrust', 'Cf', 'SpecImpulse']].to_numpy())
    noise_index = [-1]
    noise_deprecated = 10

    # 剔除大误差输入
    if noise_index:
        retain_row = np.logical_and(
            np.prod(np.abs(data_in[:, noise_index].numpy()) <= noise_deprecated, axis=1, dtype=bool),
            np.logical_and(data_out.numpy() > 0, data_out.numpy() <= data[['Cf_max']].to_numpy()).flatten()
        )
        if retain_row.sum() < len(data):
            data_in = data_in[retain_row]
            data_out = data_out[retain_row]
            data_out_all = data_out_all[retain_row]
            print("%d samples have been deprecated" % (len(data) - retain_row.sum()))

    # 构造额外输入数据集
    X, Y = torch.meshgrid(torch.tensor([1.5e6, 4.5e6, 9e6, 16e6, 24e6]),
                          torch.tensor([25., 4e3, 14e3, 28e3, 48e3, 80e3, 120e3]), indexing='ij')
    data_extra = torch.vstack([X.flatten(), Y.flatten(),
                               torch.zeros(X.shape[0] * X.shape[1])]).type(dtype=torch.float64).T

    return data, data_in, data_out, data_out_all, data_extra


def test_trainer():
    data, data_in, data_out, data_out_all, data_extra = test_data()
    noise_index = [-1]
    noise_threshold = 0.1

    # 测试网络结构 (BaseNetWrapper etc.)
    net = BaseNetWrapper(3, 4)
    X = torch.rand(size=(1, net.in_channels), dtype=torch.float64)
    print(profile(net, inputs=(X,), verbose=False))

    # 测试含物理监督样本的网络训练 (Trainer: data_in_extra)
    queue = Queue()
    trainer = Trainer(queue, data_in, data_out_all, net_type=DenoisePhysOutWrapper,
                      test_row=[2, 8, 14, 20], data_in_extra=data_extra,
                      net_n=1, lr=5e-3, max_epochs=500, verbose=True)
    trainer.run()
    tag, net, epoch = queue.get()
    print(Plotter(data_in, data_out_all, net).score())

    # 测试网络批量训练 (train_net)
    nets = train_net(DenoisePhysOutWrapper, data_in, data_out_all, test_row=[], thread_n=8, n_per_thread=16)
    summary(nets[0], (1000, nets[0].in_channels), device='cpu')

    # 测试网络性能 (Plotter)
    plotter = Plotter(data_in, data_out_all, *nets)
    torch.set_printoptions(sci_mode=True)
    print([value * 100 for value in plotter.score()], sep='\n')
    print([value * 100 for value in plotter.score(noise_i=noise_index, noise_threshold=noise_threshold)], sep='\n')
    torch.set_printoptions(profile='default')
    noise = data_in[:, noise_index[0]].mean()
    plotter.plot(in_i=0, out_i=2, margin=0.1, section=[np.nan, 8325., np.nan], interp_i=2,
                 x_label="$p_0$ / $Pa$", y_label="$C_f$")
    plotter.plot(in_i=1, out_i=2, margin=0.1, section=[6e6, np.nan, np.nan], interp_i=2,
                 x_label="$p_e$ / $Pa$", y_label="$C_f$")
    plotter.plot_all(x_label=["$p_0$ / $Pa$", "$p_e$ / $Pa$", "$R_m$"],
                     y_label=["$Q_m$ / $kg·s^{-1}$", "$F$ / $N$", "$C_f$", "$I_s$"])
    plotter.plot_all(x_label=["$p_0$ / $Pa$", "$p_e$ / $Pa$", "$R_m$"],
                     y_label=["$Q_m$ / $kg·s^{-1}$", "$F$ / $N$", "$C_f$", "$I_s$"],
                     section=[np.nan, np.nan, noise])
    plotter.plot3d(out_i=2)

    plotter.plot(in_i=0, out_i=2, margin=0.1, line_dim=2, section=[0., np.nan])
    plotter.plot(in_i=0, out_i=2, margin=0.1, section=[[np.nan, 8325., np.nan], [np.nan, 8325., noise]])
    plotter.plot(in_i=1, out_i=2, margin=0.1, section=[[6e6, np.nan, np.nan], [6e6, np.nan, noise]])

    # 训练不同类型的网络并保存为文件
    nets_name_1 = ['DenoiseNet(2+1~4,full,10x4x3,512).pth', 'DenoiseNet(2+1~1,full,10x4x3,512).pth',
                   'DenoiseNet(4+1~4,full,10x4x3,512).pth', 'DenoiseNet(4+1~1,full,10x4x3,512).pth',
                   'DenoiseNet(2+1~4p,full,10x4x3,512).pth', 'DenoiseNet(4+1~4p,full,10x4x3,512).pth']
    nets_list_1 = [
        train_net(DenoiseWrapper, data_in, data_out_all, test_row=[], thread_n=8, n_per_thread=64),
        train_net(DenoiseWrapper, data_in, data_out, test_row=[], thread_n=8, n_per_thread=64),
        train_net(DenoisePhysInWrapper, data_in, data_out_all, test_row=[], thread_n=8, n_per_thread=64),
        train_net(DenoisePhysInStackWrapper, data_in, data_out, test_row=[], thread_n=8, n_per_thread=64),
        train_net(DenoisePhysOutWrapper, data_in, data_out_all, test_row=[], thread_n=8, n_per_thread=64),
        train_net(DenoisePhysInOutWrapper, data_in, data_out_all, test_row=[], thread_n=8, n_per_thread=64),
    ]
    nets_name_2 = ['DenoiseNet(2+1~4,part,10x4x3,512).pth', 'DenoiseNet(2+1~1,part,10x4x3,512).pth',
                   'DenoiseNet(4+1~4,part,10x4x3,512).pth', 'DenoiseNet(4+1~1,part,10x4x3,512).pth',
                   'DenoiseNet(2+1~4p,part,10x4x3,512).pth', 'DenoiseNet(4+1~4p,part,10x4x3,512).pth']
    nets_list_2 = [
        train_net(DenoiseWrapper, data_in, data_out_all, test_row=[2, 8, 14, 20], thread_n=8, n_per_thread=64),
        train_net(DenoiseWrapper, data_in, data_out, test_row=[2, 8, 14, 20], thread_n=8, n_per_thread=64),
        train_net(DenoisePhysInWrapper, data_in, data_out_all, test_row=[2, 8, 14, 20], thread_n=8, n_per_thread=64),
        train_net(DenoisePhysInStackWrapper, data_in, data_out, test_row=[2, 8, 14, 20], thread_n=8, n_per_thread=64),
        train_net(DenoisePhysOutWrapper, data_in, data_out_all, test_row=[2, 8, 14, 20], thread_n=8, n_per_thread=64),
        train_net(DenoisePhysInOutWrapper, data_in, data_out_all, test_row=[2, 8, 14, 20], thread_n=8, n_per_thread=64),
    ]
    for name, nets in zip(nets_name_1 + nets_name_2, nets_list_1 + nets_list_2):
        save_net(r'.\nets\%s' % name, *nets)
    pass


def test_finetuner():
    data, data_in, data_out, data_out_all, _ = test_data()
    data_f, data_f_in, data_f_out, data_f_out_all, _ = test_data(mesh='1200')
    # train_row = np.random.choice(np.arange(len(data_f_in)), 4, replace=False)
    train_row = np.array([2, 5, 13, 16])
    train_row.sort()
    test_row = np.setdiff1d(np.arange(len(data_f_in)), train_row)
    print("Train sample:\n" + '\n'.join(map(lambda row: f"\t{row[0]}\t{row[1]}", data_f_in[train_row])))

    nets1200_1 = load_net(r'./nets/DenoiseNet1200(4+1~1,full,10x4x3,512).pth')
    nets1200_2 = load_net(r'./nets/DenoiseNet1200(4+1~1,part,10x4x3,512).pth')

    nets = load_net(r'./nets/DenoiseNet(4+1~1,part,10x4x3,512).pth')
    print("Net parameters:")
    for name, param in nets[0].named_parameters():
        print('\t' + name)

    # 收敛曲线比较
    queue = Queue()
    trainer1 = Trainer(queue, data_f_in, data_f_out, net_type=DenoisePhysInStackWrapper,
                       test_row=[], net_n=1, verbose=True)
    trainer1.run()
    _, net1, _ = queue.get()
    trainer2 = Trainer(queue, data_f_in, data_f_out, net_type=DenoisePhysInStackWrapper,
                       test_row=test_row, net_n=1, verbose=True)
    trainer2.run()
    _, net2, _ = queue.get()
    queue2 = Queue()
    tuner = FineTuner(queue, queue2, 'net.linear_out', data_f_in, data_f_out,
                      test_row=test_row, verbose=True)
    queue2.put(load_net(r'./nets/DenoiseNet(4+1~1,part,10x4x3,512).pth')[0])
    queue2.put(None)
    tuner.run()
    _, net3, _ = queue.get()
    trainer1.plot_loss()
    trainer2.plot_loss()
    tuner.plot_loss()
    fig, ax = plt.subplots()
    data_1 = np.array(trainer1.loss_history).T
    ax.plot(data_1[0], data_1[1], 'r--', linewidth=1)
    ax.plot(data_1[0], data_1[2], 'r-', linewidth=1, label="$262w,$ $19$")
    data_2 = np.array(trainer2.loss_history).T
    ax.plot(data_2[0], data_2[1], 'b--', linewidth=1)
    ax.plot(data_2[0], data_2[2], 'b-', linewidth=1, label="$262w,$ $4$")
    data_3 = np.array(tuner.loss_history).T
    data_3 = np.hstack([data_3, np.vstack([np.arange(200, 500),
                                           np.ones(300) * data_3[1, -1],
                                           np.ones(300) * data_3[2, -1]])])
    ax.plot(data_3[0], data_3[1], 'g--', linewidth=1)
    ax.plot(data_3[0], data_3[2], 'g-', linewidth=1, label="$4.4w,$ $4$ $(TL)$")
    ax.set_yscale('log')
    ax.set_xlabel("$Epoch$")
    ax.set_ylabel("$Loss$")
    ax.legend(loc='lower right')
    ax.grid()
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示指数坐标系标签
    fig.show()

    # 计算时间比较
    t0 = -time.time()
    train_net(DenoisePhysInStackWrapper, data_in, data_out, test_row=[], thread_n=8, n_per_thread=64)
    t0 += time.time()
    t1 = -time.time()
    train_net(DenoisePhysInStackWrapper, data_f_in, data_f_out, test_row=[], thread_n=8, n_per_thread=64)
    t1 += time.time()
    t2 = -time.time()
    train_net(DenoisePhysInStackWrapper, data_f_in, data_f_out, test_row=test_row, thread_n=8, n_per_thread=64)
    t2 += time.time()
    t3 = -time.time()
    nets_f = finetune_net('net.linear_out', data_f_in, data_f_out, test_row, *nets, thread_n=8)
    t3 += time.time()
    print(f"Training Time Board (sec):\n\tAll data: {t1:.0f}\n\tPart data: {t2:.0f}\n\tPart data (finetune): {t3:.0f}")

    plotter = Plotter(data_f_in, data_f_out, *nets_f)
    print([value * 100 for value in plotter.score()], sep='\n')
    print([value * 100 for value in plotter.score(noise_i=[2], noise_threshold=0.1)], sep='\n')
    plotter.plot_error(noise_i=[2], y_label='$C_f$')
    pass


def test_arch_plot(full_train=True):
    """网络架构和规模对比"""
    # FCN
    data1_x = [[370, 414], [1340, 1424], [2910, 3034], [5080, 5244], [7850, 8054], [11220, 11464]]
    if full_train:
        data1_y = \
            [[[0.0086, 0.0097, 0.0008, 0.0009], [0.0038, 0.0040, 0.0003, 0.0003]],  # 10x3
             [[0.0060, 0.0064, 0.0005, 0.0007], [0.0096, 0.0089, 0.0005, 0.0005]],  # 20x3
             [[0.0046, 0.0051, 0.0005, 0.0006], [0.0043, 0.0045, 0.0003, 0.0004]],  # 30x3
             [[0.0049, 0.0056, 0.0005, 0.0006], [0.0070, 0.0080, 0.0005, 0.0005]],  # 40x3
             [[0.0047, 0.0050, 0.0005, 0.0006], [0.0062, 0.0042, 0.0005, 0.0005]],  # 50x3
             [[0.0050, 0.0054, 0.0005, 0.0006], [0.0074, 0.0080, 0.0004, 0.0004]]]  # 60x3
    else:
        data1_y = \
            [[[0.0122, 0.0127, 0.0016, 0.0017], [0.0043, 0.0039, 0.0004, 0.0003]],
             [[0.0079, 0.0086, 0.0014, 0.0015], [0.0041, 0.0054, 0.0002, 0.0002]],
             [[0.0079, 0.0084, 0.0014, 0.0016], [0.0039, 0.0041, 0.0004, 0.0003]],
             [[0.0078, 0.0092, 0.0012, 0.0013], [0.0064, 0.0072, 0.0005, 0.0005]],
             [[0.0083, 0.0086, 0.0015, 0.0016], [0.0048, 0.0056, 0.0004, 0.0004]],
             [[0.0077, 0.0082, 0.0014, 0.0016], [0.0049, 0.0054, 0.0003, 0.0003]]]
    # DenoiseNet
    data2_x = [[335, 389], [635, 719], [2470, 2634], [5505, 5749], [9740, 10064]]
    if full_train:
        data2_y = \
            [[[0.0092, 0.0101, 0.0006, 0.0007], [0.0025, 0.0026, 0.0002, 0.0002]],  # 5(x2)x3
             [[0.0066, 0.0070, 0.0005, 0.0006], [0.0025, 0.0030, 0.0003, 0.0003]],  # 5(x4)x3
             [[0.0039, 0.0042, 0.0004, 0.0005], [0.0024, 0.0026, 0.0002, 0.0002]],  # 10(x4)x3
             [[0.0033, 0.0037, 0.0004, 0.0005], [0.0023, 0.0027, 0.0003, 0.0003]],  # 15(x4)x3
             [[0.0037, 0.0042, 0.0004, 0.0005], [0.0037, 0.0051, 0.0003, 0.0003]]]  # 20(x4)x3
    else:
        data2_y = \
            [[[0.0160, 0.0158, 0.0014, 0.0016], [0.0055, 0.0056, 0.0004, 0.0004]],
             [[0.0118, 0.0126, 0.0012, 0.0013], [0.0037, 0.0043, 0.0003, 0.0003]],
             [[0.0072, 0.0070, 0.0010, 0.0011], [0.0031, 0.0027, 0.0004, 0.0003]],
             [[0.0056, 0.0058, 0.0009, 0.0011], [0.0023, 0.0024, 0.0003, 0.0003]],
             [[0.0050, 0.0054, 0.0009, 0.0010], [0.0021, 0.0023, 0.0002, 0.0002]]]
    data1_x = np.array(data1_x)[:, 1]
    data1_y = np.array(data1_y)[:, :, 0].T * 100
    data2_x = np.array(data2_x)[:, 1]
    data2_y = np.array(data2_y)[:, :, 0].T * 100
    fig, ax = plt.subplots()
    fig.suptitle("Performance of different network size")
    ax.errorbar(data1_x, data1_y[0], yerr=data1_y[1], fmt='o-', color='skyblue',
                ecolor='skyblue', capsize=2, elinewidth=1)
    ax.errorbar(data2_x, data2_y[0], yerr=data2_y[1], fmt='o-', color='coral',
                ecolor='coral', capsize=2, elinewidth=1)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax.set_xlabel("$Params$")
    ax.set_ylabel("$Error$ $of$ $Q_m$ (%)")
    ax.legend(["$FCN$", "$DenoiseNet$"])
    # ax.grid()
    fig.show()


def test_denoise_plot(full_train=True):
    """网络拟合和去噪能力对比"""
    data, data_in, data_out, data_out_all, _ = test_data()
    if full_train:
        nets = load_net(r'./nets/DenoiseNet(4+1~4p,full,10x4x3,512).pth')
    else:
        nets = load_net(r'./nets/DenoiseNet(4+1~4p,part,10x4x3,512).pth')
    plotter = Plotter(data_in, data_out_all, *nets)
    plotter.plot_error(out_i=0, noise_i=[2], y_label="$C_f$")
    pass


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

        data_in = self.data_in[:, self.index_in]
        data_out = self.data_out[:, self.index_out]
        data_noise = self.data_noise[:, self.index_noise]
        self.data_in_scale = normalize(data_in, type='maxmin')
        self.data_out_scale = normalize(data_out, type='zscore')
        valid_row = np.prod(data_noise <= self.noise_threshold, axis=1, dtype=bool)

        dataloader = DataLoader(dataset=TensorDataset(torch.from_numpy(np.hstack([self.data_in_scale[0](data_in),
                                                                                  data_noise])),
                                                      torch.from_numpy(self.data_out_scale[0](data_out))),
                                batch_size=1, shuffle=True)
        input_n = data_in.shape[1] + data_noise.shape[1]
        # 设定网络架构
        net = DenoiseNet(input_channels=input_n, num_channels=10 * input_n, output_channels=data_out.shape[1],
                         block_n=3)
        # net = FullConnectNet(input_channels=input_n, num_channels=5 * input_n,
        #                      out_channels=data_out.shape[1], layer_n=3)
        if print_detail:
            summary(net, (1, input_n), device='cpu')
        net.double()
        net.apply(init_weights)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.00005)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.2)
        loss_f = nn.SmoothL1Loss()  # 使用平滑L1损失函数以减小离群样本点的影响

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

        valid_row = self.data_noise[:, noise_i] <= self.noise_threshold
        data_in = self.data_in[:, self.index_in]
        self.data_in_scale = normalize(data_in[valid_row], type='zscore')
        data_noise = self.data_noise[:, self.index_noise]
        data_noise[~ valid_row] = 0.
        data_out = self.data_out[:, self.index_out]
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
    # denoise.by_net(lr=0.005, num_epochs=400, in_i=[0, 1], out_i=[2])
    # denoise.by_lwlr(k=0.2, in_i=[0, 1], out_i=2)
    # denoise.plot(x_label=["$p_0$ / $Pa$", "$p_e$ / $Pa$"], y_label=["$C_f$"])
    by_net(denoise, in_idx=[0, 1], out_idx=[2], x_label=["$p_0$ / $Pa$", "$p_e$ / $Pa$"], y_label=["$C_f$"])


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
                                 output_channels=self.data_out.shape[1], block_n=3)
                # net = FullConnectNet(input_channels=self.data_in.shape[1], num_channels=5 * self.data_in.shape[1],
                #                     out_channels=self.data_out.shape[1], layer_n=3)
                # summary(net, (1, self.data_in.shape[1]), device='cpu')
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
        data_in_std = np.zeros((n ** 2, n_in))
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
        # data_out = net_bagging.sample(data_in=data_in)
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
        return Z.sum() / n_int ** 2

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
        return (Z_11.sum() + Z_12.sum() + Z_21.sum() + Z_22.sum()) / (4 * n_int ** 2)

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
        return (16 * Z.sum() + 5 * (Z_11.sum() + Z_12.sum() + Z_21.sum() + Z_22.sum())) / (36 * n_int ** 2)

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

    # test([25, 50, 100, 200, 400, 800])
    test([20, 30, 50, 80, 120, 170, 230, 300])

    return gauss_3(400)  # 误差小于1e-8


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
            'Cp': 2837.76,  # 1006.43
            'K': 0.242,  # 0.0242
            'M': 20.9,  # 28.966
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
        data = self.data[self.data['report-def-continuity'] < continuity_limit]  # 未收敛（发散）的结果直接丢弃
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
        net = DenoiseNet(input_channels=input_n, num_channels=20 * input_n, output_channels=output_n, block_n=5)
        # net = FullConnectNet(input_channels=input_n, num_channels=5 * input_n, out_channels=output_n, layer_n=3)
        if print_detail:
            summary(net, (1, input_n), device='cpu')
        net.double()
        net.apply(init_weights)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.00005)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400], gamma=0.2)
        loss_d_f = nn.SmoothL1Loss()  # 使用平滑L1损失函数以减小离群样本点的影响
        loss_p_f = nn.SmoothL1Loss()  # 物理损失项

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
    # net_bagging.plot3d(in_i=0, in_j=1, out_i=2)

    # performance(r'.\plug_result.csv')


if __name__ == '__main__':
    # test_trainer()
    # test_arch_plot()
    # test_denoise_plot()
    test_finetuner()
    # test_denoise()
    # test_pi_denoise()
    pass
