import copy
import os
import time
from typing import List, Type
from copy import deepcopy
import math

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.distributions.transforms import TanhTransform
from torch.distributions import Categorical, Normal, TransformedDistribution
from torch.utils.data import DataLoader, TensorDataset

from ML.torch.utils import all_seed, ReplayBuffer, net_arch
from ML.torch.module import *
from ML.post import PlotAniNet, Timer
from cfd_toolbox.plot import *

from denoise import DenoiseNet


class AgentConfig:

    def __init__(self, **kwargs):
        self.device = 'cuda'
        self.dtype = torch.float64

        self.observation_sample_n = 200
        self.observation_feature_n = 4
        self.action_n = 4  # 动作空间维度（非离散动作）

        self.sensor_hidden_dim = 16  # 感知器输出的特征维度
        self.actor_hidden_n = 5  # 演员网络的隐藏层层数
        self.actor_hidden_dim = 1024  # 演员网络的隐藏层通道数
        self.critic_hidden_n = 5  # 评论员网络的隐藏层层数
        self.critic_hidden_dim = 512  # 评论员网络的隐藏层通道数

        self.sensor_lr = 0.0001  # 感知器的学习率（微调）
        self.actor_lr = 0.002  # 演员网络的学习率
        self.critic_lr = 0.002  # 评论员网络的学习率
        self.gamma = 0.99  # 奖励折扣因子
        self.k_epochs = 4  # 更新策略网络的次数（内迭代）
        self.batch_size = 50  # 内迭代的批量大小
        self.eps_clip = 0.2  # PPO-clip的裁剪系数ε
        self.entropy_coef = -0.05  # 熵奖励系数（正数：探索性策略；负数：确定性策略），常用0.0 ~ 0.005
        self.update_freq = 50  # 更新频率

        self.__dict__.update(kwargs)


def init_weights(m: nn.Module):
    """基于方差缩放的参数初始化"""
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.Conv1d:
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('linear'))


class AttentionSensor(nn.Module):
    """
    环境感知块（简单注意力）

    对于单批量，输入为二维张量（环境观测），输出为一维张量（环境特征）
    用于流场线物理量的特征提取
    """

    def __init__(self, input_sample_dim: int, input_feature_dim: int, output_dim: int, **kwargs):
        super(AttentionSensor, self).__init__()
        self.out_channels = output_dim
        self.attn = SelfAttention(input_channels=input_feature_dim, output_channels=1, **kwargs)
        self.mlp = nn.Linear(in_features=input_sample_dim, out_features=output_dim)

    def forward(self, x: torch.Tensor):
        x = self.attn(x)[..., 0]
        return F.relu(self.mlp(F.relu(x)))


class MultiAttentionSensor(nn.Module):
    """
    环境感知块（门控多头注意力）

    对于单批量，输入为二维张量（环境观测），输出为一维张量（环境特征）
    用于流场线物理量的特征提取
    """

    def __init__(self, input_sample_dim: int, input_feature_dim: int, output_dim: int,
                 head_dim: int = None, head_n: int = 1):
        super(MultiAttentionSensor, self).__init__()
        self.out_channels = output_dim
        self.attn = Transformer(input_feature_dim, input_feature_dim,
                                head_channels=head_dim, head_num=head_n, gated=True)
        self.mlp = MLPReduce2D(input_sample_dim, input_feature_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.attn(x))
        return F.tanh(self.mlp(x))


class RecursiveSensor(nn.Module):
    """
    环境感知块（循环）

    对于单批量，输入为二维张量（环境观测），输出为一维张量（环境特征）
    用于流场线物理量的特征提取
    """

    def __init__(self, input_sample_dim: int, input_feature_dim: int, output_dim: int,
                 hidden_dim: int = None, **kwargs):
        super(RecursiveSensor, self).__init__()
        self.out_channels = output_dim
        if hidden_dim is None:
            self.rnn = nn.LSTM(input_size=input_feature_dim, hidden_size=output_dim, batch_first=True, **kwargs)
            self.mlp = nn.Sequential()
        else:
            self.rnn = nn.LSTM(input_size=input_feature_dim, hidden_size=hidden_dim, batch_first=True, **kwargs)
            self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None):
        if lengths is None:
            output, (hn, cn) = self.rnn(x)
        else:
            # 支持变长序列，详见pack_padded_sequence: `enforce_sorted = True` is only necessary for ONNX export.
            x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            output, (hn, cn) = self.rnn(x_packed)
        return self.mlp(hn[0])


class FlatSensor(nn.Module):
    """
    环境感知块（简单卷积）

    对于单批量，输入为二维张量（环境观测），输出为一维张量（环境特征）
    用于流场线物理量的特征提取
    """

    def __init__(self, input_sample_dim: int, input_feature_dim: int, output_dim: int, **kwargs):
        super(FlatSensor, self).__init__()
        self.out_channels = output_dim
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, input_feature_dim), padding=0, stride=1)
        self.mlp = nn.Linear(in_features=input_sample_dim, out_features=output_dim)

    def forward(self, x: torch.Tensor):
        x = self.conv(x.unsqueeze(-3)).squeeze(-3).squeeze(-1)  # 单通道卷积，需调整张量维度
        return F.relu(self.mlp(F.relu(x)))


class ConvolutionSensor(nn.Module):
    """
    环境感知块（多通道卷积）

    对于单批量，输入为二维张量（环境观测），输出为一维张量（环境特征）
    用于流场线物理量的特征提取
    """

    def __init__(self, input_sample_dim: int, input_feature_dim: int, output_dim: int, **kwargs):
        super(ConvolutionSensor, self).__init__()
        self.out_channels = output_dim
        self.conv = nn.Conv1d(in_channels=input_feature_dim, out_channels=input_feature_dim,
                              kernel_size=7, padding=3, stride=1)
        self.mlp = MLPReduce2D(input_sample_dim, input_feature_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = self.conv(x.swapaxes(-2, -1)).swapaxes(-2, -1)  # 一维卷积，需调整维度顺序
        return F.tanh(self.mlp(F.relu(x)))


class SensorDecoder(nn.Module):
    """
    与环境感知块匹配的解码器

    用于降噪自编码器，对感知块进行预训练
    """

    def __init__(self, input_dim: int, output_sample_dim: int, output_feature_dim: int, hidden_n: int = 0):
        super(SensorDecoder, self).__init__()
        self.mlp = nn.ModuleList([MLP(input_channels=input_dim, output_channels=output_sample_dim,
                                      num_channels=output_sample_dim, hidden_layer_n=hidden_n,
                                      act_func=Swish)  # 使用连续的激活函数
                                  for _ in range(output_feature_dim)])

    def forward(self, x: torch.Tensor):
        x = [m(x) for m in self.mlp]
        return torch.stack(x, dim=-1)


def train_dae(states: List[np.ndarray], feature_dim: int = 32, sample_dim: int = None, net_type: int = None,
              lr: float = 1e-3, num_epochs: int = 400, batch_size: int = 4, weight_delay: float = 0.00005,
              noise_p: float = 0.2, device: str = 'cuda', dtype: torch.dtype = torch.float64,
              net_args: dict = None, return_feature: bool = False):
    """使用降噪自编码器训练感知器"""
    net_args = {} if net_args is None else net_args
    feature_n, _ = states[0].shape
    sample_n = np.max([state.shape[1] for state in states])
    # 逻辑判断
    if sample_dim is None and sample_dim != sample_n:
        is_state_mapping = True
        is_state_pad = False
    else:
        sample_n = sample_dim
        if len({state.shape for state in states}) == 1:
            is_state_mapping = False
            is_state_pad = False
        else:
            is_state_mapping = True
            if net_type == 1:
                is_state_pad = True
            else:
                is_state_pad = False
    # 对编码器的架构作了改进
    # encoder = [AttentionSensor(sample_n, feature_n, feature_dim, query_channels=net_args.get('query_channels', 2)),
    #            RecursiveSensor(sample_n, feature_n, feature_dim, num_layers=net_args.get('num_layers', 1)),
    #            FlatSensor(sample_n, feature_n, feature_dim)]
    encoder = [ConvolutionSensor(sample_n, feature_n, feature_dim),
               RecursiveSensor(sample_n, feature_n, feature_dim,
                               hidden_dim=net_args.get('hidden_dim', 32),
                               num_layers=net_args.get('num_layers', 1)),
               MultiAttentionSensor(sample_n, feature_n, feature_dim,
                                    head_dim=net_args.get('head_dim', 2),
                                    head_n=net_args.get('head_n', feature_n))]
    if net_type is not None:
        encoder = encoder[net_type: net_type + 1]  # 从可选的三种网络架构里选择一种进行训练
    decoder = SensorDecoder(feature_dim, sample_n, feature_n, hidden_n=net_args.get('hidden_n', 0))

    print(time.ctime())
    print("training on", device, "...")
    start_time = time.time()

    # 网络初始化
    encoder_n = len(encoder)

    device = torch.device(device)
    for net in encoder:
        net_arch(net, (1, sample_n, feature_n))
        net.apply(init_weights)
        net.to(device=device, dtype=dtype)
    net_arch(decoder, (1, feature_dim))
    decoder.apply(init_weights)
    decoder.to(device=device, dtype=dtype)

    # 创建训练集
    if is_state_mapping:
        # 对监督数据进行插值，使网络的输出维度符合要求
        states_interp = []
        for state in states:
            x_new = np.linspace(state[0, 0], state[0, -1], sample_dim)
            state_new = np.vstack([x_new,
                                   interp1d(state[0], state[1], kind='quadratic')(x_new),
                                   interp1d(state[0], state[2], kind='quadratic')(x_new),
                                   interp1d(state[0], state[3], kind='quadratic')(x_new)])
            states_interp.append(state_new.T)
        states_interp = torch.tensor(np.stack(states_interp, axis=0), dtype=dtype)
    else:
        states_interp = torch.tensor(np.stack([state.T for state in states], axis=0), dtype=dtype)
    if is_state_pad:
        # 对RecursiveSensor架构下的变长数据集做补零操作
        states_pad = pad_sequence([torch.tensor(state.T, dtype=dtype) for state in states],
                                  batch_first=True).to(dtype=dtype)
    else:
        states_pad = states_interp
    lengths = torch.tensor([state.shape[1] for state in states], dtype=dtype)
    data_set = TensorDataset(states_pad, lengths, states_interp)
    train_iter = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    encoder_params = [{'params': net.parameters()} for net in encoder]
    decoder_params = [{'params': decoder.parameters()}]
    optimizer = torch.optim.SGD([*encoder_params, *decoder_params],
                                lr=lr, momentum=0.9, weight_decay=weight_delay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.5)
    loss_f = nn.MSELoss()

    animator = PlotAniNet("Net training with SGD", xlabel="epoch", ylabel="metric",
                          line_count=encoder_n + 1,
                          legend=["Avg."] + [net.__class__.__name__ for net in encoder])
    timer = Timer()
    for epoch in range(num_epochs):
        n = 0
        single_loss = np.zeros(encoder_n)
        for i, (X, L, Y) in enumerate(train_iter):
            timer.start()
            # 数据传输
            Y = Y.to(device, non_blocking=True)
            # 添加噪声
            mask = torch.bernoulli(torch.full(X.shape, noise_p))  # 可能的耗时操作
            X_noise = X.clone()
            X_noise[mask == 1] = 0.
            # 数据传输
            X = X.to(device, non_blocking=True)
            X_noise = X_noise.to(device, non_blocking=True)
            # 重置优化器梯度
            optimizer.zero_grad()
            # 前向计算
            loss_list = []
            for net in encoder:
                if is_state_pad:
                    feature = net(X_noise, L)
                else:
                    feature = net(X_noise)
                X_recons = decoder(feature)
                loss_list.append(loss_f(X_recons, Y))
            loss = sum(loss_list) / encoder_n  # L1范数（收敛速度优先）
            # loss = torch.zeros((), device=device)  # L2范数（拟合能力优先）
            # for loss_i in loss_list:
            #     loss += loss_i**2 / encoder_n
            loss.backward()  # retain_graph=True
            optimizer.step()
            with torch.no_grad():
                single_loss += np.array(list(map(float, loss_list))) * X.shape[0]
                n += X.shape[0]
            timer.stop()
        single_loss /= n
        epoch_loss = single_loss.mean()
        speed = n / timer.sum()
        single_loss = list(map(lambda x: round(x, 5), single_loss))
        print(f"Epoch {epoch + 1:d}: average loss {epoch_loss:.3f},",
              f"single loss {str(single_loss):s} ({speed:.1f} examples/sec)")
        scheduler.step()
        animator.add(epoch + 1, epoch_loss, *single_loss)
        animator.update()
        timer.reset()
    print("Total training time: %.1f h" % ((time.time() - start_time) / 3600))
    print("Last learning rate: %.2e" % scheduler.get_last_lr()[0])
    loss_history = np.vstack([np.array(animator.xdata[0]),
                              np.array(animator.ydata)]).T
    if return_feature:
        with torch.no_grad():
            feature_list = []
            for net in encoder:
                if is_state_pad:
                    feature = net(states_pad.to(device), lengths.to(device))
                else:
                    feature = net(states_pad.to(device))
                feature_list.append(feature.cpu().numpy())
        return encoder, decoder, loss_history, (states_pad, feature_list, states_interp)
    else:
        return encoder, decoder, loss_history


def test_sensor():
    """消融实验，对比不同感知器的性能"""
    sensor_path = r'F:\Nozzle\OpenFOAM\pretrain_sensor.pth'
    loss_path = r'F:\Nozzle\OpenFOAM\pretrain_dae_loss.xlsx'
    states = torch.load(r'F:\Nozzle\OpenFOAM\pretrain_data.pth') + \
             torch.load(r'F:\Nozzle\OpenFOAM\pretrain_data_all.pth')
    feature_n, sample_n = states[0].shape

    # 测试几种感知器的架构
    t1 = torch.tensor(states[0].T).unsqueeze(0)
    t2 = torch.stack([torch.tensor(states[0].T), torch.tensor(states[1].T)], dim=0)
    # 旧架构
    s1 = AttentionSensor(sample_n, feature_n, output_dim=32)
    s2 = RecursiveSensor(sample_n, feature_n, output_dim=32)
    s3 = FlatSensor(sample_n, feature_n, output_dim=32)
    # 新架构
    s4 = MultiAttentionSensor(sample_n, feature_n, output_dim=32)
    s5 = MultiAttentionSensor(sample_n, feature_n, output_dim=32, head_dim=2, head_n=4)
    s6 = ConvolutionSensor(sample_n, feature_n, output_dim=32)
    s7 = RecursiveSensor(sample_n, feature_n, hidden_dim=32, output_dim=32)
    for s in [s1, s2, s3, s4, s5, s6, s7]:
        for t in [t1, t2]:
            print(s(t).shape)

    # 对比不同架构和特征维度
    # with pd.ExcelWriter(loss_path, mode='a', if_sheet_exists='new') as writer:
    #     for feature_dim in [1, 16, 32, 64, 128]:
    #         encoder, decoder, loss = \
    #             train_dae(states, feature_dim=feature_dim, lr=5e-1, num_epochs=200,
    #                       batch_size=2, noise_p=0.1, weight_delay=1e-5)
    #         # loss数据保存至本地文件
    #         tags = ["Epoch", "Avg."] + [net.__class__.__name__ for net in encoder]
    #         df = pd.DataFrame(loss, columns=tags)
    #         df.to_excel(writer, sheet_name=f"feat. {feature_dim:d}")
    #         # 绘制loss曲线
    #         fig, ax = plt.subplots()
    #         for i, tag in zip(range(1, loss.shape[1]),
    #                           tags[1:]):
    #             ax.plot(loss[:, 0], loss[:, i], '-', label=f"${tag}$")
    #         ax.set_yscale('log')
    #         ax.set_yticks([10 ** (- i) for i in range(4)])
    #         ax.set_xlabel("$Epoch$", fontsize=20)
    #         ax.set_ylabel("$Loss$", fontsize=20)
    #         ax.legend(loc='upper right', fontsize=18)
    #         ax.grid()
    #         plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示指数坐标系标签
    #         plt.show()

    # 对比不同架构（特征维度固定为16）
    encoder_list = []
    loss_list = []
    for i in range(3):
        result = train_dae(states, feature_dim=16, net_type=i, lr=5e-1, num_epochs=200,
                           batch_size=4, noise_p=0.1, weight_delay=1e-5)
        encoder_list.append(result[0][0])
        loss_list.append(result[2][:, 2])
    # 保存编码器网络
    torch.save(encoder_list, sensor_path)
    # loss数据保存至本地文件
    loss = np.array([np.arange(len(loss_list[0]))] + loss_list).T
    tags = ["epoch", "attn", "lstm", "cnn"]
    with pd.ExcelWriter(loss_path, mode='a', if_sheet_exists='new') as writer:
        df = pd.DataFrame(loss, columns=tags)
        df.to_excel(writer, sheet_name=f"feat. 16 (final)")
    # 绘制loss曲线
    fig, ax = plt.subplots()
    for i, tag in zip(range(1, loss.shape[1]),
                      tags[1:]):
        ax.plot(loss[:, 0], loss[:, i], '-', label=f"${tag.upper()}$")
    ax.set_yscale('log')
    ax.set_yticks([10 ** (- i) for i in range(4)])
    ax.set_xlabel("$Epoch$", fontsize=20)
    ax.set_ylabel("$Loss$", fontsize=20)
    ax.legend(loc='upper right', fontsize=18)
    ax.grid()
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示指数坐标系标签
    plt.show()

    # 测试ResursiveSensor架构时train_dae对变长数据集的支持
    # length = torch.tensor([10, 4, 8])
    # _states = [state[:, :size] for size, state in zip(length, states)]
    # t = pad_sequence([torch.tensor(state.T) for state in _states], batch_first=True)
    # s = RecursiveSensor(sample_n, feature_n, output_dim=16)
    # print(s(t, length).shape)
    # result = train_dae(_states, feature_dim=16, net_type=1, lr=5e-1, num_epochs=200,
    #                    batch_size=2, noise_p=0.1, weight_delay=1e-5)


def test_sensor_2(sample_n=150):
    """使用更大的数据集（非标准预训练数据集）单纯做架构测试"""
    states = []
    for item in torch.load(r'F:\Nozzle\OpenFOAM\nozzle_data_selected.pth'):  # 非均匀非等长数据
        state = item[2]
        x_new = np.linspace(state[0, 0], state[0, -1], sample_n)
        state = np.vstack([x_new,
                           interp1d(state[0], state[1], kind='quadratic')(x_new),
                           interp1d(state[0], state[2], kind='quadratic')(x_new),
                           interp1d(state[0], state[3], kind='quadratic')(x_new)])
        states.append(state)
    print("Data shape: %s" % str((len(states), *states[0].shape)))

    # 对比不同架构和特征维度
    with pd.ExcelWriter(r'F:\Nozzle\OpenFOAM\pretrain_dae_loss.xlsx', mode='a', if_sheet_exists='new') as writer:
        for feature_dim in [1, 16, 32, 64]:
            loss_list = []
            encoder_list = []
            for net_i in [0, 1, 2]:
                encoder, decoder, loss = \
                    train_dae(states, net_type=net_i, feature_dim=feature_dim, lr=5e-1, num_epochs=400,
                              batch_size=64, noise_p=0.1, weight_delay=1e-5, device='cuda',
                              net_args={})  # 小批量相比大批量能收敛到更优解，且GPU占用更多CPU占用更少（对比8和64）
                loss_list.append(loss)
                encoder_list.extend(encoder)
            loss = np.vstack([loss[:, 2] for loss in loss_list])
            loss = np.vstack([loss_list[0][:, 0],
                              loss.mean(axis=0),
                              loss]).T
            # loss数据保存至本地文件
            tags = ["Epoch", "Avg."] + [net.__class__.__name__ for net in encoder_list]
            df = pd.DataFrame(loss, columns=tags)
            df.to_excel(writer, sheet_name=f"feat. {feature_dim:d}")
            # 绘制loss曲线
            fig, ax = plt.subplots()
            for i, tag in zip(range(1, loss.shape[1]),
                              tags[1:]):
                ax.plot(loss[:, 0], loss[:, i], '-', label=f"${tag}$")
            ax.set_yscale('log')
            ax.set_yticks([10 ** (- i) for i in range(4)])
            ax.set_xlabel("$Epoch$", fontsize=20)
            ax.set_ylabel("$Loss$", fontsize=20)
            ax.legend(loc='upper right', fontsize=18)
            ax.grid()
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示指数坐标系标签
            plt.show()


class Actor(nn.Module):
    """
    演员网络

    对于单批量，输入二维张量，输出多元高斯分布参数
    同评论员网络共享底层感知块
    """

    LOG_STD_MIN = -5
    LOG_STD_MAX = 1

    def __init__(self, sensor: nn.Module, output_dim: int, hidden_dim: int = 128, hidden_n: int = 3):
        super(Actor, self).__init__()
        self.sensor = sensor
        self.mlp = MLP(input_channels=sensor.out_channels, num_channels=hidden_dim,
                       output_channels=hidden_dim, hidden_layer_n=hidden_n)
        # self.mlp = DenoiseNet(sensor.out_channels, hidden_dim, hidden_dim, block_n=hidden_n)
        # 使用MLP的输出计算高斯分布的均值和标准差，但实际上标准差也可以作为独立参数，即nn.Parameter(torch.zeros(1, action_dim))
        self.mu_layer = nn.Linear(hidden_dim, output_dim)
        self.log_std_layer = nn.Linear(hidden_dim, output_dim)

        # self.mlp.apply(init_weights)
        # 均值层和方差层的权重采用较小初始化
        # nn.init.uniform_(self.mu_layer.weight, -0.1, 0.1)
        # nn.init.uniform_(self.mu_layer.bias, -0.1, 0.1)
        nn.init.uniform_(self.log_std_layer.weight, -0.01, 0.01)
        nn.init.uniform_(self.log_std_layer.bias, -2, 0)

    def forward(self, x: torch.Tensor):
        x = self.sensor(x)
        h = self.mlp(x)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return torch.stack([mu, torch.exp(log_std)], dim=-1)


class Critic(nn.Module):
    """
    评论员网络

    对于单批量，输入二维张量，输出为零维张量
    同演员网络共享底层感知块
    """

    def __init__(self, sensor: nn.Module, hidden_dim: int = 128, hidden_n: int = 3):
        super(Critic, self).__init__()
        self.sensor = sensor
        self.mlp = MLP(input_channels=sensor.out_channels, num_channels=hidden_dim,
                       output_channels=1, hidden_layer_n=hidden_n)
        # self.mlp = DenoiseNet(sensor.out_channels, hidden_dim, 1, block_n=hidden_n)

        # self.mlp.apply(init_weights)

    def forward(self, x: torch.Tensor):
        x = self.sensor(x)
        return self.mlp(x).squeeze(-1)


class PPO2:

    def __init__(self, cfg: AgentConfig):
        self.gamma = cfg.gamma
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype
        # self.sensor = RecursiveSensor(  # LSTM支持变长度state具有最佳效果
        #     input_sample_dim=cfg.observation_sample_n,
        #     input_feature_dim=cfg.observation_feature_n,
        #     output_dim=cfg.sensor_hidden_dim
        self.sensor = ConvolutionSensor(  # Convolve具有最佳效果
            input_sample_dim=cfg.observation_sample_n,
            input_feature_dim=cfg.observation_feature_n,
            output_dim=cfg.sensor_hidden_dim
        ).to(device=self.device, dtype=self.dtype)
        self.actor = Actor(
            sensor=self.sensor,
            output_dim=cfg.action_n,
            hidden_dim=cfg.actor_hidden_dim,
            hidden_n=cfg.actor_hidden_n - 1
        ).to(device=self.device, dtype=self.dtype)
        self.critic = Critic(
            sensor=self.sensor,
            hidden_dim=cfg.critic_hidden_dim,
            hidden_n=cfg.critic_hidden_n - 1
        ).to(device=self.device, dtype=self.dtype)
        # self.actor_optimizer = torch.optim.Adam([{'params': self.actor.mlp.parameters(), 'lr': cfg.actor_lr},
        #                                          {'params': self.sensor.parameters(), 'lr': cfg.sensor_lr}])
        # self.critic_optimizer = torch.optim.Adam([{'params': self.critic.mlp.parameters(), 'lr': cfg.critic_lr},
        #                                           {'params': self.sensor.parameters(), 'lr': cfg.sensor_lr}])
        self.actor_optimizer = torch.optim.SGD([{'params': self.actor.mlp.parameters(), 'lr': cfg.actor_lr},
                                                {'params': self.sensor.parameters(), 'lr': cfg.sensor_lr}],
                                               momentum=0.9, weight_decay=1e-7)
        self.critic_optimizer = torch.optim.SGD([{'params': self.critic.mlp.parameters(), 'lr': cfg.critic_lr},
                                                 {'params': self.sensor.parameters(), 'lr': cfg.sensor_lr}],
                                                momentum=0.9, weight_decay=1e-7)
        self.memory = ReplayBuffer()
        self.k_epochs = cfg.k_epochs  # update policy for K epochs
        self.batch_size = cfg.batch_size
        self.eps_clip = cfg.eps_clip  # clip parameter for PPO
        self.entropy_coef = cfg.entropy_coef  # entropy coefficient
        self.action_dim = cfg.action_n
        self.sample_count = 0
        self.current_log_probs = .0
        self.update_freq = cfg.update_freq

    def action_dist(self, state):
        """使用演员网络构建有界概率分布"""
        if isinstance(state, torch.Tensor):
            state = state.clone().detach().to(device=self.device, dtype=self.dtype)
        else:
            state = torch.tensor(state, device=self.device, dtype=self.dtype)
        if len(state.shape) == 2:
            state = state.unsqueeze(dim=0)  # 添加批量维度
        probs = self.actor(state)
        dist = Normal(probs[..., 0], probs[..., 1])  # Normal(mu, sigma)
        return dist

    @torch.no_grad()
    def sample_action(self, state):
        """动作采样
        输入: (batch, sample_n, feature_n) or (sample_n, feature_n)
        输出: (batch, action_n)"""
        self.sample_count += 1
        dist = self.action_dist(state)
        # 重参数化采样得到无界动作
        action = dist.rsample()
        # 应用tanh变换
        squashed_action = torch.tanh(action)
        # 计算对数概率（高斯分布+雅可比修正），并对动作维度求和
        gaussian_log_prob = dist.log_prob(action)
        log_det_jacobian = (2 * (math.log(2) - action - F.softplus(-2 * action)))
        log_prob = gaussian_log_prob.sum(dim=-1) - log_det_jacobian.sum(dim=-1)
        self.current_log_probs = log_prob.detach().cpu().numpy()
        return squashed_action.detach().cpu().numpy()

    def logprob_of_action(self, dist, squashed_action):
        """在指定分布上计算重映射动作的对数概率"""
        def atanh(x):
            # 数值稳定版 atanh
            return 0.5 * (torch.log1p(x + 1e-8) - torch.log1p(-x + 1e-8))

        action = atanh(squashed_action.clamp(-0.999, 0.999))
        gaussian_log_prob = dist.log_prob(action)
        log_det_jacobian = torch.log(1 - squashed_action**2 + 1e-8)
        log_prob = gaussian_log_prob.sum(dim=-1) - log_det_jacobian.sum(dim=-1)
        return log_prob

    @torch.no_grad()
    def predict_action(self, state):
        """动作预测
        输入: (batch, sample_n, feature_n) or (sample_n, feature_n)
        输出: (batch, action_n)"""
        dist = self.action_dist(state)
        action = torch.tanh(dist.mean)
        return action.detach().cpu().numpy()

    def update(self):
        """更新智能体（应当和sample_action同等频率调用）"""
        # update policy every n steps
        if len(self.memory) < self.update_freq:  # self.sample_count % self.update_freq != 0
            return
        else:
            print(f"Update policy... (memory size: {len(self.memory):d}, inner epoch: {self.k_epochs:d})")
        old_states, old_actions, old_log_probs, old_rewards, old_dones = self.memory.sample_all()
        # convert to tensor
        old_states = torch.tensor(np.array(old_states), device=self.device, dtype=self.dtype)
        old_actions = torch.tensor(np.array(old_actions), device=self.device, dtype=self.dtype)
        old_log_probs = torch.tensor(np.array(old_log_probs), device=self.device, dtype=self.dtype)
        # monte carlo estimate of state rewards
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(old_rewards), reversed(old_dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        # Normalizing the rewards
        returns = torch.tensor(returns, device=self.device, dtype=self.dtype)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)  # 1e-5 to avoid division by zero
        # create training batch
        data_loader = DataLoader(TensorDataset(old_states, old_actions, old_log_probs, returns),
                                 batch_size=self.batch_size, shuffle=True)
        for _ in range(self.k_epochs):
            for states_i, actions_i, log_probs_i, returns_i in data_loader:
                # compute advantage
                values = self.critic(states_i)
                advantage = returns_i - values.detach()  # detach to avoid backprop through the critic
                # get action probabilities
                dist = self.action_dist(states_i)
                # get new action probabilities
                new_probs = self.logprob_of_action(dist, actions_i)  # Size: (batch, )
                # compute ratio (pi_theta / pi_theta_old):
                ratio = torch.exp(new_probs - log_probs_i)  # old_log_probs must be detached
                # compute surrogate loss
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                # compute entropy approximately
                entropy = dist.entropy().mean()  # 直接使用高斯熵（去掉经验常数项：- 0.174 * self.action_dim）
                # compute actor loss
                actor_loss = - torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                # compute critic loss
                critic_loss = (returns_i - values).pow(2).mean()
                # take gradient step
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                # torch.nn.utils.clip_grad_value_(self.actor.mlp.parameters(), clip_value=1.0)
                # torch.nn.utils.clip_grad_value_(self.critic.mlp.parameters(), clip_value=1.0)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()

    def push(self, state, action, reward, next_state, done):
        """将当前时间步的快照存入智能体记忆池"""
        self.memory.push((state, action, self.current_log_probs, reward, done))

    def dump(self):
        """保存智能体网络参数为状态字典"""
        state_dict = {'sensor': self.sensor.state_dict(),
                      'actor': self.actor.state_dict(),
                      'critic': self.critic.state_dict()}
        return state_dict

    def load(self, state_dict, strict=True):
        """从状态字典读取智能体网络参数"""
        self.sensor.load_state_dict(state_dict['sensor'], strict=strict)
        self.actor.load_state_dict(state_dict['actor'], strict=strict)
        self.critic.load_state_dict(state_dict['critic'], strict=strict)


def pretrain_sensor(data_dir: str, agent: PPO2):
    """使用喷管前置数据集预训练感知器网络（递归架构）"""
    sensor_path = os.path.join(data_dir, 'pretrain_sensor.pth')
    if os.path.exists(sensor_path):
        sensor = torch.load(sensor_path)[0]
        print("Load pretrained sensor from '%s'" % sensor_path)
        assert agent.sensor.rnn.input_size == sensor.rnn.input_size and \
               agent.sensor.rnn.hidden_size == sensor.rnn.hidden_size, \
               "Net size of agent and local file mismatch"
        agent.sensor = sensor.to(agent.device)
    else:
        states = []
        for f in os.listdir(data_dir):
            if f.startswith('pretrain_data') and f.endswith('.pth'):
                states += torch.load(os.path.join(data_dir, f))['state']  # 使用已有的全部数据进行预训练
        assert states[0].shape[0] == agent.sensor.rnn.input_size, \
               "Net size of agent and training data mismatch"
        encoder, decoder, loss = \
            train_dae(states, feature_dim=agent.sensor.rnn.hidden_size, net_type=1,  # RecursiveSensor
                      lr=5e-1, num_epochs=200, batch_size=4, noise_p=0.1, weight_delay=1e-5)
        torch.save(encoder, sensor_path)
        print("Save pretrained sensor to '%s'" % sensor_path)
        agent.sensor = encoder[0].to(agent.device)


def pretrain_sensor_from_surr(surr_path: str, agent: PPO2):
    """从喷管代理模型中加载感知器网络（卷积架构）"""
    sensor = torch.load(surr_path)['encoder']
    print("Load pretrained sensor from '%s'" % surr_path)
    assert agent.sensor.conv.in_channels == sensor.conv.in_channels and \
           agent.sensor.mlp.mlp_out.in_features == sensor.mlp.mlp_out.in_features, \
           "Net size of agent and local file mismatch"
    agent.sensor = sensor.to(agent.device)


if __name__ == '__main__':
    data_path = r'F:/Nozzle/OpenFOAM/'  # r'/home/zhuofeng/lgq/OpenFOAM/test/FluentWithDL/'

    # 测试感知器网络
    # test_sensor()
    test_sensor_2()

    # 测试演员和评论员网络
    # states = torch.load(data_path + 'pretrain_data.pth')
    # x = torch.tensor(np.swapaxes(np.array(states[: 10]), 1, 2))
    # s = RecursiveSensor(input_sample_dim=states[0].shape[1], input_feature_dim=states[0].shape[0], output_dim=16)
    # print(f"Sensor: {x.shape} --> {s(x).shape}")
    # a = Actor(s, output_dim=7)
    # print(f"Actor: {x.shape} --> {a(x).shape}")
    # c = Critic(s)
    # print(f"Critic: {x.shape} --> {c(x).shape}")

    # 测试智能体
    # agent_cfg = AgentConfig(
    #     observation_sample_n=151,
    #     observation_feature_n=4,
    #     action_n=7,
    #     update_freq=2
    # )
    # agent = PPO2(agent_cfg)
    # for i in range(2):
    #     action = agent.sample_action(x[i])[0]
    #     agent.push(x[i].numpy(), action, i, None, False)
    # agent.update()


