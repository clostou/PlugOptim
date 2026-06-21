import copy
import os
import sys
import traceback
import math
import time
from copy import deepcopy
from collections import deque, Counter
from pathlib import Path
import pyDOE2
from PIL import Image
from io import BytesIO
from typing import Union, Tuple, Optional, Sequence

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.linalg import eigh, svd
from scipy.optimize import minimize
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torch.multiprocessing import set_start_method
import gym

from thop import profile

sys.path.append('/home/zhuofeng/lgq/python/')

from plugDesign import External, ExternalSpine, profile_to_msh
from bellDesign import CharacteristicsNozzle
from agent import AgentConfig, train_dae
from ML.regress import RLS
from ML.reduce import PCA
from ML.post import PlotAniNet
from cfd_toolbox.submit import *
from cfd_toolbox.utils import *
from cfd_toolbox.gasdy import *
from cfd_toolbox.plot import *
from denoise import DenoisePhysInStackWrapper, train_net, Plotter, save_net, load_net, DenoiseWrapper


class DataStruct:
    """
    动态的数据结构类
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class NozzleConfig:

    def __init__(self, **kwargs):
        self.script_path = r'./'  # Fluent脚本路径
        self.work_path = r'./'  # 工作路径
        self.fluent_path = 'fluent'  # fluent求解器路径
        self.thread_n = 8  # 求解进程数
        self.case_dir = None  # 算例目录，指定时则尝试读取已有文件

        if os.name == 'nt':
            self.work_path = r'F:/Nozzle/OpenFOAM'
            self.fluent_path = r'E:/Ansys/2022R1/v221/fluent/ntbin/win64/fluent.exe'
            self.thread_n = 4
            self.is_linux = False
        elif os.name == 'posix':
            self.work_path = r'/home/zhuofeng/lgq/OpenFOAM/test/Fluent/'
            self.fluent_path = r'/public/software/ansys_inc/v221/fluent/bin/fluent'
            self.thread_n = 16
            self.is_linux = True
        else:
            print("Unknown OS when auto-configure")
            pass

        self.jet_type = 'bell'  # 喷管类型（钟形喷管bell/塞式喷管plug/样条塞式喷管plug-sp）
        self.r_t = 0.2  # 等效喉道半径
        self.epsilon = 16  # 喷管扩张比
        self.mesh_n = 20  # 网格划分数
        self.throat_theta = 40  # 钟形喷管喉道角
        self.throat_rho = 10 * self.r_t  # 钟形喷管喉道曲率半径
        self.factor = 3  # 网格偏置系数
        self.spline_p = [1, 0.3, 0.5, 0.2, 0.1]  # 样条塞式喷管插值点

        self.cfd_params = {  # 仿真材料参数和边界条件
            'inlet_p': [3e6, 6e6, 12e6, 20e6],
            'atmo_p': [p + 325 for p in [101e3, 60e3, 36e3, 20e3, 8e3, 0]],
            'inlet_t': 3500,
            'Cp': 2837.76,  # 1006.43
            'K': 0.242,  # 0.0242
            'M': 20.9e-3,  # 28.966e-3
        }
        self.render = True  # 绘制喷管构型
        self.verbose = False  # 信息打印切换（主要是网格生成的输出）

        self.update(**kwargs)

    def type_assign(self, jet_type):
        self.jet_type = jet_type
        if self.jet_type == 'bell':
            self.mesh_n = 20
            self.throat_theta = 40
            self.throat_rho = 10 * self.r_t
            self.factor = 3
        elif self.jet_type == 'plug':
            self.mesh_n = 150
            self.factor = 6
        elif self.jet_type == 'plug-sp':
            self.mesh_n = 350
            self.factor = 6
            self.spline_p = [1, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2]
        else:
            print("Unknown type of nozzle when auto-configure")

    def update(self, **kwargs):
        if 'jet_type' in kwargs.keys():
            self.type_assign(kwargs.pop('jet_type'))
        if 'cfd_params' in kwargs.keys():
            self.cfd_params.update(kwargs.pop('cfd_params'))
        self.__dict__.update(kwargs)


class NozzleCFD:
    """
    定义了喷管仿真的特定问题，与fluent脚本配合使用
    """

    def __init__(self, config: NozzleConfig):
        self.thread = config.thread_n
        self.plot = config.render

        self.base_path = ''
        self.r_t = 0.
        self.epsilon = 0.
        self.spline_p = []
        self.mesh_n = 0.
        self.params = {}

        self.A_inlet = 0.
        self.model: Optional[CharacteristicsNozzle, External, ExternalSpine] = None
        self.task: Optional[FluentQuest] = None

        self.data: Optional[pd.DataFrame] = None
        self.data_field: Optional[np.ndarray] = None

        if config.case_dir is None:  # 选择加载方式：建模计算/读取结果
            self._init_new_task(config)
        else:
            self._init_load_file(config)

    def _init_new_task(self, config: NozzleConfig):
        """初始化：创建新的喷管计算任务"""
        self.r_t = config.r_t
        self.epsilon = config.epsilon
        self.mesh_n = config.mesh_n
        self.params = config.cfd_params
        params_extra = thermo(self.params['Cp'], self.params['M'])
        self.params.update(params_extra)
        # 根据喷管类型分别处理
        if config.jet_type == 'bell':
            self.base_path = os.path.join(config.work_path,
                                          f'bell_Rt{self.r_t:.1e}_eps{self.epsilon:.1f}_n{self.mesh_n:d}')
            os.makedirs(self.base_path, exist_ok=True)
            jou_path = os.path.join(self.base_path, 'bell.jou')
            copy_file(jou_path, os.path.join(config.script_path, 'bell.jou'))
            with open(os.path.join(self.base_path, 'config.txt'), 'w', encoding='utf-8') as f:  # 写出喷管几何参数
                f.write(f'{self.r_t:.6e}\n{self.epsilon:.6e}\n')
            # 计算喷管构型
            bell = CharacteristicsNozzle(r_t=self.r_t, rho_t=config.throat_rho,
                                         axial_sym=True, gamma=self.params['gamma'])
            bell.derive(epsilon=self.epsilon, throat_theta=config.throat_theta)
            if self.plot:
                bell.plot_field()
            # 模型生成及网格划分
            bell.generate(size=self.r_t / self.mesh_n, factor=config.factor)
            if self.plot:
                bell.plot_profile()
            profile, tag = bell.get_profile()
            profile_to_msh(profile, tag, lc=0., planner=True, plot=self.plot, verbose=config.verbose,
                           save_path=os.path.join(self.base_path, 'bell.bdf'))
            self.A_inlet = np.pi * profile[tag['inlet'][0]][1] ** 2
            # 创建Fluent任务
            self.model = bell
            self.task = FluentQuest(config.fluent_path, os.path.abspath(jou_path),
                                    planar_geom=True, thread_n=self.thread)
        elif config.jet_type == 'plug':
            self.base_path = os.path.join(config.work_path,
                                          f'plug_Rt{self.r_t:.2e}_eps{self.epsilon:.1f}_n{self.mesh_n:d}')
            os.makedirs(self.base_path, exist_ok=True)
            jou_path = os.path.join(self.base_path, 'plug.jou')
            copy_file(jou_path, os.path.join(config.script_path, 'plug.jou'))
            with open(os.path.join(self.base_path, 'config.txt'), 'w', encoding='utf-8') as f:  # 写出喷管几何参数
                f.write(f'{self.r_t:.6e}\n{self.epsilon:.6e}\n')
            # 计算喷管构型
            plug = External(epsilon=self.epsilon, r_t=self.r_t, gamma=self.params['gamma'])
            plug.derive()
            # 模型生成及网格划分
            plug.generate(n=self.mesh_n, factor=config.factor)
            if self.plot:
                plug.plot()
            profile, tag = plug.get_profile()
            profile_to_msh(profile, tag, lc=0., planner=True, plot=self.plot, verbose=config.verbose,
                           save_path=os.path.join(self.base_path, 'plug.bdf'))
            self.A_inlet = np.pi * (profile[tag['inlet'][0]][1] ** 2 - profile[tag['inlet'][-1]][1] ** 2)
            # 创建Fluent任务
            self.model = plug
            self.task = FluentQuest(config.fluent_path, os.path.abspath(jou_path),
                                    planar_geom=True, thread_n=self.thread)
        elif config.jet_type == 'plug-sp':
            self.spline_p = config.spline_p
            self.base_path = os.path.join(config.work_path,
                                          f'plug-sp_Rt{self.r_t:.2e}_eps{self.epsilon:.1f}_n{self.mesh_n:d}_' +
                                          ''.join(map(lambda x: str(int(x * 1e4)), self.spline_p)))  # 插值参数保留四位有效数字
            os.makedirs(self.base_path, exist_ok=True)
            jou_path = os.path.join(self.base_path, 'plug.jou')
            copy_file(jou_path, os.path.join(config.script_path, 'plug.jou'))
            with open(os.path.join(self.base_path, 'config.txt'), 'w', encoding='utf-8') as f:  # 写出喷管几何参数
                f.write(f'{self.r_t:.6e}\n{self.epsilon:.6e}\n')
                f.write(' '.join(map(lambda x: str(round(x, 6)), self.spline_p)) + '\n')
            # 计算喷管构型
            plug = ExternalSpine(epsilon=self.epsilon, r_t=self.r_t, gamma=self.params['gamma'])
            # 模型生成及网格划分
            plug.generate(*self.spline_p, n=self.mesh_n, factor=config.factor)
            if self.plot:
                plug.plot()
            profile, tag = plug.get_profile()
            profile_to_msh(profile, tag, lc=0., planner=True, plot=self.plot, verbose=config.verbose,
                           save_path=os.path.join(self.base_path, 'plug.bdf'))
            self.A_inlet = np.pi * (profile[tag['inlet'][0]][1] ** 2 - profile[tag['inlet'][-1]][1] ** 2)
            # 创建Fluent任务
            self.model = plug
            self.task = FluentQuest(config.fluent_path, os.path.abspath(jou_path),
                                    planar_geom=True, thread_n=self.thread)
        else:
            raise ValueError("Unknown type of nozzle. (Supported type: bell, plug, plug-sp)")

        # 需要求解的参数组，注意Fluent内分子量的单位为kg/kmol，非SI
        self.task.add_params(
            Cp=[self.params['Cp']], K=[self.params['K']], M=[self.params['M'] * 1e3],
            inlet_p=self.params['inlet_p'], inlet_t=[self.params['inlet_t']],
            atmo_p=self.params['atmo_p'], inlet_area=[self.A_inlet])
        print(self.task)

    def _init_load_file(self, config: NozzleConfig):
        """初始化：读取本地已有的结果文件"""
        self.base_path = os.path.join(config.work_path, config.case_dir)
        with open(os.path.join(self.base_path, 'config.txt'), 'r', encoding='utf-8') as f:
            self.r_t = float(f.readline().strip())
            self.epsilon = float(f.readline().strip())
            self.spline_p = list(map(float, f.readline().strip().split()))  # 分隔符支持空格和制表

    def postproc(self):
        # 读取工况和计算结果
        result_file = os.path.join(self.base_path, 'fluent_result.txt')
        if not os.path.exists(result_file):
            if not hasattr(self.task, '_job_dir_list'):
                for _ in self.task:
                    pass
            self.task.get_result('report-def-0-rfile.out')
            self.task.get_xyplot('xy-plot-ycoord.txt')
            self.task.get_xyplot('xy-plot-pressure.txt')
            self.task.get_xyplot('xy-plot-mach.txt')
        # 计算结果处理（不完整算例会引发nan空值或KeyError）
        with open(result_file, 'r', encoding='utf-8') as fr:
            header = fr.readline().strip().split(',')
            data = pd.read_csv(fr, delimiter=' ', index_col=0, names=header)
        gas_prop = thermo(data['Cp'], data['M'] * 1e-3)
        data['Qm_max'] = Qm_max(data['inlet_p'], data['inlet_t'], np.pi * self.r_t ** 2,
                                gas_prop['gamma'], gas_prop['R'])
        data['Cf_max'] = Cf_max(data['inlet_p'], data['atmo_p'], gas_prop['gamma'])
        data['Cf'] = data['report-def-thrust'] / (np.pi * self.r_t ** 2 * data['inlet_p'])
        data['SpecImpulse'] = data['report-def-thrust'] / (9.80665 * data['report-def-massflow'])
        data.to_csv(os.path.join(self.base_path, 'result.csv'))
        self.data = data
        # 读取场变量（不完整算例会引发0空值）
        with open(os.path.join(self.base_path, 'xy-plot-ycoord.txt'), 'r', encoding='utf-8') as fr:
            # arr1 = np.fromfile(fr, dtype=float, sep=' ')  # 这里读出的是一维数组
            # arr1 = arr1.reshape((len(data), 2, -1))
            arr1 = pd.read_csv(fr, delimiter=' ', header=None, skip_blank_lines=False)  # 使用更健壮的读取方式
            arr1 = arr1.to_numpy(na_value=0).reshape((len(data), 2, -1))
        with open(os.path.join(self.base_path, 'xy-plot-pressure.txt'), 'r', encoding='utf-8') as fr:
            arr2 = pd.read_csv(fr, delimiter=' ', header=None, skip_blank_lines=False)
            arr2 = arr2.to_numpy(na_value=0).reshape((len(data), 2, -1))
        with open(os.path.join(self.base_path, 'xy-plot-mach.txt'), 'r', encoding='utf-8') as fr:
            arr3 = pd.read_csv(fr, delimiter=' ', header=None, skip_blank_lines=False)
            arr3 = arr3.to_numpy(na_value=0).reshape((len(data), 2, -1))
        if np.any(arr1[:, 0, :] != arr2[:, 0, :]) or np.any(arr1[:, 0, :] != arr3[:, 0, :]):
            raise ValueError("fluent xy-plot files have different x coordinate. (y-coord/pressure/mach)")
        data_field = np.concatenate([arr1, arr2[:, 1:2, :], arr3[:, 1:2, :]], axis=1)
        x, ind = np.unique(data_field[0, 0, :], return_index=True)  # 去除可能存在的重复节点
        self.data_field = data_field[:, :, ind]
        print(f"{len(data)} samples and {len(x)} points have been read. (point range: [{x[0]:.3e}, {x[-1]:.3e}])")
        return data

    def analyse(self):
        if self.data is None:
            print("Please run `NozzleCFD.postproc()` to collect data first.")
            return
        # NPR为设计工况时Cf的变化
        try:
            my_sheet1 = pd.DataFrame()
            my_sheet1['NPR'] = self.data['inlet_p'] / self.data['atmo_p']
            NPR_tartget = self.model._Ma2NPR(self.model.Ma_e, self.model.gamma)
            my_sheet1['ln(NPR/NPR_target)'] = np.log(my_sheet1['NPR'] / NPR_tartget)
            my_sheet1['Cf/Cf_max'] = self.data['Cf'] / self.data['Cf_max']
            my_sheet1 = my_sheet1.sort_values(by='NPR', ascending=True)
        except Exception as e:
            print("Failed to generate sheet 1:", repr(e), file=sys.stderr)
            my_sheet1 = None
        # 收集塞锥喉道处的物理量
        try:
            n_sample = len(self.data_field)
            ind_throat = np.argmin(self.data_field[0, 0, :] - self.model.profile['plug_div'][0, 0])
            plug_throat = np.concatenate([np.repeat(self.model.profile['plug_div'][0: 1, :], n_sample, axis=0),
                                          self.data_field[:, 1:, ind_throat]], axis=1)
            my_sheet2 = pd.DataFrame(plug_throat, columns=['mesh_x', 'mesh_y', 'y', 'pressure', 'mach'])
        except Exception as e:
            print("Failed to generate sheet 2:", repr(e), file=sys.stderr)
            my_sheet2 = None
        # 收集塞锥壁面的坐标
        try:
            ind_plug1, ind_plug2 = self.model.get_profile()[1]['plug']
            plug = np.concatenate([self.model.points[ind_plug1: ind_plug2 + 1],
                                   self.data_field[0, :2, :].T], axis=1)
            my_sheet3 = pd.DataFrame(plug, columns=['mesh_x', 'mesh_y', 'x', 'y'])
            my_sheet3['delta_y'] = my_sheet3['mesh_y'] - my_sheet3['y']
        except Exception as e:
            print("Failed to generate sheet 3:", repr(e), file=sys.stderr)
            my_sheet3 = None
        return my_sheet1, my_sheet2, my_sheet3

    def calc_cf(self, continuity_limit: float = 10, cf_limit: float = 5, n_net: int = 16, n_int: int = 400,
                read_net: bool = True, read_cf: bool = False):
        net_path = os.path.join(self.base_path, 'net.pth')
        # 尝试直接读取本地已有结果
        if read_cf and os.path.exists(net_path):
            print("Reading results from '%s'" % net_path)
            return torch.load(net_path)['desc']['Cf_int']
        # 数据处理
        if self.data is None:
            print("Please run `NozzleCFD.postproc()` to collect data first.", file=sys.stderr)
            return
        data = self.data[np.logical_and(np.abs(self.data['report-def-continuity']) < continuity_limit,
                                        np.logical_and(self.data['Cf'] > 0.,
                                                       self.data['Cf'] < cf_limit)
                                        )]  # 未收敛（发散）的结果直接丢弃
        data_in = torch.tensor(data[['inlet_p', 'atmo_p', 'report-def-continuity']].to_numpy())
        data_out = torch.tensor(data[['Cf']].to_numpy())
        # 创建拟合网络
        if read_net and os.path.exists(net_path):  # 从本地文件加载
            print("Reading existing nets from '%s'" % net_path)
            nets = load_net(net_path)
        else:  # 拟合cfd计算结果
            nets = train_net(DenoisePhysInStackWrapper, data_in, data_out, test_row=[], net_args={'hidden_n': 15},
                             thread_n=self.thread, n_per_thread=math.ceil(n_net / self.thread), seed=42)  # 设置种子
        plotter = Plotter(data_in, data_out, *nets)
        err, err_std = plotter.score(noise_i=[2], noise_threshold=0.1)
        print("Net error: %.2e ± %.2e %%" % (err.item() * 100, err_std.item() * 100))
        # 绘制推力系数Cf关于燃烧室压强p0和环境压强pe的拟合曲面
        if self.plot:
            fig, ax = plotter.plot3d(margin=0.05, return_ax=True)
            ax.view_init(elev=30, azim=150)  # 调整视角
            fig.savefig(os.path.join(self.base_path, 'net_cf.png'))
            # fig.show()

        # 假定部分参数不变，并简化去噪网络的输入和输出
        def net_surface(X, Y):
            x, y = X.flatten(), Y.flatten()
            _data_in = torch.tensor(np.vstack([x, y, np.zeros_like(x)]).T)
            _data_out = plotter.eval_net(_data_in)[0]
            return _data_out.detach().numpy().reshape(X.shape)

        # 按高度平均的权重系数
        def weight_1(X, Y):
            x, y = X.flatten(), Y.flatten()
            return

        # 定义坐标变换
        alt2pres = standard_atmo(indep='alt')  # ind: 1
        pres2alt = standard_atmo(indep='pres')  # ind: 0
        # 计算参数范围
        _x_range = data_in[:, 0].min().item(), data_in[:, 0].max().item()  # p0
        _y_range = data_in[:, 1].min().item(), data_in[:, 1].max().item()  # pe
        x_range = _x_range  # p0
        y_range = pres2alt(_y_range)[0]  # H
        # 构建非均匀积分网格
        dx = (x_range[1] - x_range[0]) / n_int
        dy = (y_range[1] - y_range[0]) / n_int
        _x = torch.arange(x_range[0] + 0.5 * dx, x_range[1], dx)  # p0
        _y = torch.arange(y_range[0] + 0.5 * dy, y_range[1], dy)  # H
        X, Y = torch.meshgrid(_x,
                              torch.tensor(alt2pres(_y)[1]),
                              indexing='ij')  # p0, pe
        # 使用二维三点高斯积分计算当前参数域下Cf的加权平均值
        weight = ...
        k_x = 0.5 * dx * np.sqrt(3 / 5)
        k_y = 0.5 * dy * np.sqrt(3 / 5)
        Z = net_surface(X, Y)
        Z_11 = net_surface(X - k_x, Y - k_y)
        Z_12 = net_surface(X - k_x, Y + k_y)
        Z_21 = net_surface(X + k_x, Y - k_y)
        Z_22 = net_surface(X + k_x, Y + k_y)
        cf_avg = (16 * Z.sum() + 5 * (Z_11.sum() + Z_12.sum() + Z_21.sum() + Z_22.sum())) / (36 * n_int ** 2)

        # 保存去噪网络（会覆盖已有文件）
        info = {
            'input/output': 'inlet_p, atmo_p, continuity -> Cf',
            'MACs/params': profile(nets[0], inputs=torch.rand(size=(1, 1, nets[0].in_channels)), verbose=False)[1],
            'error': err,
            'error_std': err_std,
            'n_int': n_int,
            'Cf_int': cf_avg
        }
        save_net(net_path, *nets, desc=info)

        return cf_avg


class PlugSplineEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    基于NozzleCFD构建的样条塞式喷管强化学习环境
    """

    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        # 参数设置
        # self.cfd_params = {'inlet_p': [13.32e6], 'atmo_p': [101325]}  # 这里给定当前r_t和ε下的最佳压强比
        self.cfd_params = {'inlet_p': [3e6, 6e6, 12e6, 20e6],
                           'atmo_p': [p + 325 for p in [101e3, 60e3, 36e3, 20e3, 8e3, 0]]}

        # self.work_path = r'/home/zhuofeng/lgq/OpenFOAM/test/FluentWithDL_all/'
        # self.work_path = r'/home/zhuofeng/lgq/OpenFOAM/test/FluentWithDL/'
        # self.fluent_path = r'/public/software/ansys_inc/v221/fluent/bin/fluent'
        # self.cfdpost_path = r'/public/software/ansys_inc/v221/CFD-Post/bin/cfdpost'
        # self.script_path = r'/home/zhuofeng/lgq/python/OpenFOAM/'
        self.work_path = r'F:\Nozzle\OpenFOAM\FluentWithDL'
        self.fluent_path = r'E:\Ansys\2022R1\v221\fluent\ntbin\win64\fluent.exe'
        self.cfdpost_path = r'E:\Ansys\2022R1\v221\CFD-Post\bin\cfdpost.exe'
        self.cfdpost_path = r'E:\Ansys\2021R1\ANSYS Inc\v211\CFD-Post\bin\cfdpost.exe'
        self.script_path = r'F:\Nozzle\OpenFOAM'
        self.render_path = os.path.join(self.work_path, "render_%d" % int(time.time()))

        self.thread_n = 16  # 单求解器进程数
        self.worker_n = 6  # 求解器数量（建议设置为工况数的因数）
        self.spline_n = 5
        self.delta_sp_max = 0.1
        self.sp_margin = 0.01
        self.reward_factor_step = 1.0
        self.reward_factor_epoch = 20.0
        self.reward_base = 10
        self.reward_threshold = 0.7
        self.reward_threshold_dynamic = self.reward_threshold  # 动态阈值，由回合初始奖励计算得到
        self.cfd_continuity_limit = 5
        self.cfd_cf_limit = 3
        self.step_max = 20
        self.step_time_max = 3600
        self.convergence_criterion = 1.05

        self.render_mode = render_mode
        self.plot_n = 10

        # 更新自定义参数
        self.__dict__.update(**kwargs)

        # 理想喷管的CFD计算
        self.queue = QuestManager(parallel_n=self.thread_n * self.worker_n)
        self.queue.start()
        self.config = NozzleConfig(jet_type='plug',
                                   cfd_params=self.cfd_params,
                                   work_path=self.work_path,
                                   fluent_path=self.fluent_path,
                                   script_path=self.script_path,
                                   thread_n=self.thread_n,
                                   render=False)
        self.nozzle_target = NozzleCFD(self.config)
        ret = self._cfd_run(self.nozzle_target)
        assert ret and self._cfd_is_converge(self.nozzle_target), "plug calculation failed"

        # 定义动作空间和状态空间
        self.action_integral: Optional[np.ndarray] = None  # (length, theta, sp_1, ..., sp_n)
        self.state: Optional[np.ndarray] = None  # ((x1, ..., xn), (y1, ..., yn), (p1, ..., pn), (Ma_1, ..., Ma_n))
        self.step_count: Optional[int] = None
        self.epoch_count: Optional[int] = None
        self.cf_current: Optional[float] = None
        self.reward_current: Optional[Tuple[float, float]] = None
        self.nozzle_dir_current: Optional[str] = None  # 当前喷管构型的工作目录

        self.action_dim = (self.spline_n + 2,)
        self.action_space = gym.spaces.Box(low=-self.delta_sp_max * np.ones(self.action_dim),
                                           high=self.delta_sp_max * np.ones(self.action_dim),
                                           dtype=np.float64)
        self.action_integral_space = gym.spaces.Box(low=np.zeros(self.action_dim) + self.sp_margin,
                                                    high=np.ones(self.action_dim),
                                                    dtype=np.float64)
        # observation_dim = self.nozzle_target.data_field.shape[-1]
        self.observation_dim = (self.nozzle_target.data_field.shape[1],
                                len(self.nozzle_target.model.profile['plug_div']))
        self.observation_space = gym.spaces.Box(low=0,
                                                high=1,  # 暂时不满足，实际值为0~1.几
                                                shape=self.observation_dim,
                                                dtype=np.float64)

        # 定义基准量
        _, profile_target = self._cfd_get_state(self.nozzle_target)
        self.spline_target: Optional[Sequence] = None  # self._calc_target_spline()
        # self.spline_target = self._calc_target_spline()
        self.cf_baseline = self.nozzle_target.data['Cf'].max()  # 对于多工况选取最大值作为基准量（或平均值）
        # self.cf_baseline = self._calc_cf_baseline()
        self.profile_target = profile_target[: 2]
        print("Cf_baseline / Cf_max = ", self.cf_baseline / self.nozzle_target.data['Cf_max'].max())

        # 绘图窗口
        self.screen = None
        self.plot_lines = []
        self.recent_profile = deque(maxlen=self.plot_n)
        self.cf_data = []
        self.reward_text = None

    def _generate_spline_plug(self, spline_p: Sequence, timeout: Optional[int] = 10) -> Optional[NozzleCFD]:
        """使用给定插值参数生成样条喷管"""
        config = deepcopy(self.config)
        config.update(jet_type='plug-sp', spline_p=spline_p)
        config.mesh_n = int(config.mesh_n * spline_p[0])  # 根据样条喷管的长度参数调节扩张段节点数量 (0.9 * spline_p[0] + 0.1)
        try:
            # NozzleCFD->profile_to_msh->gmsh.model.mesh.generate有极小概率报错"Unable to recover the edge ..."
            # 可能是几何存在closed-loop，且二次报错会导致卡死（已在profile_to_msh内添加超时功能）
            nozzle = NozzleCFD(config)
        except Exception as e:
            traceback.print_exc()
            return None
        self.nozzle_dir_current = nozzle.base_path
        return nozzle

    def _model_is_valid(self, nozzle: NozzleCFD) -> bool:
        """返回几何模型是否满足要求"""
        return nozzle.model.is_increasing

    def _cfd_run(self, nozzle: NozzleCFD) -> bool:
        """用内置服务器队列提交fluent计算任务"""
        try:
            result_file = os.path.join(nozzle.base_path, 'fluent_result.txt')
            if os.path.exists(result_file):  # 若结果文件存在，则跳过计算
                print("Read existing result file: '%s'" % result_file)
            else:
                # 遍历并去除计算过的子任务
                job_skipped = []
                task_to_run = deepcopy(nozzle.task)  # 复制一份缩减版用于增量计算，不影响原任务组后处理
                for i, _ in enumerate(nozzle.task):
                    data_file = os.path.join(nozzle.task.work_path, 'plugNozzle-end.dat.h5')
                    if os.path.exists(data_file):  # 这里仅检查*-end.dat.h5文件
                        job_skipped.append(nozzle.task._job_list[i])
                for job in job_skipped:
                    task_to_run.job_list.remove(job)
                # 提交任务并计算
                if len(task_to_run) > 0:
                    worker_n = min(max(1, self.worker_n), len(task_to_run))
                    self.queue.submit(task_to_run, worker_n=worker_n)
                    time_wait = 0
                    task_id = len(self.queue.quest_info)
                    while True:
                        running = self.queue.state_single(task_id)
                        if not running:
                            break
                        time.sleep(1)
                        time_wait += 1
                        assert time_wait < self.step_time_max, "Maximum CFD running time reached"
            nozzle.postproc()  # 求解失败往往会导致后处理环节报错（结果文件非法）
        except Exception as e:
            traceback.print_exc()
            # print(repr(e), file=sys.stderr)
            return False
        else:
            return True

    def _cfd_get_state(self, nozzle: NozzleCFD) -> Tuple[float, np.ndarray]:
        """计算样条喷管的观测量（仅读取第一个工况的计算结果）"""
        n = len(nozzle.model.profile['plug_div'])  # 塞锥型面几何点的数量
        mesh_x, mesh_y, pressure, mach = nozzle.data_field[0, :, -n:]
        if hasattr(nozzle.model, 'L_max'):
            L_max = nozzle.model.L_max
        else:
            L_max = nozzle.model._Ma2plugXY(nozzle.model.Ma_e + 1e-6)[0, 0]
        p_a = np.log10(nozzle.data['atmo_p'][1])
        p_b = np.log10(nozzle.data['inlet_p'][1])
        p = np.log10(pressure + nozzle.data['atmo_p'][1])
        state = np.vstack([mesh_x / L_max,
                           mesh_y / self.nozzle_target.model.R_e,
                           (p - p_a) / (p_b - p_a),
                           mach / nozzle.model.Ma_e])
        # if nozzle.data_field.shape[-1] != self.nozzle_target.data_field.shape[-1]:  # 该条件会有极小概率失效（plug段网格点数量相等但扩张段不相等）
        if n != self.observation_dim[1]:
            # 对state进行插值，保证采样点数和nozzle_target一致（处理网格不一致问题）
            x_new = np.linspace(state[0, 0], state[0, -1], self.observation_dim[1])
            state = np.vstack([x_new,
                               interp1d(state[0], state[1], kind='quadratic')(x_new),
                               interp1d(state[0], state[2], kind='quadratic')(x_new),
                               interp1d(state[0], state[3], kind='quadratic')(x_new)])
        return nozzle.data['Cf'][1], state

    def _cfd_get_reward_epoch(self) -> float:
        """计算样条喷管的回合奖励"""
        # def dtw_distance_simple(series1, series2, factor=0.1):
        #     """简化版DTW实现"""
        #     n, m = len(series1), len(series2)
        #     dtw_matrix = np.zeros((n + 1, m + 1))
        #     dtw_matrix[0, 1:] = np.inf
        #     dtw_matrix[1:, 0] = np.inf
        #     # 动态规划
        #     for i in range(1, n + 1):
        #         for j in range(1, m + 1):
        #             cost = np.linalg.norm(series1[i - 1] - series2[j - 1])
        #             dtw_matrix[i, j] = cost + min(
        #                 dtw_matrix[i - 1, j],  # 插入
        #                 dtw_matrix[i, j - 1],  # 删除
        #                 dtw_matrix[i - 1, j - 1]  # 匹配
        #             )
        #     length = 0.5 * (np.linalg.norm(series1[1:] - series1[: -1], axis=1).sum() +
        #                     np.linalg.norm(series2[1:] - series2[: -1], axis=1).sum())
        #     normalized_distance = dtw_matrix[n, m] / length / (n + m)
        #     print(length, dtw_matrix[n, m], n, m, dtw_matrix[n, m] / (n + m))
        #     similarity = np.exp(-factor * normalized_distance)  # 使用指数衰减函数作为相似度分数: similarity = exp(-λ * distance)
        #     return similarity

        if self.step_count == 0:
            return 0.
        else:
            reward_min = 1.
            reward_max = reward_min * self.reward_factor_epoch
            cf_max = self.convergence_criterion * self.cf_baseline
            # cf_min = self.reward_threshold * self.cf_baseline
            cf_min = self.reward_threshold_dynamic * self.cf_baseline
            value = (self.cf_current - cf_min) / (cf_max - cf_min) * (np.log(reward_max) / np.log(self.reward_base))
            return np.power(self.reward_base, value)

    def _cfd_get_reward_step(self, nozzle: NozzleCFD) -> float:
        """计算样条喷管的单步奖励（仅读取第一个工况的计算结果）"""
        Cf_prob = nozzle.data['Cf'][1] / nozzle.data['Cf_max'][1]  # 注意该值可能大于一
        print("Cf / Cf_max = ", Cf_prob)
        # 积累奖励
        # reward = min(1.2, Cf_prob) * self.reward_factor_step
        # 指数积累奖励
        reward = np.power(self.reward_base, Cf_prob - 1) * self.reward_factor_step
        # 指数增量奖励
        # reward_0 = np.power(self.reward_base, self.cf_current / self.cf_baseline - 1)
        # reward = (np.power(self.reward_base, Cf_prob - 1) - reward_0) * self.reward_factor_step
        return reward

    def _cfd_is_converge(self, nozzle: NozzleCFD) -> bool:
        """返回计算结果是否收敛"""
        converge = np.all(np.abs(nozzle.data['report-def-continuity']) < self.cfd_continuity_limit) and \
                   np.all(nozzle.data['Cf'] > 0.) and np.all(nozzle.data['Cf'] < self.cfd_cf_limit)
        return converge

    def _env_is_finish(self, nozzle: NozzleCFD) -> bool:
        """返回当前回合是否结束（仅检查第一个工况的计算结果）"""
        return self.step_count >= self.step_max or \
            nozzle.data['Cf'][1] / self.cf_baseline >= self.convergence_criterion

    def _calc_target_spline(self):
        """通过最小化函数来计算目标型面对应的spline_p参数"""
        target_profile = interp1d(*zip(*self.nozzle_target.model.profile['plug_div']), kind='quadratic')
        config = deepcopy(self.config)
        config.update(jet_type='plug-sp',
                      work_path=os.path.join(self.work_path, 'profile_search'))
        func_value = None
        index = 0
        monitor = PlotAniNet(title="Search for target spline")

        def func(spline_p):
            nonlocal func_value, index
            config.update(spline_p=(1.0, *spline_p[1:]))
            try:
                nozzle = NozzleCFD(config)
            except Exception as e:
                traceback.print_exc()
            else:
                profile_x, profile_y = zip(*nozzle.model.profile['plug_div'])
                target_profile_y = target_profile(profile_x)
                func_value = np.linalg.norm(profile_y - target_profile_y)
            monitor.add(index, func_value)
            monitor.update()
            index += 1
            return func_value

        init_x = 0.5 * np.ones(self.spline_n + 1)
        result = minimize(func, x0=init_x, method='Nelder-Mead',
                          bounds=[(self.sp_margin, 1.0) for _ in range(self.spline_n + 1)])
        print(result)
        if result.success:
            self.target_spline = [1.0] + result.x.tolist()
            config.update(spline_p=self.target_spline, render=True)
            NozzleCFD(config)

    def _calc_cf_baseline(self):
        """计算nozzle_target的推力系数cf，支持多工况缩减"""
        if len(self.nozzle_target.data) == 1:
            cf_baseline = self.nozzle_target.data['Cf'][1]
        else:
            if self.config.is_linux:
                self.nozzle_target.thread = 1  # linux系统不支持多进程训练神经网络
            cf_baseline = self.nozzle_target.calc_cf(n_int=400, read_net=True)
        return cf_baseline

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        np.random.seed(seed)

        options = {} if options is None else options
        action_integral = options.get('action_integral')  # 可以指定初始构型
        epoch_count = options.get('epoch_count')  # 可以指定初始回合数
        cf_limit = options.get('cf_limit')  # 可以指定初始构型的最小cf数
        action_specified = action_integral is not None
        if action_specified or cf_limit is None:
            cf_limit = 0.

        self.cf_current, self.state = 0., None
        while True:
            # generate model and mesh
            if action_specified:
                action_integral = np.array(action_integral)
                if self.action_integral_space.contains(action_integral):
                    nozzle = self._generate_spline_plug(action_integral)
                    if nozzle is None:
                        raise ValueError("Invalid 'action_integral' (profile generation failure)")
                    if not self._model_is_valid(nozzle):
                        raise ValueError("Invalid 'action_integral' (non-monotone initial value)")
                else:
                    raise ValueError("Invalid 'action_integral' (initial value out of range)")
            else:
                print("Choose initial integral action with random seed %s" % seed)
                # randomly choose initial integral action
                while True:
                    action_integral = np.random.rand(self.action_integral_space.shape[0])
                    if self.action_integral_space.contains(action_integral):
                        nozzle = self._generate_spline_plug(action_integral)
                        if nozzle and self._model_is_valid(nozzle):
                            break

            # calculate initial state
            ret = self._cfd_run(nozzle)
            assert ret and self._cfd_is_converge(nozzle), \
                "plug-sp calculation failed"
            self.action_integral = action_integral
            self.cf_current, self.state = self._cfd_get_state(nozzle)
            self.reward_threshold_dynamic = self.cf_current / self.cf_baseline
            self.reward_current = (0., 0.)

            if self.cf_current >= cf_limit:
                break

        if epoch_count is not None:
            self.epoch_count = epoch_count
        elif self.epoch_count is None:
            self.epoch_count = 0
        else:
            self.epoch_count += 1
        self.step_count = 0

        if self.render_mode is not None:
            self.render()
        return self.state, {}

    def step(self, action: np.ndarray):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        new_action_integral = self.action_integral + action

        # truncated：动作空间限制
        if not self.action_integral_space.contains(new_action_integral):
            return self.state, self.reward_current[1], True, False, {'type': 'invalid action space'}

        # 样条喷管：创建构型
        nozzle = self._generate_spline_plug(new_action_integral)
        if nozzle is None:
            return self.state, self.reward_current[1], True, False, {'type': 'generation failed'}

        # truncated：样条型面单调性限制
        if not self._model_is_valid(nozzle):
            return self.state, self.reward_current[1], True, False, {'type': 'non-monotonic spline'}

        # 样条喷管：CFD计算
        ret = self._cfd_run(nozzle)

        # truncated：计算失败或CFD收敛性限制
        if not ret:
            return self.state, self.reward_current[1], True, False, {'type': 'calculation failed'}
        if not self._cfd_is_converge(nozzle):
            return self.state, self.reward_current[1], True, False, {'type': 'calculation divergence'}

        self.action_integral = new_action_integral
        self.step_count += 1

        reward_step = self._cfd_get_reward_step(nozzle)  # 需要在cf_current的值被覆盖前计算单步奖励
        self.cf_current, self.state = self._cfd_get_state(nozzle)
        reward_epoch = self._cfd_get_reward_epoch()
        truncated = self._env_is_finish(nozzle)

        self.reward_current = (reward_step, reward_epoch)

        if self.render_mode is not None:
            self.render()
        if truncated:
            return self.state, sum(self.reward_current), False, True, {}
        else:
            return self.state, self.reward_current[0], False, False, {}

    def render(self):
        assert self.state is not None, "Call reset before using step method."
        # 初始化绘图窗口
        if self.screen is None:
            fig, self.screen = plt.subplots(2, 1, figsize=(10, 8))
            fig.suptitle("Spline Plug Environment")
            self.plot_lines.append(self.screen[1].plot([], [], '.-')[0])
            for i in range(self.plot_n):
                alpha = (i + 1) / self.plot_n
                self.plot_lines.append(self.screen[0].plot([], [], '-', alpha=alpha)[0])
            # 绘制理想喷管推力系数和理论最优型面
            self.screen[0].plot(*self.profile_target, '--', c='gray', linewidth=1.2)
            self.screen[1].axhline(y=self.cf_baseline, linestyle='--', color='gray', linewidth=1.2)
            # 创建动态文本用于显示实时奖励
            self.reward_text = self.screen[0].text(1, 1, s='', fontsize=16, color='red', ha='right', va='top')
            # 绘图窗口设置
            self.screen[1].set_xlabel('$Step$')
            self.screen[1].set_ylabel('$C_f$')
            self.screen[0].grid()
            self.screen[1].grid()
            # 创建渲染输出目录
            os.makedirs(self.render_path, exist_ok=True)
        else:
            fig = self.screen[0].get_figure()
        # 每轮次需要清空上一轮次的绘图数据
        if self.step_count == 0:
            self.recent_profile.clear()
            self.cf_data = []
            for line in self.plot_lines:
                line.set_data([], [])

        # 绘制推力系数曲线
        self.cf_data.append([self.step_count, self.cf_current])
        self.plot_lines[0].set_data(*np.array(self.cf_data).T)

        # 更新奖励提示文本
        self.reward_text.set_text('' if self.reward_current is None else '%.1f\n%.1f' % self.reward_current)

        # 绘制最新的{self.plot_n}条喷管型面
        # self.recent_profile.append(self.state[: 2])
        # profile = list(self.recent_profile)
        # n = min(len(profile), self.plot_n)
        # order = np.argsort(np.array(self.cf_data[-n:])[:, 1])
        # cmap = plt.get_cmap('cool')
        # for i in range(1, n + 1):
        #     j = order[-i]
        #     k = int(256 * (n - i) / n)
        #     self.plot_lines[-i].set_data(*profile[j])
        #     self.plot_lines[-i].set_color(cmap(k))
        self.recent_profile.append(self.state[: 2])
        profile = list(self.recent_profile)
        n = min(len(profile), self.plot_n)
        scale = 256 / (self.cf_baseline - 1)  # 1 < Cf < Cf_baseline
        cmap = plt.get_cmap('cool')
        for i in range(1, n + 1):
            k = int(scale * (self.cf_data[-i][1] - 1))
            self.plot_lines[-i].set_data(*profile[-i])
            self.plot_lines[-i].set_color(cmap(k))

        for ax in self.screen:
            ax.relim()
            ax.autoscale_view()
        # fig.show()
        plt.pause(0.01)

        fig.savefig(os.path.join(self.render_path, f'{self.epoch_count:d}-{self.step_count:d}.png'))

    def post_processing(self, path: str = None):
        """使用CFDPost对指定路径或者工作路径下的全部算例文件执行后处理"""
        if path is None:
            cse_path = os.path.join(self.work_path, 'plug.cse')
        else:
            cse_path = os.path.join(path, 'plug.cse')
        copy_file(cse_path, os.path.join(self.script_path, 'plug.cse'))
        task = CFDPostQuest(self.cfdpost_path, cse_path)
        task.set_params(At=np.pi * self.nozzle_target.r_t ** 2,
                        gamma_R=self.nozzle_target.params['gamma'] * self.nozzle_target.params['R'])
        task.set_datafile(suffix='-end.dat.h5')
        self.queue.submit(task, worker_n=self.thread_n)
        # 循环等待计算结束并收集结果
        task_id = len(self.queue.quest_info)
        while True:
            running = self.queue.state_single(task_id)
            if not running:
                break
            time.sleep(1)
        task.get_result('result.txt', 'machNumber.png')

    def close(self):
        self.queue.stop()
        if self.screen is not None:
            fig = self.screen[0].get_figure()
            plt.close(fig)
            self.screen = None

    def __del__(self):
        self.close()


def generate_pretrain_data(data_file: str = 'pretrain_data.pth', n: int = 30, seed: int = 42, render: bool = False):
    """借助LHS使用`PlugSplineEnv`生成特定数量的样本用于预训练
    数据集仅包含第一个工况，获取完整数据需要借助`collect_nozzle_data`函数重新读取本地算例文件
    另外，区别于非等距的全nozzle数据，该数据集等距"""
    # 创建CFD环境
    env = PlugSplineEnv(render_mode='human' if render else None,
                        delta_sp_max=1.0, step_max=10000)
    # 定义文件路径
    data_path = os.path.join(env.work_path, data_file)
    config_list, state_list, label_list, dir_list = [], [], [], []  # 输入（动作）、样本观测（状态）、目标值（奖励）、算例目录
    # 采样并计算样本点
    samples = pyDOE2.lhs(env.action_dim[0], samples=n, criterion='center', random_state=seed)
    failure = []
    offset = 0
    while offset < n:
        try:
            state, _ = env.reset(seed=seed, options={'action_integral': samples[offset]})
        except (ValueError, AssertionError) as e:
            print("Calculation failed, skip point %s (%s)" % (samples[offset], e))
            failure.append(repr(e))
            offset += 1
        else:
            samples[offset] = env.action_integral
            state_list.append(state)
            label_list.append(env.cf_current)
            dir_list.append(env.nozzle_dir_current)
            break
    for i in range(offset + 1, n):
        state, reward, terminated, truncated, info = env.step(samples[i] - samples[offset])
        if terminated:
            print("Calculation failed, skip point %s (%s)" % (samples[i], info['type']))
            failure.append(info['type'])
        else:
            config_list.append(env.action_integral)
            state_list.append(state)
            label_list.append(env.cf_current)
            dir_list.append(env.nozzle_dir_current)
            offset = i  # 错误：offset += 1
    print("Failure reasons:")
    for item, count in Counter(failure).most_common():
        print(f"{count:>5d}  {item}")
    if len(config_list) > 0:
        # 将采样数据保存至文件
        data = {
            'config': config_list,
            'state': state_list,
            'label': label_list,
            'tag': dir_list
        }
        torch.save(data, data_path)
        print(f"{len(config_list):d}/{n:d} samples have beem saved to '{data_path:s}'")
    return env, state_list, label_list


def reduce_pretrain_data(data_dir: str, data_file: str = 'pretrain_data_175.pth',
                         n: int = None, profile_only: bool = True):
    """使用不同算法对预训练数据进行降维，从而可视化采样空间"""
    # 定义文件路径
    data_path = os.path.join(data_dir, data_file)  # 喷管
    picture_path = os.path.join(data_dir, 'cfdpost_pictures')
    # 从文件中读取数据
    data = torch.load(data_path)
    states = data['state']
    if profile_only:  # 仅考虑构型，或同时考虑构型和流场
        states = np.array([state[:2].flatten() for state in states])
    else:
        states = np.array([state.flatten() for state in states])
    # states = (states - states.mean()) / states.std()
    label = np.array(data['label'])
    assert len(states) == len(label), "Length of states and labels mismatch"
    tag = data['tag']
    assert len(tag) == len(states), "Length of tags and states mismatch"
    if n is not None:
        states = states[:n]
        label = label[:n]
        tag = tag[:n]
    # 绘制label分布
    fig0, ax0 = plt.subplots(figsize=(5, 5))
    ax0.hist(label, bins=30, alpha=0.75, color='blue', edgecolor='black')
    ax0.set_xlabel('Value')
    ax0.set_ylabel('Frequency')
    ax0.set_title('Frequency Distribution Histogram')
    ax0.grid(alpha=0.5)
    fig0.show()
    # 递归最小二乘
    rls = RLS(states.T, label)
    rls.train(cycle_count=3)
    states_p0 = np.vstack([rls.classify(states.T), label]).T
    # PCA线性降维
    pca = PCA(states.T)
    pca.train(ndim=2)
    states_p1 = pca.project(states.T).T
    # t-SNE非线性降维
    pca_ = PCA(states.T)
    pca_.train(ndim=20)  # 先基于PCA降至20维
    states_p2 = pca_.project(states.T).T
    tsne = TSNE(n_components=2, random_state=42, perplexity=50)  # 10, 30 or 50 (perplexity: 5~50, less than sample_n)
    states_p2 = tsne.fit_transform(states_p2)
    # 可视化样本点
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    kwargs = {'marker': 'o', 's': 14, 'c': label, 'cmap': plt.get_cmap('seismic')}
    ax[0].set_title('RLS', fontsize=20)
    diag = [0.6, 1.75]
    ax[0].plot(diag, diag, '--', color='gray', alpha=0.6)
    scatter_1 = ax[0].scatter(*states_p0.T, **kwargs)
    ax[1].set_title('PCA', fontsize=20)
    scatter_2 = ax[1].scatter(*states_p1.T, **kwargs)
    ax[2].set_title('t-SNE', fontsize=20)
    scatter_3 = ax[2].scatter(*states_p2.T, **kwargs)
    for i in range(3):
        ax[i].set_xticklabels([''] * len(ax[i].get_xticklabels()))
        ax[i].set_yticklabels([''] * len(ax[i].get_xticklabels()))
        ax[i].grid(alpha=0.5)

    # 设置散点图注解
    ax = ax.tolist()
    state = [states_p0, states_p1, states_p2]
    scatter = [scatter_1, scatter_2, scatter_3]
    annotate = [[], [], []]
    image = []
    for i, s in enumerate(tag):
        # 设置算例名注释
        for j in range(3):
            ann = ax[j].annotate(str(s).split('/')[-1], state[j][i], textcoords="figure fraction",
                                 xytext=(0, 0), ha='left', va='bottom', size=10)
            ann.set_visible(False)  # 初始时设置为不可见
            annotate[j].append(ann)
        # 预加载流场缩略图
        img_file = os.path.join(picture_path,
                                'machNumber-' + os.path.basename(s) + '_' +
                                [d for d in os.listdir(s) if os.path.isdir(os.path.join(s, d))][0] + '.png')
        if os.path.exists(img_file):
            img = Image.open(img_file)
            img.thumbnail((100, 100), Image.Resampling.LANCZOS)
            image.append(np.array(img))
        else:
            image.append(None)
    # 定义注解更新函数并绑定悬停事件
    this_ann = annotate[0][0]
    this_img = None

    def handle(event):
        nonlocal this_ann, this_img
        this_ax = event.inaxes
        if this_ax is None:
            return
        this_ann.set_visible(False)
        if this_img is not None:
            this_img.remove()
            this_img = None
        i = ax.index(this_ax)  # 查找当前子图的索引
        cont, ind = scatter[i].contains(event)
        if cont:
            j = ind['ind'][0]
            this_ann = annotate[i][j]
            this_ann.set_visible(True)
            if image[j] is not None:
                imagebox = OffsetImage(np.array(image[j]), zoom=1.0)
                imagebox.image.axes = this_ax
                img = AnnotationBbox(
                    imagebox,
                    state[i][j].tolist(),
                    xybox=(-60, 40),
                    xycoords='data',
                    boxcoords="offset points",
                    frameon=True,
                    pad=0.3,
                    bboxprops=dict(
                        boxstyle="round,pad=0.1",
                        facecolor="white",
                        edgecolor="gray",
                        linewidth=1,
                        alpha=0.9
                    )
                )
                this_img = this_ax.add_artist(img)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', handle)  # button_press_event

    fig.tight_layout()
    fig.show()


def collect_nozzle_data(data_dir: str, data_file: str = 'nozzle_data.pth', reduce_data: bool = True):
    """收集给定目录下的所有nozzle数据（不作筛选和后处理），用于构建代理模型
    对于多工况数据，可以给定缩减函数，将其压缩至单一工况
    区别于等距的预训练数据，该数据集非等距"""
    def clip_outlier(_data, threshold=3):
        # 离群值裁剪（假定不包含负值）：实际阈值非常难取，因为均值和方差也会受到离群值的强烈影响
        # 测试代码：ret = np.histogram(clip_outlier(deepcopy(pod_data[2]), threshold=3))
        z_score = (_data - _data.mean()) / _data.std()
        outlier_mask = abs(z_score) > threshold
        clip_value = np.max(_data[~outlier_mask])
        _data[outlier_mask] = clip_value
        rate = 100 * outlier_mask.sum() / outlier_mask.size
        print(f"{rate:.2f}% data has been clipped to {clip_value:.2e}")
        return _data

    def clip_hard(_data):
        # 硬裁剪，直接根据物理量手动指定裁剪值
        # 测试代码：ret = np.histogram(clip_hard(deepcopy(pod_data))[2])
        p_mask = _data[2] > 50e6  # 最大压强
        _data[2][p_mask] = 50e6
        rate_p = 100 * p_mask.sum() / p_mask.size
        m_mask = _data[3] > 10  # 最大马赫数
        _data[3][m_mask] = 10
        rate_m = 100 * m_mask.sum() / m_mask.size
        print(f"pressure clip: {rate_p:.2f}%, mach clip: {rate_m:.2f}%")
        return _data

    def reduce_field_pca(*field):
        # 基于PCA的场变量降维：(24*4*ni) -> 24*(1*4*ni)，且不同场量分开处理
        pca_reducer = []
        pca_data = np.concatenate(field, axis=-1).swapaxes(0, 1)
        pca_data = clip_hard(pca_data)
        for _data in pca_data:
            # _data = clip_outlier(_data)
            pca = PCA(_data)
            pca.train(ndim=1)  # 压强重构率30.5%（随ndim提升缓慢），其他量均接近100%
            pca.W = np.abs(pca.W) / np.abs(pca.W).sum()  # 将投影矩阵转换为归一化权重
            pca_reducer.append(pca)
        field_reduced = []
        for field_i in field:
            field_reduced_i = np.stack([pca_reducer[j].project(field_i[:, j, :])
                                           for j in range(len(pca_reducer))], axis=1)
            field_reduced.append(np.repeat(field_reduced_i, len(field_i), axis=0))
        return field_reduced

    def reduce_field_pod(*field):
        # 基于POD的场变量降维：(24*4*ni) -> 24*(1*4*ni)，且不同场量分开处理
        pod_data = np.concatenate(field, axis=-1).swapaxes(0, 1)
        pod_data = clip_hard(pod_data)
        mode_first = []
        for _data in pod_data:
            # _data = clip_outlier(_data)
            U, s, Vh = svd(_data.T, full_matrices=False)
            print("POD Recon. %.2f%%" % (100 * s[0] / s.sum()))  # 第一个奇异值即为最大奇异值
            mode_first.append(U.T[0])
        mode_first = np.stack(mode_first, axis=0)  # 第一阶POD模态
        field_reduced = []
        start = 0
        for field_i in field:
            end = start + field_i.shape[-1]
            field_reduced_i = mode_first[np.newaxis, :, start: end]
            field_reduced.append(np.repeat(field_reduced_i, len(field_i), axis=0))
        return field_reduced

    data_list = []
    # 对于完整的plug-sp的Nozzle仿真结果，以下文件必须存在（plug也存在，但spline_p字段为空；bell不存在）
    files = ['config.txt', 'fluent_result.txt', 'xy-plot-ycoord.txt', 'xy-plot-pressure.txt', 'xy-plot-mach.txt']
    for item in os.listdir(data_dir):
        path = os.path.join(data_dir, item)
        if os.path.isdir(path) and item.startswith('plug-sp'):
            if np.sum([os.path.exists(os.path.join(path, f)) for f in files]) == len(files):
                print("Reading directory '%s'..." % path)
                try:
                    # with open(os.path.join(path, 'config.txt'), 'r', encoding='utf-8') as f:
                    #     config = {'r_t': float(f.readline().strip()),
                    #               'epsilon': float(f.readline().strip()),
                    #               'spline_p': list(map(float, f.readline().strip().split()))  # 分隔符支持空格和制表
                    #               }
                    # 读取最终结果（files列表中的文件）
                    # nozzle = DataStruct(base_path=path,
                    #                     thread=16,
                    #                     plot=True,
                    #                     r_t=config['r_t'],
                    #                     data=None,
                    #                     data_field=None)  # 虚构的NozzleCFD类
                    config = NozzleConfig(case_dir=path)  # 读取本地文件
                    nozzle = NozzleCFD(config)
                    nozzle.postproc()  # 这里没有筛选发散的数据
                    # 检查数据完整性
                    if np.any(nozzle.data.isna()) or np.any(nozzle.data_field[:, 0, :].sum(axis=-1) == 0):
                        print("find null value in nozzle data or field data, please check out", file=sys.stderr)
                        continue
                    # 多工况数据重构
                    config = {'r_t': nozzle.r_t, 'epsilon': nozzle.epsilon, 'spline_p': nozzle.spline_p}
                    data = deepcopy(nozzle.data)
                    data_field = deepcopy(nozzle.data_field)
                    if reduce_data:
                        data['Cf'] = nozzle.calc_cf(n_int=400, read_cf=True)  # 加权缩减
                        data['Cf_max'] = np.mean(data['Cf_max'])  # 平均值缩减（作为Cf的参考值，也需要处理）
                        data['SpecImpulse'] = np.mean(data['SpecImpulse'])  # 平均值缩减
                        # data_field = np.repeat(data_field.mean(axis=0, keepdims=True),
                        #                        len(data_field), axis=0)  # 平均值缩减
                        # data_field = reduce_field_pca(data_field)[0]  # PCA缩减
                    data_list.append([config, data, data_field])
                except Exception as e:
                    traceback.print_exc()
    print("Data samples count:", Counter([str(len(d)) for c, d, f in data_list]))
    if reduce_data:  # 数据缩减要求工况数大于一
        configs, datas, data_fields = zip(*data_list)
        data_fields = reduce_field_pca(*data_fields)  # 全样本PCA缩减：压强重构率约70%（随ndim提升缓慢），马赫数约82%
        # data_fields = reduce_field_pod(*data_fields)  # 全样本POD缩减：压强重构率约32%，马赫数约47%
        data_list = list(zip(configs, datas, data_fields))
        # 对缩减后的喷管数据集，修改文件命名
        cuts = data_file.split('.')
        data_file = '.'.join(cuts[: -1]) + '_reduced.' + cuts[-1]
    data_path = os.path.join(data_dir, data_file)
    torch.save(data_list, data_path)
    print(f"{len(data_list):d} nozzle data has been saved to '{data_path:s}'")
    return data_list


def recalculate_plug_nozzle(dir_path: str):
    """重新计算特定的塞式喷管型面（调试工具，用于解决计算失败问题）"""
    # TODO: 功能未实现
    join = os.path.join
    fluent_path = NozzleConfig().fluent_path  # 仅使用NozzleConfig提供的fluent路径
    result_file = join(dir_path, 'fluent_result.txt')
    if os.path.exists(result_file):
        print("Find existing result file, it will be overwritten after a while")
    sub_path_list = []
    _sub_path_list = []
    for name in os.listdir(dir_path):
        sub_path = join(dir_path, name)
        if os.path.isdir(sub_path) and name not in ['fluent_pictures', 'cfdpost_pictures']:
            sub_path_list.append(Path(sub_path))
            data_file = join(sub_path, 'plugNozzle-end.dat.h5')
            if not os.path.exists(data_file):  # 这里仅检查*-end.dat.h5文件
                _sub_path_list.append(sub_path)
    if _sub_path_list:
        exe_args = []
        sub_dirs = []
        for i, sub_path in enumerate(_sub_path_list, start=1):
            dir_name = os.path.basename(sub_path)
            exe_args.append(f"2ddp -g t16 -mpi=intel -ssh < {join(sub_path, 'FluentScript.jou')}"
                            f" > {join(sub_path, 'fluent.out')}")  # 实际上进程数通过命令行参数指定
            sub_dirs.append(dir_name)
        task0 = CustomQuest(dir_path, fluent_path, exe_args, sub_dirs, thread_n=16)  # 根据已有文件手动指定计算任务
        print(task0)
        queue = QuestManager(parallel_n=32)
        queue.submit(task0, worker_n=2)
        task_id = len(queue.quest_info)
        while True:  # 等待计算完成
            running = queue.state_single(task_id)
            if not running:
                break
            time.sleep(1)
    script_path = join(dir_path, 'plug.jou')
    task = FluentQuest(fluent_path, script_path)  # 仅使用FluentQuest的数据收集功能
    task._job_dir_list = sub_path_list
    task.get_result('report-def-0-rfile.out')
    task.get_xyplot('xy-plot-ycoord.txt')
    task.get_xyplot('xy-plot-pressure.txt')
    task.get_xyplot('xy-plot-mach.txt')


class PlugSplineSurrEnv(PlugSplineEnv):
    """
    基于NozzleCFD的强化学习环境，使用历史训练数据构建的代理模型来替换父类PlugSplineEnv中的耗时CFD计算
    """

    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        self._initialized = False
        self.model_check = True  # 是否开启模型检查（通过关键字形参设置）
        super().__init__(render_mode, **kwargs)

        self.nozzle_data_path = os.path.join(self.work_path, 'nozzle_data_reduced.pth')
        self.nozzle_surrogate_path = os.path.join(self.work_path, 'nozzle_surrogate_reduced_v3.pth')
        self.feature_dim = AgentConfig().sensor_hidden_dim

        if os.path.exists(self.nozzle_surrogate_path):
            self.nozzle_surrogate = torch.load(self.nozzle_surrogate_path)
            print("Read surrogate model of PlugSplineEnv from '%s'" % self.nozzle_surrogate_path)
        else:
            self.nozzle_surrogate = self._train()
            torch.save(self.nozzle_surrogate, self.nozzle_surrogate_path)
            print("Save surrogate model of PlugSplineEnv to '%s'" % self.nozzle_surrogate_path)

        self.observation_current = None

        self._initialized = True

    def _train(self, n_net: int = 16, outfile: bool = True):
        """使用PlugSplineEnv的历史计算数据构建代理模型"""
        nozzle_data = torch.load(self.nozzle_data_path)
        # 预处理喷管数据集，设置筛选：（1）r_t、epsilon、spline_n和环境设置一致（2）满足收敛条件（3）仅第一个工况
        nozzle_key = np.array([(item[0]['r_t'], item[0]['epsilon'], len(item[0]['spline_p'])) for item in nozzle_data])
        nozzle_query = np.array([self.nozzle_target.r_t, self.nozzle_target.epsilon, self.action_dim[0]])
        cond_1 = np.sum((np.abs(nozzle_key / nozzle_query - 1.) < 1e-6), axis=1) == len(nozzle_query)
        nozzle_converge = np.array([
            (np.all(np.abs(item[1]['report-def-continuity']) < self.cfd_continuity_limit),  # 连续性
             np.all(item[1]['Cf'] > 0.) and np.all(item[1]['Cf'] < self.cfd_cf_limit),  # 推力系数
             item[2][:, 2, :].max() <= self.nozzle_target.data['inlet_p'].max() and
             item[2][:, 2, :].min() > - self.nozzle_target.data['atmo_p'].max()  # 型面压强
             ) for item in nozzle_data])
        cond_2 = nozzle_converge[:, 0] & nozzle_converge[:, 1] & nozzle_converge[:, 2]
        ind = np.nonzero(np.logical_and(cond_1, cond_2))[0]
        print(f"[Surr] use {len(ind):d}/{len(nozzle_data):d} nozzle data to train surrogate model")
        config, data, data_field = np.array(nozzle_data, dtype=object)[ind].T  # np.ndarray
        data = [val.iloc[0] for val in data]

        # 预处理喷管数据集，根据当前强化学习环境的参数来进行缩放
        def scale_data_field(states_raw):
            mach_raw = states_raw[0, 3]
            n = len(mach_raw) - np.nonzero(mach_raw > 1.)[0][0] + 1  # 根据马赫数等于一的位置推测塞锥型面几何点的数量
            mesh_x, mesh_y, pressure, mach = states_raw[0, :, -n:]
            L_max = self.nozzle_target.model._Ma2plugXY(self.nozzle_target.model.Ma_e + 1e-6)[0, 0]
            p_a = np.log10(self.nozzle_target.data['atmo_p'].max())
            p_b = np.log10(self.nozzle_target.data['inlet_p'].max())
            p = np.log10(pressure + self.nozzle_target.data['atmo_p'].max())
            state = np.vstack([mesh_x / L_max,
                               mesh_y / self.nozzle_target.model.R_e,
                               (p - p_a) / (p_b - p_a),
                               mach / self.nozzle_target.model.Ma_e])
            return state

        data_field = list(map(scale_data_field, data_field))
        # 可以将经过处理的数据保存至本地文件
        if outfile:
            cuts = self.nozzle_data_path.split('.')
            outfile_path = '.'.join(cuts[: -1]) + '_selected.' + cuts[-1]
            torch.save(list(zip(config, data, data_field)), outfile_path)
            print("[Surr] Preprocessed data (r_t=%.3e, epsilon=%.1f, spline_n=%d)" % tuple(nozzle_query),
                  "has been saved to '%s'" % outfile_path, sep=' ')
        # 首先使用data_field数据训练编码器和解码器
        print(f"[Surr] hidden dimension of sensor: {self.feature_dim:d}")
        t_start = time.time()
        encoder, decoder, loss, feature = train_dae(  # ConvolutionSensor（对样本点数有限制，但性能更好）
            data_field, feature_dim=self.feature_dim, net_type=0, sample_dim=self.observation_dim[1],
            lr=5e-1, num_epochs=600, batch_size=8, noise_p=0.1, weight_delay=1e-5, device='cuda',
            net_args={}, return_feature=True)  # 返回包含输入、特征、输出在内的中间结果
        encoder = encoder[0]
        t_end = time.time() - t_start
        print(f"[Surr] sensor training finished (loss: {loss[-1, 1]:.3e}, time: {t_end:.2e} s)")
        # 构建用于训练主网络的数据集
        parameter = np.array([val['spline_p'] for val in config])
        variable = np.array([val[['Cf', 'SpecImpulse', 'report-def-continuity']].to_numpy() for val in data])
        input_data = torch.tensor(np.hstack([parameter, variable[:, [2]]]))
        output_data = torch.tensor(np.hstack([feature[1][0], variable[:, [0, 1]]]))
        # 然后使用data和feature数据训练主网络（代理模型）
        print(f"[Surr] backbone: {input_data.shape[1]:d} channels -> {output_data.shape[1]:d} channels")
        t_start = time.time()
        nets = train_net(DenoiseWrapper, input_data, output_data, test_row=[],
                         net_args={'hidden_n': 32, 'layer_n': 5}, thread_n=1, n_per_thread=n_net)
        plotter = Plotter(input_data, output_data, *nets)
        err, err_std = plotter.score()
        t_end = time.time() - t_start
        print(f"[Surr] main network training finished",
              f"(error: {err.mean().item() * 100:.2e} ± {err_std.mean().item() * 100:.2e} %, time: {t_end:.2e} s)")
        # 对整个网络进行微调
        t_start = time.time()
        decoder.to('cpu')
        encoder.to('cpu')
        train_iter = DataLoader(TensorDataset(input_data, output_data, feature[2]),
                                batch_size=32, shuffle=True, pin_memory=True)
        optimizer = torch.optim.Adam([*[{'params': net.parameters()} for net in nets],
                                      {'params': decoder.parameters()}],
                                     lr=2e-4, weight_decay=1e-5)
        loss_f = torch.nn.MSELoss()
        loss_std = np.array([*output_data[:, -2:].std(dim=0), *feature[2].reshape((-1, 4)).std(dim=0)])

        # print(loss_std)

        def calc_loss(X, F, Y, grad=False):  # 计算各个物理量的归一化损失
            if grad:
                _F_list = []
                for net in nets:
                    _F_list.append(net.forward(X))
                _F = torch.stack(_F_list, dim=0).mean(dim=0)
                _Y = decoder(_F[:, : -2])
                return torch.stack([
                    loss_f(_F[:, -2], F[:, -2]),  # C_f
                    loss_f(_F[:, -1], F[:, -1]),  # I_s
                    loss_f(_Y[:, :, 0], Y[:, :, 0]),  # x
                    loss_f(_Y[:, :, 1], Y[:, :, 1]),  # y
                    loss_f(_Y[:, :, 2], Y[:, :, 2]),  # p
                    loss_f(_Y[:, :, 3], Y[:, :, 3]),  # Ma
                ]) / torch.tensor(loss_std) ** 2
            else:
                _F, _ = plotter.eval_net(X)
                with torch.no_grad():
                    _Y = decoder(_F[:, : -2])
                return (np.array([
                    float(loss_f(_F[:, -2], F[:, -2])),  # C_f
                    float(loss_f(_F[:, -1], F[:, -1])),  # I_s
                    float(loss_f(_Y[:, :, 0], Y[:, :, 0])),  # x
                    float(loss_f(_Y[:, :, 1], Y[:, :, 1])),  # y
                    float(loss_f(_Y[:, :, 2], Y[:, :, 2])),  # p
                    float(loss_f(_Y[:, :, 3], Y[:, :, 3])),  # Ma
                ]) / loss_std ** 2).tolist()

        ft_loss_list = [[*calc_loss(input_data, output_data, feature[2]), None]]
        for epoch in range(200):
            n, epoch_loss = 0, 0.
            for i, (X, F, Y) in enumerate(train_iter):
                optimizer.zero_grad()
                ft_loss = calc_loss(X, F, Y, grad=True).mean()
                ft_loss.backward()
                optimizer.step()
                with torch.no_grad():
                    epoch_loss += float(ft_loss) * X.shape[0]
                    n += X.shape[0]
            epoch_loss /= n
            test_loss = calc_loss(input_data, output_data, feature[2])
            ft_loss_list.append([*test_loss, epoch_loss])
            print(f"Epoch {epoch + 1:d}: average train loss {epoch_loss:.4f},",
                  f"single test loss {list(map(lambda x: round(x, 4), test_loss))}")
        t_end = time.time() - t_start
        print(f"[Surr] surrogate finetuning finished",
              f"(value loss: {np.mean(ft_loss_list[0][:2]):.3f} -> {np.mean(ft_loss_list[-1][:2]):.3f},",
              f"field loss: {np.mean(ft_loss_list[0][2:-1]):.3f} -> {np.mean(ft_loss_list[-1][2:-1]):.3f},",
              f"time: {t_end:.2e} s)")
        # 生成代理模型
        surrogate = {
            'main': plotter,
            'decoder': decoder,  # 输出等间距样本点，同PlugSplineEnv一致
            'encoder': encoder,  # 输入非等距样本点（全nozzle数据集训练），因此不可作为感知器使用  ## 已改动
            'loss': (ft_loss_list[-1], loss[-1, 1], err, err_std)
        }
        return surrogate

    def _generate_spline_plug(self, spline_p: Sequence, timeout: Optional[int] = 10) -> NozzleCFD:
        # 型面生成阶段就完成代理模型的计算
        parameter = torch.tensor(np.hstack([spline_p, np.zeros(1)])).unsqueeze(0)
        feature, feature_std = self.nozzle_surrogate['main'].eval_net(parameter)
        cf = feature[0, self.feature_dim].item()
        state = self.nozzle_surrogate['decoder'](feature[:, : self.feature_dim])
        state = state.detach().numpy()[0].T
        confidence = torch.exp(- feature_std[0] / self.nozzle_surrogate['main'].data_out.std(dim=0))
        self.observation_current = (cf, state, confidence.mean().item())
        return self.nozzle_target

    def _model_is_valid(self, nozzle: NozzleCFD) -> bool:
        # 返回型面预测值的单调性
        if self.model_check:
            y = self.observation_current[1][1]
            # # 使用平滑可以大幅缓解代理模型预测型面时产生的锯齿状噪声
            # kernel_size = int(0.5 * len(y) / self.spline_n)
            # y = np.convolve(y, np.ones(kernel_size), mode='valid')
            return np.all(y[:-1] >= y[1:])
        else:
            return True

    def _cfd_run(self, nozzle: NozzleCFD) -> bool:
        if self._initialized:
            print("[Surr] average prediction confidence: %.2f%%" % (self.observation_current[2] * 100))
            return True
        else:
            return super()._cfd_run(nozzle)

    def _cfd_get_state(self, nozzle: NozzleCFD) -> Tuple[float, np.ndarray]:
        if self._initialized:
            return self.observation_current[0], self.observation_current[1]
        else:
            return super()._cfd_get_state(nozzle)

    def _cfd_get_reward_step(self, nozzle: NozzleCFD) -> float:
        Cf_prob = self.observation_current[0] / nozzle.data['Cf_max'][1]  # 注意该值可能大于一
        cf_delta = self.observation_current[0] - self.cf_current
        # 缩放后有：delta = 2 * (prob - prob_last) 且 prob,|delta| ~∈ [0, 1]
        scale = (1. - self.reward_threshold_dynamic)  # 全区间（乘以0.5则为半区间）
        Cf_prob_scaled = (Cf_prob - self.reward_threshold_dynamic) / scale
        cf_delta_scaled = 2 * cf_delta / (scale * nozzle.data['Cf_max'][1])
        print("Cf / Cf_max = ", Cf_prob)
        # 指数积累奖励
        # reward = np.power(self.reward_base, Cf_prob_scaled - 1.) * self.reward_factor_step
        # 线性积累奖励
        # reward = Cf_prob_scaled * self.reward_factor_step
        # 指数增量奖励
        # reward = np.power(self.reward_base, cf_delta_scaled) * self.reward_factor_step
        # 线性增量奖励（更大惩罚）
        if cf_delta_scaled >= 0:
            reward = cf_delta_scaled * self.reward_factor_step
        else:
            reward = cf_delta_scaled * (5 * self.reward_factor_step)
        # 存活奖励
        # reward_survive = self.step_count / self.step_max
        # 空奖励
        # reward = 0.
        return reward

    def _cfd_is_converge(self, nozzle: NozzleCFD) -> bool:
        return True

    def _env_is_finish(self, nozzle: NozzleCFD) -> bool:
        return self.step_count >= self.step_max or \
            self.observation_current[0] / self.cf_baseline >= self.convergence_criterion

    def _calc_field_error(self) -> Optional[Sequence[float]]:
        """计算代理模型在基线喷管上的误差（包含spline_i的拟合误差）"""
        if self.target_spline is None:
            print("Please set the value of `target_spline` first before calculating error.", file=sys.stderr)
            return None
        _observation = copy.deepcopy(self.observation_current)
        self._generate_spline_plug(self.target_spline)
        cf, state, confidence = self.observation_current[:]
        self.observation_current = _observation
        # 将nozzle_target的状态插值为均匀点
        _, state0, = super()._cfd_get_state(self.nozzle_target)  # 注意多工况下该函数不包含缩减操作，单工况则不存在此问题
        x_new = np.linspace(state0[0, 0], state0[0, -1], self.observation_dim[1])
        state0 = np.vstack([x_new,
                            interp1d(state0[0], state0[1], kind='quadratic')(x_new),
                            interp1d(state0[0], state0[2], kind='quadratic')(x_new),
                            interp1d(state0[0], state0[3], kind='quadratic')(x_new)])
        # 绘制state对比曲线
        fig, axes = plt.subplots(3, 1, sharex='col')
        for i, tag in enumerate(['y', 'p', 'M']):
            axes[i].plot(state0[0], state0[i+1], '--', c='gray')
            axes[i].plot(state[0], state[i + 1], '-', c='black')
            axes[i].set_ylabel(r"$\bar{%s}$" % tag, fontsize=18)
            axes[i].grid()
            if i == 0:
                axes[i].legend(["$CFD$", "$Surrogate$"])
            if i == 2:
                axes[i].set_xlabel(r"$\bar{x}$", fontsize=18)
        fig.show()
        # 分别计算不同物理量的误差
        errors = []
        for i in range(4):
            # error_i = np.mean(np.abs(state[i] - state0[i]) / np.mean(state0[i]))
            error_i = np.mean(np.abs(state[i] - state0[i]) / 1.0)  # 对于归一化物理量，可以选择直接将分母设为1
            errors.append(error_i)
        real_err = lambda x: (x - errors[0]) / (1.0 - errors[0])
        print("error of y-coordinate (%%): %.3f" % (real_err(errors[1]) * 100))
        print("error of pressure (%%): %.3f" % (real_err(errors[2]) * 100))
        print("error of mach number (%%): %.3f" % (real_err(errors[3]) * 100))
        cf_baseline = self._calc_cf_baseline()
        error_cf = np.abs(cf - cf_baseline) / cf_baseline
        errors.append(error_cf)
        print("error of thrust coefficient (%%): %.3f" % (error_cf * 100))
        return errors


def reduce_surrogate_model(model_path: str, sample_n: int = 10_000, seed: int = 42,
                           profile_only: bool = True, kernel_size: Optional[int] = None):
    """喷管代理模型可视化"""
    # TODO: 有概率抽卡抽到不一样的tSNE分布（为什么？）还有小概率tsne.fit_transform报错reshape失败
    model = torch.load(model_path)
    input_dim = model['main'].data_in.shape[1]
    encoder_dim = model['encoder'].out_channels

    # 代理模型采样
    samples = pyDOE2.lhs(input_dim - 1,
                         samples=sample_n, criterion='center', random_state=seed)
    # LHS采样到的数据就处于(0, 1)范围内，与action_integral范围一致，因此不需要缩放
    input_data = torch.tensor(np.hstack([samples, np.zeros((sample_n, 1))]))
    feature = model['main'].eval_net(input_data)[0]
    output_data = feature[:, encoder_dim:]
    output_state = model['decoder'](feature[:, : encoder_dim])
    output_data = output_data.detach().numpy()
    output_state = output_state.detach().numpy()

    # 计算可视化矩阵
    cmap = mpl.colormaps['viridis']  # RdBu_r
    scale_func = lambda x: np.nan_to_num(x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
    log_func = lambda x, w=100, o=0.0: o + (1. - o) * np.tanh(w * x)
    grid_n = output_state.shape[1]
    grid = np.zeros((grid_n, grid_n))  # 直方图网格
    contour = np.zeros((grid_n, grid_n))  # 直方图网格（cf加权）
    grid_valid = grid.copy()
    contour_valid = contour.copy()
    count_valid = 0
    images = np.zeros((sample_n, grid_n, grid_n, 4))  # 提前创建图片缓存用于后续计算
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 3))  # np.ones(3, np.uint8)
    x_scaled = scale_func(output_state[:, :, 0])
    cf_scaled = scale_func(output_data[:, 0])
    for i in range(sample_n):
        x = x_scaled[i]
        y = scale_func(output_state[i, :, 1])
        ind_x = ((grid_n - 1) * x).astype(dtype=int)
        ind_y = ((grid_n - 1) * (1. - y)).astype(dtype=int)
        # 去除重复元素
        # ind_x, ind_ind = np.unique(ind_x, return_index=True)
        # ind_y = ind_y[ind_ind]
        grid[ind_y, ind_x] += 1
        contour[ind_y, ind_x] += output_data[i, 0]
        if kernel_size is not None:
            y = np.convolve(y, np.ones(kernel_size), mode='valid')  # 缓解代理模型预测型面时产生的锯齿状噪声
        if np.all(y[: -1] > y[1:]):  # 单调曲线
            grid_valid[ind_y, ind_x] += 1
            contour_valid[ind_y, ind_x] += output_data[i, 0]
            count_valid += 1
        # 保存构型图片
        mask = np.zeros_like(images[i, :, :, 3])
        mask[ind_y, ind_x] = 1.
        mask = cv2.dilate(mask, kernel, iterations=5)  # 形态学膨胀
        images[i] = cmap(cf_scaled[i])
        if np.any(y[: -1] <= y[1:]):
            images[i] = 200 / 256  # 非单调曲线使用浅灰色标注
        images[i, :, :, 3] = mask
        # images[i, ind_y, ind_x, :] = cmap(cf_scaled[i])
    with np.errstate(invalid='ignore'):
        contour = scale_func(contour / grid)  # 均值
        contour_valid = scale_func(contour_valid / grid_valid)
    grid = log_func(scale_func(grid))
    grid_valid = log_func(scale_func(grid_valid))
    accept_rate = 100 * count_valid / sample_n
    print("Accept rate: %.2f%%" % accept_rate)

    # 绘制喷管构型直方图
    cmap = mpl.colormaps['viridis']
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    color_array = cmap(contour)
    color_array[:, :, 3] = grid
    im_1 = ax[0].imshow(color_array)
    color_array_valid = cmap(contour_valid)
    color_array_valid[:, :, 3] = grid_valid
    im_2 = ax[1].imshow(color_array_valid)
    ax[1].text(int(0.98 * grid_n), int(0.02 * grid_n), s=f"$Accept\\ rate:\\ {accept_rate:.2f}\\%$",
               fontsize=16, color='black', ha='right', va='top')
    # plt.colorbar(im_2)
    for ax_i in ax:
        ax_i.set_xlabel("x")
        ax_i.set_ylabel("y")
        ax_i.set_xticks(ticks=np.linspace(0, grid_n - 1, 5), labels=[])
        ax_i.set_yticks(ticks=np.linspace(0, grid_n - 1, 5), labels=[])
    fig.tight_layout()
    fig.show()

    # t-SNE降维
    if profile_only:
        state_flat = output_state[:, :, :2].reshape((sample_n, -1))
    else:
        state_flat = output_state.reshape((sample_n, -1))
    pca = PCA(state_flat.T)
    pca.train(ndim=20)  # 先基于PCA降至20维
    states_proj = pca.project(state_flat.T).T
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30)  # 10 or 50 (perplexity: 5~50, less than sample_n)
    states_proj = tsne.fit_transform(states_proj)

    # 计算可视化矩阵
    grid_n = 20
    contour = [[[] for _ in range(grid_n)] for _ in range(grid_n)]
    grid = np.zeros((grid_n, grid_n), dtype=int)  # 以低分辨率重置直方图网格
    states_proj_min = states_proj.min(axis=0)
    states_proj_max = states_proj.max(axis=0)
    states_proj = (states_proj - states_proj_min) / (states_proj_max - states_proj_min)  # 归一化投影坐标
    for i in range(sample_n):
        x_ind, y_ind = ((grid_n - 1) * states_proj[i, :] + 0.5).astype(dtype=int)
        contour[x_ind][y_ind].append([i, output_state[i], states_proj[i]])
        grid[x_ind, y_ind] += 1

    # 绘制喷管构型的低维流形图
    canvas_n = 50
    scale_factor = 2 / grid.max()
    scale_img_to_canvas = canvas_n / grid_n
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    for ind in np.argsort(grid.flatten()):
        x_ind = ind % grid_n
        y_ind = ind // grid_n
        if grid[y_ind, x_ind] == 0:
            continue
        # 对每个区块计算中心位置，并查找离中心最近的样本点
        _sample_i_list, _state_list, _state_proj_list = zip(*contour[y_ind][x_ind])
        _state_proj_arr = np.array(_state_proj_list)
        z_ind = np.argsort(np.linalg.norm(_state_proj_arr - _state_proj_arr.mean(axis=0), axis=1))[0]
        _state = _state_list[z_ind]
        _sample_i = _sample_i_list[z_ind]
        _scale = grid[y_ind, x_ind] * scale_factor
        # 计算子图在画布上的位置和大小
        center_x = scale_img_to_canvas * (x_ind + 0.5)
        center_y = scale_img_to_canvas * (y_ind + 0.5)
        half_side = 0.5 * scale_img_to_canvas * _scale
        extent = [int(center_x - half_side),
                  int(center_x + half_side),
                  int(center_y - half_side),
                  int(center_y + half_side)]
        ax2.imshow(images[_sample_i], extent=extent, aspect='auto')
    ax2.set_xlim(0, canvas_n)
    ax2.set_ylim(0, canvas_n)
    fig2.show()


def test_nozzle_cfd():
    q = QuestManager(parallel_n=64)
    q.start()

    # 理想喷管
    # config = NozzleConfig(jet_type='plug',
    #                       cfd_params={'inlet_p': [13.32e6], 'atmo_p': [101325]},
    #                       work_path=r'/home/zhuofeng/lgq/OpenFOAM/test/FluentWithDL/')
    # nozzle = NozzleCFD(config)

    # 样条喷管
    config = NozzleConfig(jet_type='plug-sp',
                          spline_p=[1, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2],
                          cfd_params={'inlet_p': [13.32e6], 'atmo_p': [101325]},
                          work_path=r'/home/zhuofeng/lgq/OpenFOAM/test/FluentWithDL/')
    nozzle = NozzleCFD(config)

    q.submit(nozzle.task, worker_n=4)
    nozzle.postproc()
    print(nozzle.calc_cf(n_net=16))

    q.stop()


if __name__ == '__main__':
    # RuntimeError: Unable to handle autograd's threading in combination with fork-based multiprocessing
    # set_start_method('spawn')

    env = PlugSplineEnv(render_mode='human')
    base_path = env.work_path
    # env.reset(seed=42)
    # env.step(np.array([0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0]))
    # env.post_processing()

    # 预训练样本数：101/(30+110)
    # env, states, cf = generate_pretrain_data(data_file='pretrain_data_1.pth', n=30, render=True)
    # env, states, cf = generate_pretrain_data(data_file='pretrain_data_2.pth', n=110, render=True)

    # 预训练样本数：175/250
    # env, states, cf = generate_pretrain_data(n=250, render=True)

    # 预训练样本可视化
    # reduce_pretrain_data(base_path)

    # 基于代理模型的强化学习环境
    # nozzle_data = collect_nozzle_data(data_path)
    # env = PlugSplineSurrEnv(render_mode='human')
    # data_in = env.nozzle_surrogate['main'].data_in.numpy()
    # data_out = env.nozzle_surrogate['main'].data_out.numpy()
    # plotter = env.nozzle_surrogate['main']
    # env.reset(seed=42, options={'action_integral': data_in[0, : -1]})
    # env.step(np.array([-0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]))

    # 代理模型可视化
    # reduce_surrogate_model(os.path.join(base_path, 'nozzle_surrogate_reduced_v3.pth'), profile_only=False)

    # 构建多工况数据集
    # 注意，下面的代码执行前需要设置PlugSplineEnv的工况参数和工作路径！！！
    # generate_pretrain_data('pretrain_data_seed10.pth', seed=10, n=400, render=False)  # 181/400, 检查所有工况的收敛性
    # generate_pretrain_data('pretrain_data_seed11.pth', seed=11, n=400, render=False)  # 171/400
    # generate_pretrain_data('pretrain_data_seed12.pth', seed=12, n=400, render=False)  # 176/400
    # generate_pretrain_data('pretrain_data_seed13.pth', seed=13, n=400, render=False)  # 194/400
    # recalculate_plug_nozzle(r'/home/zhuofeng/lgq/OpenFOAM/test/FluentWithDL_all/'
    #                         r'plug-sp_Rt2.00e-01_eps16.0_n141_403792627812863733371624937/')
    # collect_nozzle_data(base_path)

    # 测试代理模型的性能
    env = PlugSplineSurrEnv(render_mode='human')
    env.target_spline = [0.99, 0.758, 0.727, 0.566, 0.513, 0.351, 0.225]
    errors = env._calc_field_error()
    print(errors)


