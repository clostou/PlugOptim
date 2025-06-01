"""



"""

import sys
import numpy as np
from scipy.optimize import root_scalar

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from cfd_toolbox.vec import offset_space, func_space_2d, ThirdPoly, BSpline, CubicSpline

import gmsh

__all__ = ['External', 'ExternalSpine', 'profile_to_msh']


class External:
    """塞式喷管型面设计（Lee ,1963）"""

    def __init__(self, Ma_e=None, epsilon=None,
                 R_t=None, h_t=None, R_e=None, r_t=None, gamma=1.4, n=100):
        # 喷管基本参数
        self.Ma_e = Ma_e
        self.epsilon = epsilon
        self.n = n
        # 喷管喉道及扩张段参数
        self.R_t = R_t
        self.h_t = h_t
        self.R_e = R_e
        self.r_t = r_t    # 等效钟形喷管喉道半径
        self.gamma = gamma
        self._theta = None    # 喉道速度矢量角
        # 喷管入口及收敛段参数
        self.R_c = None
        self.L_c = None
        self.alpha_ext = None
        # 计算域参数
        self.domain_w = None
        self.domain_h = None
        self.nozzle_w = None
        self.nozzle_h = None
        # 带计算域的喷管型面
        self.profile = None
        self.points = None

    @staticmethod
    def _Ma2epsilon(Ma, gamma=1.4):
        """由喷管出口设计马赫数计算喷管面积比，基于一维等熵流"""
        a = 2 / (gamma + 1)
        b = (gamma - 1) / 2
        c = (gamma + 1) / (2 * (gamma - 1))
        return (a * (1 + b * Ma ** 2)) ** c / Ma

    @staticmethod
    def _Ma2nu(Ma, gamma=1.4):
        """量热完全气体从声速膨胀至特定马赫数产生的转折角，基于普朗特-迈耶（Prandtl-Meyer, P-M）膨胀波，即稀疏波"""
        a = ((gamma + 1) / (gamma - 1)) ** 0.5
        b = ((gamma - 1) / (gamma + 1)) ** 0.5
        x = np.sqrt(Ma ** 2 - 1)
        return a * np.arctan(b * x) - np.arctan(x)

    @staticmethod
    def _ideal_cf(Ma_e, gamma):
        """喷管最大推力系数， 基于一维绝热等熵流和质量守恒"""
        a = gamma * (2 / (gamma + 1)) ** \
            ((gamma + 1) / (2 * (gamma - 1)))
        b = (gamma - 1) / 2
        return a * Ma_e / np.sqrt(1 + b * Ma_e)

    def derive(self):
        none = type(None)
        # 等熵关系：ε <--> Ma
        if not isinstance(self.Ma_e, none):
            self.epsilon = self._Ma2epsilon(self.Ma_e, self.gamma)
        elif not isinstance(self.epsilon, none):
            self.Ma_e = root_scalar(lambda x: self._Ma2epsilon(x, self.gamma) - self.epsilon,
                                    x0=1., x1=10.).root
        else:
            print("One of exit Mach number (Ma_e), expansion ratio (ε) must be set.")
            return
        # P-M膨胀波关系：Ma --> ν(or δ)
        delta = np.pi / 2 - self._Ma2nu(self.Ma_e, self.gamma)
        k = np.sin(delta)  # sin(δ)
        # 几何关系：R_e <--> h_t <--> R_t
        ratio = (self.epsilon - (self.epsilon * (self.epsilon - k)) ** 0.5) / (self.epsilon * k)  # h_t/R_e
        if not isinstance(self.R_t, none):
            self.R_e = self.R_t / (1 - ratio * k)
            self.h_t = ratio * self.R_e
            self.r_t = np.sqrt((self.R_e**2 - self.R_t**2) / k)
        elif not isinstance(self.h_t, none):
            self.R_e = self.h_t / ratio
            self.R_t = self.R_e - k * self.h_t
            self.r_t = np.sqrt((self.R_e ** 2 - self.R_t ** 2) / k)
        elif not isinstance(self.R_e, none):
            self.h_t = ratio * self.R_e
            self.R_t = self.R_e - k * self.h_t
            self.r_t = np.sqrt((self.R_e ** 2 - self.R_t ** 2) / k)
        elif not isinstance(self.r_t, none):
            self.R_e = self.r_t / np.sqrt(ratio * (2 - ratio * k))
            self.h_t = ratio * self.R_e
            self.R_t = self.R_e - k * self.h_t
        else:
            print("One of throat radius from plug axis (R_t), width of throat gap (h_t), \
            lip radius from plug axis (R_e),equivalent throat radius (r_t) must be set.")
            return
        self._theta = delta - np.pi / 2
        print(f"External plug config:\n\tMa_e = {self.Ma_e:.3f}\n\tepsilon = {self.epsilon:.2f}\n",
              f"\tR_t = {self.R_t:.6e}\n\th_t = {self.h_t:.6e}\n",
              f"\tR_e = {self.R_e:.6e}\n\tr_t = {self.r_t:.6e}", sep='')
        # 按比例确定其余尺寸
        self.R_c = 10 * self.h_t
        self.L_c = 10 * self.h_t
        self.alpha_ext = np.pi * 10 / 180
        self.domain_w = 18 * self.R_e
        self.domain_h = 6 * self.R_e
        self.nozzle_w = 2 * self.R_e
        self.nozzle_h = 2 * self.R_e

    def _Ma2plugXY(self, Ma_i):
        """基于P-M膨胀波、质量守恒、一维等熵流及几何关系，由马赫数得到塞锥型面坐标"""
        theta = self._theta + self._Ma2nu(Ma_i, self.gamma)    # 速度矢量角
        mu = np.arcsin(1 / Ma_i)    # 马赫角
        k = np.sin(- theta + mu) / np.sin(mu) / self.epsilon
        R_x = self.R_e * np.sqrt(1 - self._Ma2epsilon(Ma_i, self.gamma) * k)
        X_x = (self.R_e - R_x) / np.tan(- theta + mu)
        return np.vstack([X_x, R_x]).T

    def generate(self, n=None, factor=None):
        if n:
            self.n = n
        else:
            n = self.n
        points = {}

        # 计算塞锥型面坐标
        if factor:
            # 若指定线网格偏置factor，则根据偏置重构坐标点
            Ma_i = np.linspace(1, self.Ma_e - 1e-6, 1000)
            plug_points = self._Ma2plugXY(Ma_i)
            length = np.sum(np.linalg.norm(plug_points[1: ] - plug_points[: -1], axis=1))
            t = np.convolve(offset_space(0, length, n, factor=factor), np.array([1, -1]), mode='valid')
            Ma_i = 1.
            plug_points = [self._Ma2plugXY(1)]
            for size in t:
                Ma_i = root_scalar(lambda x: np.linalg.norm(self._Ma2plugXY(x) - plug_points[-1]) - size,
                                   method='secant', x0=Ma_i, x1=self.Ma_e - 1e-6).root
                plug_points.append(self._Ma2plugXY(Ma_i))
            points['plug_div'] = np.vstack(plug_points)
        else:
            # 默认按马赫数等间隔划分（坐标点会呈现先变密再变稀的特征）
            Ma_i = np.linspace(1, self.Ma_e - 1e-6, n)
            points['plug_div'] = self._Ma2plugXY(Ma_i)

        # 根据塞锥扩张段划分获取其他区域尺寸
        ls_small = np.linalg.norm(points['plug_div'][1] - points['plug_div'][0])
        ls_large = 1.5 * np.linalg.norm(points['plug_div'][-1] - points['plug_div'][-2])
        ls_middle = 0.8 * ls_small + 0.2 * ls_large

        # 计算其余边界
        p_lip = np.array([.0, self.R_e])

        t = offset_space(np.pi, self._theta + np.pi / 2, ls_middle / self.R_c, factor=ls_small / ls_middle)
        circle_r = self.R_c * np.vstack([np.cos(t), np.sin(t)]).T
        circle_c = points['plug_div'][0] - circle_r[-1]
        points['plug_con'] = circle_c + circle_r

        R_c = - points['plug_con'][0, 0] / np.cos(self._theta + np.pi / 2)
        t = offset_space(self._theta + np.pi / 2, np.pi / 2, ls_small / R_c, factor=ls_middle / ls_small)
        circle_r = R_c * np.vstack([np.cos(t), np.sin(t)]).T
        circle_c = p_lip - circle_r[0]
        points['wall_con'] = circle_c + circle_r

        t = offset_space(points['plug_con'][0, 0] - self.L_c, points['plug_con'][0, 0], ls_middle)
        points['plug_inlet'] = np.vstack([t, points['plug_con'][0, 1] * np.ones(len(t))]).T
        t = offset_space(points['wall_con'][-1, 0], points['wall_con'][-1, 0] - self.L_c, ls_middle)
        points['wall_inlet'] = np.vstack([t, points['wall_con'][-1, 1] * np.ones(len(t))]).T
        t = offset_space(points['wall_inlet'][-1, 1], points['plug_inlet'][0, 1], ls_middle)
        points['inlet'] = np.vstack([points['wall_inlet'][-1, 0] * np.ones(len(t)), t]).T

        t = offset_space(points['plug_div'][-1, 0], self.domain_w, ls_large)
        points['axis'] = np.vstack([t, np.zeros(len(t))]).T
        t = offset_space(0, self.domain_h, ls_large)
        points['outlet_right'] = np.vstack([points['axis'][-1, 0] * np.ones(len(t)), t]).T
        t = offset_space(points['outlet_right'][-1, 0], - self.nozzle_w, ls_large)
        points['outlet_upper'] = np.vstack([t, points['outlet_right'][-1, 1] * np.ones(len(t))]).T
        t = offset_space(points['outlet_upper'][-1, 1], self.nozzle_h, ls_large)
        points['outlet_left'] = np.vstack([points['outlet_upper'][-1, 0] * np.ones(len(t)), t]).T
        t = offset_space(points['outlet_left'][-1, 0],
                               (points['outlet_left'][-1, 1] - self.R_e) * np.tan(self.alpha_ext), ls_large)
        points['wall_outer'] = np.vstack([t, points['outlet_left'][-1, 1] * np.ones(len(t))]).T
        line = points['wall_outer'][-1] - p_lip
        points['wall_end'] = p_lip + line * offset_space(1, 0, ls_middle / np.linalg.norm(line),
                                                               factor=ls_small / ls_middle).reshape(-1, 1)

        self.profile = points
        self.points = np.vstack([points['wall_outer'][: -1],
                                 points['wall_end'][: -1],
                                 points['wall_con'][: -1],
                                 points['wall_inlet'][: -1],
                                 points['inlet'][: -1],
                                 points['plug_inlet'][: -1],
                                 points['plug_con'][: -1],
                                 points['plug_div'][: -1],
                                 points['axis'][: -1],
                                 points['outlet_right'][: -1],
                                 points['outlet_upper'][: -1],
                                 points['outlet_left'][: -1]])
        print(f"Line division info:\n\tls_small = {ls_small: .3e}\n\tls_middle = {ls_middle: .3e}\n\t",
              f"ls_large = {ls_large:.3e}\n\tn_total = {len(self.points)}", sep='')

    def get_profile(self):
        n_wall = sum([len(self.profile[part]) - 1 for part in
                      ['wall_outer', 'wall_end', 'wall_con', 'wall_inlet']])
        n_inlet = len(self.profile['inlet']) - 1
        n_plug = sum([len(self.profile[part]) - 1 for part in
                      ['plug_inlet', 'plug_con', 'plug_div']])
        n_axis = len(self.profile['axis']) - 1
        n_outlet = sum([len(self.profile[part]) - 1 for part in
                        ['outlet_right', 'outlet_upper', 'outlet_left']])
        ind = np.cumsum([n_wall, n_inlet, n_plug, n_axis, n_outlet])
        ind_pair = [0] + np.repeat(ind[: -1], 2).tolist() + [ind[-1]]
        tag = dict([(name, [ind_pair[2 * i], ind_pair[2 * i + 1]])
                    for i, name in enumerate(['shell', 'inlet', 'plug', 'axis', 'outlet'])])
        return self.points, tag

    def plot(self):
        if isinstance(self.profile, type(None)):
            print("Please generate plug profile first before plotting.")
            return
        ddx = 2 * 10**(np.ceil(np.log10(self.R_e)) - 1)
        dx = ddx * 5
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle("Profile of External Expansion Plug Nozzle", fontsize=22)
        ax.fill(*self.points.T, color='gray', alpha=0.5)
        for tag, line in self.profile.items():
            if tag == 'inlet':
                ax.plot(*line.T, '.-g', markersize=1, lw=1)
            elif tag.startswith('outlet'):
                ax.plot(*line.T, '.-b', markersize=1, lw=1)
            else:
                ax.plot(*line.T, '.-k', markersize=3, lw=1)
        throat = np.vstack([self.profile['wall_con'][0], self.profile['plug_con'][-1]])
        ax.plot(*throat.T, '-r', lw=2)
        ax.plot(ax.get_xlim(), [0, 0], '--y', lw=2)
        ax.set_xlabel("x", fontsize=18)
        ax.set_ylabel("y", fontsize=18)
        ax.set_xticks(np.arange(ax.get_xlim()[0] // dx * dx, ax.get_xlim()[1], dx))
        ax.set_ylim(ax.get_ylim()[0], 1.5 * ax.get_ylim()[1])
        ax.set_yticks(np.arange(ax.get_ylim()[0] // dx * dx, ax.get_ylim()[1], dx))
        ax.grid()

        axins = inset_axes(ax, width='20%', height='40%', loc='lower left',
                           bbox_to_anchor=(0.3, 0.5, 1, 1),
                           bbox_transform=ax.transAxes)
        axins.fill(*self.points.T, color='gray', alpha=0.5)
        for tag, line in self.profile.items():
            if tag.startswith('shell') or tag.startswith('plug'):
                axins.plot(*line.T, '.-k', markersize=0.8, lw=1)
        axins.plot(*throat.T, '-r', lw=2)
        axins.set_xticks(np.arange(axins.get_xlim()[0] // ddx * ddx, ax.get_xlim()[1], ddx))
        axins.set_xlim(- 0.4 * self.R_e, 0.4 * self.R_e)
        axins.set_yticks(np.arange(axins.get_ylim()[0] // ddx * ddx, ax.get_ylim()[1], ddx))
        axins.set_ylim(0.6 * self.R_e, 1.4 * self.R_e)
        axins.grid()
        mark_inset(ax, axins, loc1=4, loc2=2, fc='none', ec='k', lw=1)

        fig.tight_layout()
        fig.show()
        plt.pause(0.01)


class ParamProfile:
    """可指定起始端斜率的平滑拟合型面"""

    def __init__(self, points, theta, _weight=50):
        k = 0.1 * np.linalg.norm(points[-1] - points[0]) / (len(points) - 1)
        p = points[0] + k * np.array([np.cos(theta), np.sin(theta)])
        self.points = np.vstack([points[0], p, points[1:]])
        weight = np.ones(self.points.shape[0])
        weight[[0, 1, -1]] = 1e6
        weight[1] = _weight
        self.curve = BSpline(*self.points.T, smooth=True, weight=weight)

    def __call__(self, x):
        return self.curve.sample(x)

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(*self.points.T, 'bo', label='Original points')
        xx = np.linspace(self.points[0, 0], self.points[-1, 0], 100)
        ax.plot(xx, self.__call__(xx), 'r', label='BSpline')
        ax.grid()
        ax.legend(loc='best')
        fig.show()


def profile_test():
    from cfd_toolbox.plot import plot

    points = np.array([[-0.02468736, 0.68822575, 1.37645149, 2.06467724, 2.75290298, 3.41644137],
                       [0.79564695, 0.63651756, 0.47738817, 0.31825878, 0.15912939, 0.]])
    theta = -30 / 180 * np.pi
    x = np.linspace(points[0, 0], points[0, -1], 100)
    c_list = ['#E6ED4F', '#C8E87D', '#26B78E']

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    for i, dy in enumerate(np.linspace(-0.1, 0.1, 3)):
        _points = points.copy()
        _points[1, 2] += dy
        pp = ParamProfile(_points.T, theta)
        ax1.plot(*pp.points.T, 'o', c='#343B1A')
        ax1.plot(x, np.array(list(map(pp, x))), '-', c=c_list[i], label='BSpline (dy=%.1f)' % dy)
    ax1.grid()
    ax1.legend(loc='best')
    ax2 = fig.add_subplot(122)
    for i, dt in enumerate(np.linspace(-50, 20, 8)):
        pp = ParamProfile(points.T, theta + dt * np.pi / 180)
        ax2.plot(*pp.points.T, 'o', c='#343B1A')
        ax2.plot(x, np.array(list(map(pp, x))), '-', label='BSpline (dt=%d°)' % dt)
    ax2.grid()
    ax2.legend(loc='best')
    fig.show()


class ParamProfile2:
    """可指定起始端斜率的平滑拟合型面"""

    def __init__(self, points, theta, _weight=0.5):
        self.points = points
        self.m, _ = self.points.shape
        weight = np.ones(self.m)
        weight[[0, -1]] = 1e6
        self.curve = BSpline(*self.points.T, smooth=True, weight=weight)
        self.curve_bias = CubicSpline(*self.points[[0, -1]].T, type='clamped', v0=np.tan(theta), vn=0)
        self._k = - _weight * (self.points[-1, 0] - self.points[0, 0]) / (self.m - 1)

    def __call__(self, x):
        alpha = np.exp((x - self.points[0, 0])**0.5 / self._k)
        return (1 - alpha) * self.curve.sample(x) + alpha * self.curve_bias.sample(x)

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(*self.points.T, 'bo', label='Original points')
        xx = np.linspace(self.points[0, 0], self.points[-1, 0], 500)
        ax.plot(xx, self.curve.sample(xx), 'b-.', label='BSpline-1')
        ax.plot(xx, self.curve_bias.sample(xx), 'g-.', label='BSpline-2')
        ax.plot(xx, self.__call__(xx), 'r-', label='Composite')
        ax.grid()
        ax.legend(loc='best')
        fig.show()


def transform_points(normalized_points, l=1., h=1., by_prop=False):
    """将网络输出的归一化参数（取值范围(0, 1)，一维数组）转换为塞锥扩张段的样条曲线控制点"""
    if by_prop:  # 累加映射（即输入值为控制点间的间隔）
        n = len(normalized_points)
        _y = np.cumsum(normalized_points[1:])
        _y = _y[: -1] / _y[-1]
    else:  # 非累加映射
        n = len(normalized_points) + 1
        _y = normalized_points[1:]
    x = np.linspace(0, l, n)
    y = np.concatenate([[h], (1. - _y) * h, [0.]])
    points = np.vstack([x, y]).T
    theta = normalized_points[0] * (- np.pi / 2)
    return points, theta


class ExternalSpine(External):
    """基于样条曲线的参数化塞式喷管型面"""

    def __init__(self, r_t, epsilon):
        super(ExternalSpine, self).__init__(epsilon=epsilon, r_t=r_t)
        self.L_max = None    # 塞锥最大长度（即理想塞锥长度）
        self.L = None    # 塞锥长度（喉道至末端）
        self.theta = None    # 喉道速度矢量角
        self.bsp = None    # 喷管扩张段样条曲线

    def derive(self):
        pass

    def generate(self, length, delta, *spline_p, n=100, factor=1.):
        """ 给定塞锥样条曲线参数，生成型面
        length: 塞锥长度，range(0,1]
        delta: 声速面与轴线的夹角，range(0,1)
        spline_p: 样条控制点参数，实际上为各点间隔的比重，range(0,1]
        """
        if n:
            self.n = n
        else:
            n = self.n
        points = {}

        # 计算理想塞式喷管参数
        super(ExternalSpine, self).derive()
        self.L_max = self._Ma2plugXY(self.Ma_e + 1e-6)[0, 0]

        # 计算喷管的基本几何参数
        self.theta = delta * (- np.pi / 2)
        _delta = np.pi / 2 + self.theta
        k = np.sin(_delta)  # sin(δ)
        ratio = (self.epsilon - (self.epsilon * (self.epsilon - k)) ** 0.5) / (self.epsilon * k)  # h_t/R_e
        self.R_e = self.r_t / np.sqrt(ratio * (2 - ratio * k))
        self.h_t = ratio * self.R_e
        self.R_t = self.R_e - k * self.h_t
        self.L = length * self.L_max

        print(f"Spline config:\n\tspline_n = {len(spline_p):d}\n",
              f"\ttheta = {180*self.theta/np.pi:.2f}° (ideal {180*self._theta/np.pi:.2f}°)\n"
              f"\tL = {self.L:.6e}\n\tL_max = {self.L_max:.6e}", sep='')

        # 使用样条曲线生成塞锥扩张段型面
        self.bsp = ParamProfile2(
            *transform_points(
                (delta,) + spline_p,
                l=self.L,
                h=self.R_t,
                by_prop=True
            ),
            _weight=0.8
        )
        self.bsp.plot()
        x = np.linspace(0, self.L, n)
        y = self.bsp(x)
        is_increasing = np.all(y[1:] < y[:-1])  # 判断插值曲线单调性
        print(is_increasing)
        points['plug_div'] = func_space_2d(lambda x: (x, self.bsp(x)), 0, self.L, self.L / n, factor=factor)
        points['plug_div'][:, 0] -= self.h_t * np.cos(_delta)
        '''x = offset_space(0, self.L, n, factor=factor)
        points['plug_div'] = np.vstack([x + self.h_t * np.sin(self.theta),
                                        cubic_spline(*spline_p, x, type='half-clamped',
                                                     v0=np.tan(self.theta), vn=.0)]).T'''

        # 根据塞锥扩张段划分获取其他区域尺寸
        ls_small = np.linalg.norm(points['plug_div'][1] - points['plug_div'][0])
        ls_large = 1.5 * np.linalg.norm(points['plug_div'][-1] - points['plug_div'][-2])
        ls_middle = 0.8 * ls_small + 0.2 * ls_large

        # 计算其余边界
        p_lip = np.array([.0, self.R_e])

        t = offset_space(np.pi, self.theta + np.pi / 2, ls_middle / self.R_c, factor=ls_small / ls_middle)
        circle_r = self.R_c * np.vstack([np.cos(t), np.sin(t)]).T
        circle_c = points['plug_div'][0] - circle_r[-1]
        points['plug_con'] = circle_c + circle_r

        R_c = - points['plug_con'][0, 0] / np.cos(self.theta + np.pi / 2)
        t = offset_space(self.theta + np.pi / 2, np.pi / 2, ls_small / R_c, factor=ls_middle / ls_small)
        circle_r = R_c * np.vstack([np.cos(t), np.sin(t)]).T
        circle_c = p_lip - circle_r[0]
        points['wall_con'] = circle_c + circle_r

        t = offset_space(points['plug_con'][0, 0] - self.L_c, points['plug_con'][0, 0], ls_middle)
        points['plug_inlet'] = np.vstack([t, points['plug_con'][0, 1] * np.ones(len(t))]).T
        t = offset_space(points['wall_con'][-1, 0], points['wall_con'][-1, 0] - self.L_c, ls_middle)
        points['wall_inlet'] = np.vstack([t, points['wall_con'][-1, 1] * np.ones(len(t))]).T
        t = offset_space(points['wall_inlet'][-1, 1], points['plug_inlet'][0, 1], ls_middle)
        points['inlet'] = np.vstack([points['wall_inlet'][-1, 0] * np.ones(len(t)), t]).T

        t = offset_space(points['plug_div'][-1, 0], self.domain_w, ls_large)
        points['axis'] = np.vstack([t, np.zeros(len(t))]).T
        t = offset_space(0, self.domain_h, ls_large)
        points['outlet_right'] = np.vstack([points['axis'][-1, 0] * np.ones(len(t)), t]).T
        t = offset_space(points['outlet_right'][-1, 0], - self.nozzle_w, ls_large)
        points['outlet_upper'] = np.vstack([t, points['outlet_right'][-1, 1] * np.ones(len(t))]).T
        t = offset_space(points['outlet_upper'][-1, 1], self.nozzle_h, ls_large)
        points['outlet_left'] = np.vstack([points['outlet_upper'][-1, 0] * np.ones(len(t)), t]).T
        t = offset_space(points['outlet_left'][-1, 0],
                         (points['outlet_left'][-1, 1] - self.R_e) * np.tan(self.alpha_ext), ls_large)
        points['wall_outer'] = np.vstack([t, points['outlet_left'][-1, 1] * np.ones(len(t))]).T
        line = points['wall_outer'][-1] - p_lip
        points['wall_end'] = p_lip + line * offset_space(1, 0, ls_middle / np.linalg.norm(line),
                                                         factor=ls_small / ls_middle).reshape(-1, 1)

        self.profile = points
        self.points = np.vstack([points['wall_outer'][: -1],
                                 points['wall_end'][: -1],
                                 points['wall_con'][: -1],
                                 points['wall_inlet'][: -1],
                                 points['inlet'][: -1],
                                 points['plug_inlet'][: -1],
                                 points['plug_con'][: -1],
                                 points['plug_div'][: -1],
                                 points['axis'][: -1],
                                 points['outlet_right'][: -1],
                                 points['outlet_upper'][: -1],
                                 points['outlet_left'][: -1]])
        print(f"Line division info:\n\tls_small = {ls_small: .3e}\n\tls_middle = {ls_middle: .3e}\n\t",
              f"ls_large = {ls_large:.3e}\n\tn_total = {len(self.points)}", sep='')


def profile_to_msh(profile_points, partition_tag, lc=0.1, planner=True, save_path=None):
    """使用gmsh创建喷管几何并划分非结构网格"""
    gmsh.initialize()
    gmsh.clear()
    gmsh.model.add("profile")
    # 创建计算域几何
    pts = []
    for point in profile_points:
        pts.append(gmsh.model.geo.addPoint(*point, .0, lc))
    gmsh.model.geo.rotate([(0, p) for p in pts], 0, 0, 0, 1, 0, 0, -np.pi * 1.5 / 180)
    pts.append(pts[0])
    curve = [gmsh.model.geo.addLine(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    cl = gmsh.model.geo.addCurveLoop(curve)
    s = gmsh.model.geo.addPlaneSurface([cl])
    if planner:
        gmsh.model.geo.synchronize()
        # 创建物理组命名
        gmsh.model.addPhysicalGroup(2, [s], name='nozzle')
        ind = 0
        for name, section in partition_tag.items():
            n = section[1] - section[0]
            gmsh.model.addPhysicalGroup(1, curve[ind: ind + n], name=name)
            ind += n
        # 网格生成
        gmsh.model.mesh.generate(2)
    else:
        ov = gmsh.model.geo.revolve([(2, s)], 0, 0, 0, 1, 0, 0, np.pi * 3 / 180, [1], recombine=True)
        gmsh.model.geo.synchronize()
        # 创建物理组命名
        gmsh.model.addPhysicalGroup(3, [ov[1][1]], name='nozzle')
        gmsh.model.addPhysicalGroup(2, [s], name='asym1')
        gmsh.model.addPhysicalGroup(2, [ov[0][1]], name='asym2')
        ind = 2
        for name, section in partition_tag.items():
            if name != 'axis':
                n = section[1] - section[0]
                gmsh.model.addPhysicalGroup(2, [s[1] for s in ov[ind: ind + n]], name=name)
                ind += n
        # 网格生成
        gmsh.model.mesh.generate(3)
    if save_path:
        suffix = save_path.split('.')[-1]
        if suffix == 'msh':
            gmsh.option.setNumber('Mesh.Format', 1)
            gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
        elif suffix == 'bdf':
            gmsh.option.setNumber('Mesh.Format', 31)
            gmsh.option.setNumber('Mesh.SaveElementTagType', 2)
        else:
            pass
        gmsh.write(save_path)
    try:
        gmsh.fltk.run()
    except:
        pass
    gmsh.finalize()


if __name__ == '__main__':
    '''plug = External(epsilon=16, r_t=0.2)
    plug.derive()
    plug.generate(n=150, factor=6)
    '''
    #profile_test()
    plug = ExternalSpine(epsilon=16, r_t=0.2)
    plug.generate(1, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2, n=300, factor=6)

    plug.plot()
    profile, tag = plug.get_profile()
    profile_to_msh(profile, tag, lc=0., planner=True, save_path='plug.bdf')


