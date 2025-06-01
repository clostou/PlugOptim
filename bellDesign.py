"""



"""

import sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.integrate import quad
from matplotlib import pyplot as plt

from cfd_toolbox.vec import ThirdPoly, offset_space_nd, func_space_2d
from cfd_toolbox.plot import func_plot, plot

from plugDesign import profile_to_msh

__all__ = ['CharacteristicsNozzle']


class CharacteristicsNozzle:
    """推力最优钟形喷管型面设计（Rao ,1957）"""

    def __init__(self, r_t, rho_t, T_0=300, gamma=1.4, M=28.966e-3, axial_sym=False):
        self.r_t = r_t
        self.rho_t = rho_t
        self.gamma = gamma
        self.M = M
        self.delta = 1 if axial_sym else 0

        R = 8.314 / M
        c_p = R * gamma / (gamma - 1)
        h_0 = c_p * T_0

        if rho_t < 2 * r_t:
            raise ValueError("Parameter rho_t is not much larger than r_t, this will derive inaccurate flow field"
                             "(current rho_t / r_t = %.2f)" % (rho_t / r_t))
        L = rho_t + r_t
        c_star = np.sqrt((2 * (gamma - 1) * h_0) / (gamma + 1))
        alpha = np.sqrt((1 + self.delta) / ((gamma + 1) * r_t * rho_t)) * L
        self.sauer = \
            {
                'L': L,
                'c_star': c_star,
                'y2x_k': ((gamma + 1) * alpha) / (2 * (3 + self.delta) * L),
                'U_a': alpha / L,
                'U_b': ((gamma + 1) * alpha**2) / (2 * (1 + self.delta) * L**2)
            }

        self.chara = \
            {
                'U2c_a': -0.5 * (gamma - 1),
                'U2c_b': 0.5 * (gamma + 1) * c_star**2,
                'U_max': np.sqrt(2 * h_0)
            }

        self.field_param = \
            {
                'throat_n': 0,
                'kernel_n': 0,
                'kernel_dtheta': 0.,
                'kernel_thetaB_max': 0.,
                'kernel_thetaB': 0.,
                'wall_n': 0
            }
        self.field = {}
        self.wall_p = None

        # 喷管入口及收敛段参数
        self.L_c = None
        self.theta_c = None
        # 计算域参数
        self.domain_w = None
        self.domain_h = None
        self.nozzle_w = None
        self.nozzle_h = None
        # 带计算域的喷管型面
        self.profile = None
        self.points = None

    def _throat_line(self, y):
        """使用索尔(Sauer,R. ,1947)方法来近似计算喉道区的流场
        给定喉道截面上起始面的纵坐标值y，返回该点处的流动参数
        """
        x = self.sauer['y2x_k'] * (self.r_t**2 - y**2)
        U = self.sauer['c_star'] * (1. + self.sauer['U_a'] * x + self.sauer['U_b'] * y**2)
        return x, y, U, 0.

    def _get_mach(self, U):
        """根据能量方程由速度计算马赫数"""
        c = np.sqrt(self.chara['U2c_a'] * U ** 2 + self.chara['U2c_b'])
        return U / c

    def _get_vel(self, Ma):
        """根据能量方程由马赫数计算速度"""
        _Ma = 1. / Ma**2
        return np.sqrt(self.chara['U2c_b'] / (_Ma - self.chara['U2c_a']))

    def _get_mach_p(self, p):
        """根据能量方程由速度计算声速，并得到对应的马赫数和马赫角"""
        _, _, U, _ = p
        c = np.sqrt(self.chara['U2c_a'] * U ** 2 + self.chara['U2c_b'])
        Ma = U / c
        mu = np.arcsin(1. / Ma)
        return c, Ma, mu

    @staticmethod
    def _eqs_solver(x1, y1, x2, y2,
                    a, b, c, d, e, f):
        """求解形如下列的二元一次方程组：
        a (x - x1) + c (y - y1) = e
        b (x - x2) + d (y - y2) = f
        """
        e += a * x1 + c * y1
        f += b * x2 + d * y2
        det = a * d - b * c
        x = (e * d - f * c) / det
        y = (a * f - b * e) / det
        return x, y

    def interp_p(self, p1, p2, x, y):
        """在两点之间对二维场变量进行线性插值"""
        x1, y1, U1, theta1 = p1
        x2, y2, U2, theta2 = p2
        k = ((x2 - x1) * (x - x1) + (y2 - y1) * (y - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return x, y, (1. - k) * U1 + k * U2, (1. - k) * theta1 + k * theta2

    def x_sym_p(self, p):
        """返回关于x轴对称点的流动参数"""
        x, y, U, theta = p
        return x, -y, U, -theta

    def chara_line_p(self, p1, p2, n=1):
        """使用定常二维超声速无旋流动的特征线法计算流场内部点"""
        x1, y1, U1, theta1 = p1
        _, _, mu1 = self._get_mach_p(p1)
        x2, y2, U2, theta2 = p2
        _, _, mu2 = self._get_mach_p(p2)

        # 预估步
        x3, y3 = self._eqs_solver(x1, y1, x2, y2,
                                  np.tan(theta1 - mu1), np.tan(theta2 + mu2), -1., -1., 0., 0.)
        if self.delta == 0:
            e = f = 0.
        else:
            e = np.sin(mu1)**2 * np.sin(theta1) / (np.cos(mu1) * np.cos(theta1 - mu1)) * (x3 - x1) / y1
            if y2 != 0:
                f = np.sin(mu2) ** 2 * np.sin(theta2) / (np.cos(mu2) * np.cos(theta2 + mu2)) * (x3 - x2) / y2
            else:
                f = np.sin(mu2) ** 2 * np.sin(theta1) / np.cos(mu2) * np.cos(theta2 + mu2) * (x3 - x2) / y1
        U3, theta3 = self._eqs_solver(U1, theta1, U2, theta2,
                                      1. / U1, 1. / U2, np.tan(mu1), -np.tan(mu2),
                                      e, f)
        p3 = x3, y3, U3, theta3
        _, _, mu3 = self._get_mach_p(p3)

        # （迭代）校正步
        if self.delta == 0:
            e = f = lambda x, y, theta, mu: 0.
        else:
            e = lambda x, y, theta, mu: \
                np.sin(mu) ** 2 * np.sin(theta) / (np.cos(mu) * np.cos(theta - mu)) * (x - x1) / y
            if y2 != 0:
                f = lambda x, y, theta, mu: \
                    np.sin(mu) ** 2 * np.sin(theta) / (np.cos(mu) * np.cos(theta + mu)) * (x - x2) / y
            else:
                f = lambda x, y, theta, mu: \
                    np.sin(mu) ** 2 * np.sin(theta) / (np.cos(mu) * np.cos(theta + mu)) * (x - x2) / y
        i = 0
        while i < n:
            x3, y3 = self._eqs_solver(x1, y1, x2, y2,
                                      np.tan(0.5 * (theta1 + theta3 - mu1 - mu3)),
                                      np.tan(0.5 * (theta2 + theta3 + mu2 + mu3)),
                                      -1., -1., 0., 0.)
            U3, theta3 = self._eqs_solver(U1, theta1, U2, theta2,
                                          2. / (U1 + U3), 2. / (U2 + U3),
                                          np.tan(0.5 * (mu1 + mu3)), -np.tan(0.5 * (mu2 + mu3)),
                                          e(x3, 0.5 * (y1 + y3), 0.5 * (theta1 + theta3), 0.5 * (mu1 + mu3)),
                                          f(x3, 0.5 * (y2 + y3), 0.5 * (theta2 + theta3), 0.5 * (mu2 + mu3)))
            p3 = x3, y3, U3, theta3
            _, _, mu3 = self._get_mach_p(p3)
            i += 1
        return p3

    def chara_line_wall(self, p1, p2, p3, n=1):
        """使用定常二维超声速无旋流动的特征线法计算流场壁面点，
        对p3壁面点采用逆处理，其坐标和速度方向已知（速度与壁面相切），速度大小为待求量"""
        x1, y1, U1, theta1 = p1
        _, _, mu1 = self._get_mach_p(p1)
        x2, y2, U2, theta2 = p2
        _, _, mu2 = self._get_mach_p(p2)
        x3, y3, _, theta3 = p3

        # 预估步
        U3 = U1
        mu3 = mu1

        x4, y4 = self._eqs_solver(x2, y2, x3, y3,
                                  np.tan(theta2 - mu2), np.tan(theta3 + mu3), -1., -1., 0., 0.)
        p4 = self.interp_p(p1, p2, x4, y4)
        _, _, U4, theta4 = p4
        _, _, mu4 = self._get_mach_p(p4)

        U3 = U4 + U4 * (np.tan(mu4) * (theta3 - theta4) + 0. if self.delta == 0 else
                        np.sin(mu4) ** 2 * np.sin(theta4) / (np.cos(mu4) * np.cos(theta4 + mu4)) * (x3 - x4) / y4)
        p3 = x3, y3, U3, theta3
        _, _, mu3 = self._get_mach_p(p3)

        x4, y4 = self._eqs_solver(x2, y2, x3, y3,
                                  np.tan(theta2 - mu2), np.tan(theta3 + mu3), -1., -1., 0., 0.)
        p4 = self.interp_p(p1, p2, x4, y4)
        _, _, U4, theta4 = p4
        _, _, mu4 = self._get_mach_p(p4)

        # （迭代）校正步
        i = 0
        while i < n:
            x4, y4 = self._eqs_solver(x2, y2, x3, y3,
                                      np.tan(0.5 * (theta2 + theta4 - mu2 - mu4)),
                                      np.tan(0.5 * (theta3 + theta4 + mu3 + mu4)),
                                      -1., -1., 0., 0.)
            p4 = self.interp_p(p1, p2, x4, y4)
            _, _, U4, theta4 = p4
            _, _, mu4 = self._get_mach_p(p4)

            _mu = 0.5 * (mu3 + mu4)
            _theta = 0.5 * (theta3 + theta4)
            U3 = U4 + 0.5 * (U3 + U4) * (np.tan(_mu) * (theta3 - theta4) + 0. if self.delta == 0 else
                            np.sin(_mu) ** 2 * np.sin(_theta) / (np.cos(_mu) * np.cos(_theta + _mu)) *
                            (x3 - x4) / (0.5 * (y3 + y4)))
            p3 = x3, y3, U3, theta3
            _, _, mu3 = self._get_mach_p(p3)
            i += 1
        return p3

    def net_throat(self, n=10):
        """求解喷管喉道起始线影响区域的流场"""
        self.field_param['throat_n'] = n
        field = [[self._throat_line(y)] for y in np.linspace(self.r_t, 0, n + 1)]
        i = 0
        while i < n:
            j = 0
            sub_n = n - i
            while j < sub_n:
                field[j].append(self.chara_line_p(field[j][-1], field[j + 1][-1]))
                j += 1
            j = 0
            sub_n = n - i - 1
            while j < sub_n:
                field[j].append(self.chara_line_p(field[j][-1], field[j + 1][-1]))
                j += 1
            field[sub_n].append(self.chara_line_p(field[sub_n][-1], self.x_sym_p(field[sub_n][-1])))
            i += 1
        self.field['throat'] = field
        return field

    def net_kernel(self, dtheta=0.2, n=100):
        """求解喷管喉道型面影响区域的流场（其中参数theta为角度制）"""
        self.field_param['kernel_n'] = n
        self.field_param['kernel_dtheta'] = dtheta
        field = [self.field['throat'][0]] + \
                [[(self.rho_t * np.cos(theta), self.sauer['L'] + self.rho_t * np.sin(theta), None, theta + np.pi / 2)]
                 for theta in np.linspace(-90 + dtheta, -90 + dtheta * n, n) * np.pi / 180]
        i = 0
        while i < n:
            j = 1
            while True:
                # 计算喷管喉道膨胀型线壁面点的速度
                field[i + 1][0] = self.chara_line_wall(field[i][0], field[i][j], field[i + 1][0])
                # 计算以壁面点起始的右特征线上的第一个点
                p_near = self.chara_line_p(field[i + 1][0], field[i][j])
                j += 1
                # 若上游两支特征线的交点在壁面外，则重新计算
                if p_near[0] ** 2 + (self.sauer['L'] - p_near[1]) ** 2 > self.rho_t**2:
                    field[i + 1].append(p_near)
                    break
            while j < len(field[i]):
                field[i + 1].append(self.chara_line_p(field[i + 1][-1], field[i][j]))
                j += 1
            field[i + 1].append(self.chara_line_p(field[i + 1][-1], self.x_sym_p(field[i + 1][-1])))
            if self.field_param['kernel_thetaB_max'] == 0 and np.isnan(field[i + 1][-1][2]):
                # 对于靠近喷管出口的流场，按特征线法计算出的流场速度可能超过等熵流的最大速度
                print(f"Flow limiting expansion reached at {dtheta * i:.1f}° "
                      f"(U_max: {self.chara['U_max']:.0f} m/s)", file=sys.stderr)
                self.field_param['kernel_thetaB_max'] = dtheta * i
            i += 1
        if self.field_param['kernel_thetaB_max'] == 0:
            self.field_param['kernel_thetaB_max'] = dtheta * n
        self.field['kernel'] = field
        return field

    def net_kernel_i(self, theta):
        """求解喷管喉道型面影响区域中特定角度处的流场（其中参数theta为角度制）"""
        ind = theta / self.field_param['kernel_dtheta']
        if ind >= self.field_param['kernel_n'] + 1 or ind < 0:
            raise ValueError("Parameter theta is out of range.")
        if ind == int(ind):
            # 若当前theta角处的特征线已经计算过，则直接返回已有结果，否则根据上一条特征线重新计算
            return self.field['kernel'][int(ind)]
        else:
            last_line = self.field['kernel'][int(ind)]
            theta = (- 90 + theta) * np.pi / 180
            line = [(self.rho_t * np.cos(theta),
                     self.sauer['L'] + self.rho_t * np.sin(theta),
                     None, theta + np.pi / 2)]
            i = 1
            while True:
                # 计算喷管喉道膨胀型线壁面点的速度
                line[0] = self.chara_line_wall(last_line[0], last_line[i], line[0])
                # 计算以壁面点起始的右特征线上的第一个点
                p_near = self.chara_line_p(line[0], last_line[i])
                i += 1
                # 若上游两支特征线的交点在壁面外，则重新计算
                if p_near[0] ** 2 + (self.sauer['L'] - p_near[1]) ** 2 > self.rho_t ** 2:
                    line.append(p_near)
                    break
            while i < len(last_line):
                line.append(self.chara_line_p(line[-1], last_line[i]))
                i += 1
            line.append(self.chara_line_p(line[-1], self.x_sym_p(line[-1])))
            return line

    def _ratio_Ma2W(self, Ma):
        """根据一维等熵关系，由马赫数比值计算速度的比值"""
        return 1. / np.sqrt(self.gamma - 1 + 2. / Ma ** 2)

    def _ratio_Ma2rhoW(self, Ma):
        """根据一维等熵关系，由马赫数比值计算单位面积质量流率（比流量密度）的比值"""
        exp = -0.5 * (self.gamma + 1) / (self.gamma - 1)
        return Ma * (2 / (self.gamma + 1) * (1 + 0.5 * (self.gamma - 1) * Ma ** 2)) ** exp

    def _ratio_Ma2rhoWW(self, Ma):
        """根据一维等熵关系，由马赫数比值计算单位面积动量流率的比值"""
        exp = - self.gamma / (self.gamma - 1)
        MaMa = Ma ** 2
        return MaMa * (1 + 0.5 * (self.gamma - 1) * MaMa) ** exp

    def _Ma2epsilon(self, Ma):
        """根据一维等熵关系，由喷管出口设计马赫数计算喷管面积比"""
        a = 2 / (self.gamma + 1)
        b = (self.gamma - 1) / 2
        c = (self.gamma + 1) / (2 * (self.gamma - 1))
        return (a * (1 + b * Ma ** 2)) ** c / Ma

    def Ma2thetaE(self, Ma):
        """根据变分极值条件，在喷管型面端点E点处由马赫数计算速度角（假设背压为0）"""
        alpha = np.arcsin(1. / Ma)
        eq_value = 2. / (self.gamma * Ma**2 * np.tan(alpha))
        return 0.5 * np.arcsin(eq_value)

    def Ma_theta(self, Ma, theta):
        """根据变分极值条件，得到控制面（即喷管内流场的最后一条左特征线）上各点应满足的不变量之一"""
        alpha = np.arcsin(1. / Ma)
        return self._ratio_Ma2W(Ma) * np.cos(theta - alpha) / np.cos(alpha)

    def y_Ma_theta(self, y, Ma, theta):
        """根据变分极值条件，得到控制面（即喷管内流场的最后一条左特征线）上各点应满足的不变量之二"""
        alpha = np.arcsin(1. / Ma)
        return y * Ma ** 2 * self._ratio_Ma2rhoWW(Ma) * np.sin(theta)**2 * np.tan(alpha)

    def control_face(self, Ma_E, n=10):
        """给定喷管出口端点E的马赫数，在已计算的喷管流场内寻找并求解控制面"""
        theta_E = self.Ma2thetaE(Ma_E)
        line_bd = np.array([])
        line_de = np.array([])

        def theta2mf(theta_i):
            nonlocal line_bd, line_de
            # 寻找从喉道圆弧型面上任意一B'点出发的右特征线B'D'和喷管内最后一条左特征线（控制面）CE的交点D'
            line_bd = np.array(self.net_kernel_i(theta_i)).T
            line_bd_nan = np.isnan(line_bd[0])
            if line_bd_nan.any():
                # 若特征线B'D'末端出现nan值，则截去
                line_bd = line_bd[: , :np.where(line_bd_nan)[0][0]]
            inp_bd = {
                'x': line_bd[0],
                'y': interp1d(line_bd[0], line_bd[1], kind='cubic'),
                'Ma': interp1d(line_bd[0], self._get_mach(line_bd[2]), kind='cubic'),
                'theta': interp1d(line_bd[0], line_bd[3], kind='cubic')
            }
            # func_plot(lambda x: self.Ma_theta(inp_bd['Ma'](x), inp_bd['theta'](x)) - self.Ma_theta(Ma_E, theta_E),
            #          x0=inp_bd['x'][0], x1=inp_bd['x'][-1])
            try:
                solver = root_scalar(lambda x:
                                     self.Ma_theta(inp_bd['Ma'](x), inp_bd['theta'](x)) - self.Ma_theta(Ma_E, theta_E),
                                     bracket=[inp_bd['x'][0], inp_bd['x'][-1]])
            except ValueError:
                # 若方程无根即无交点（不满足变分条件一），则返回B'D'和D'E两面的质量流率残差为inf
                return float('inf')

            x_D = solver.root
            y_D = inp_bd['y'](x_D)
            Ma_D = inp_bd['Ma'](x_D)
            theta_D = inp_bd['theta'](x_D)
            y_E = self.y_Ma_theta(y_D, Ma_D, theta_D) / self.y_Ma_theta(1, Ma_E, theta_E)
            # 根据得到的D'点位置对line_bd数组进行截取
            cut_i = 0
            while cut_i < len(line_bd[0]):
                if line_bd[0, cut_i] >= x_D:
                    break
                cut_i += 1
            line_bd = np.hstack([line_bd[:, : cut_i],
                                 np.array([[x_D, y_D, self._get_vel(Ma_D), theta_D]]).T])

            # 计算满足函数Ma_theta的所有点（即D'E线上的流动参数）
            theta = np.linspace(theta_D, theta_E, n + 1)
            Ma = np.array([root_scalar(lambda x: self.Ma_theta(x, theta_i) - self.Ma_theta(Ma_E, theta_E),
                                       bracket=[1 + 0.1 * Ma_E, 2 * Ma_E]).root
                           for theta_i in theta])
            try:
                y = np.array([root_scalar(lambda x:
                                          self.y_Ma_theta(x, Ma_i, theta_i) - self.y_Ma_theta(y_E, Ma_E, theta_E),
                                          bracket=[0, 1.1 * y_E]).root
                              for Ma_i, theta_i in zip(Ma, theta)])
            except ValueError:
                # 若在喷管范围内y坐标无解（不满足变分条件二），同样也返回inf
                return float('inf')
            line_de = np.vstack([np.zeros(theta.shape), y, self._get_vel(Ma), theta])
            inp_de = {
                'y': y,
                'Ma': interp1d(y, Ma, kind='cubic'),
                'theta': interp1d(y, theta, kind='cubic')
            }

            # 积分计算B'D'和D'E线上的质量流率
            def func_1(x):
                Ma = inp_bd['Ma'](x)
                alpha = np.arcsin(1. / Ma)
                return self._ratio_Ma2rhoW(Ma) * inp_bd['y'](x) / (Ma * np.cos(inp_bd['theta'](x) - alpha))

            def func_2(y):
                Ma = inp_de['Ma'](y)
                alpha = np.arcsin(1. / Ma)
                return self._ratio_Ma2rhoW(Ma) * y / (Ma * np.sin(inp_de['theta'](y) + alpha))

            # 根据质量流率相等来修正theta角（即B'点和D'点的位置）
            mf_error = quad(func_1, line_bd[0, 0], x_D)[0] - quad(func_2, y_D, y_E)[0]
            return mf_error

        # 计算满足变分条件和质量守恒的B点位置
        last_mf = float('inf')
        solver = None
        for theta_i in np.linspace(0, self.field_param['kernel_n'] * self.field_param['kernel_dtheta'],
                                   self.field_param['kernel_n'] + 1):
            mf = theta2mf(theta_i)
            if last_mf * mf < 0:
                solver = root_scalar(theta2mf,
                                     bracket=[theta_i - self.field_param['kernel_dtheta'], theta_i])
                break
            else:
                last_mf = mf
        if not solver:
            raise ValueError("Cannot find a control face in current net, "
                             "please consider recalculating to get a larger 'kernel' field.")

        # 由于DE线为左特征线，根据相容关系可以求出每点的x坐标
        x = [line_bd[0, -1]]
        for i in range(len(line_de[0]) - 1):
            mu = 0.5 * (self._get_mach_p(line_de[:, i + 1])[2] + self._get_mach_p(line_de[:, i])[2])
            theta = 0.5 * (line_de[3, i + 1] + line_de[3, i])
            x.append(x[-1] + (line_de[1, i + 1] - line_de[1, i]) / np.tan(theta + mu))
        line_de[0] = np.array(x)

        return solver.root, line_bd, line_de

    def net_wall(self, Ma_E=None, epsilon=None, n=10):
        """给定出口马赫数或喷管膨胀比，计算喷管扩张段壁面附近区域的流场并得到壁面型线"""
        self.field_param['wall_n'] = n
        if not isinstance(Ma_E, type(None)):
            pass
        elif not isinstance(epsilon, type(None)):
            print("Epsilon interp：\n\tMa_E\tepsilon")
            samples = []
            Ma2eps = lambda x: (self.control_face(x, n)[2][1][-1] / self.r_t) ** 2
            for Ma_E_i in np.linspace(1, 12, 12):    # 马赫数范围1~12
                try:
                    eps = Ma2eps(Ma_E_i)
                except ValueError:
                    print(f"    {Ma_E_i:.1f}\t\tNone")
                else:
                    samples.append([Ma_E_i, eps])
                    print(f"    {Ma_E_i:.1f}\t\t{eps:.2f}")
                    if eps >= epsilon:
                        break
            try:
                # 对马赫数Ma_E和膨胀比epsilon进行插值
                Ma_1 = (2 * samples[-2][0] + samples[-1][0]) / 3
                Ma_2 = (samples[-2][0] + 2 * samples[-1][0]) / 3
                samples = np.array([samples[-2],
                                    [Ma_1, Ma2eps(Ma_1)],
                                    [Ma_2, Ma2eps(Ma_2)],
                                    samples[-1]]).T
                interp = interp1d(*samples, kind='cubic')
                solver = root_scalar(lambda x: interp(x) - epsilon, bracket=[samples[0, 0], samples[0, -1]])
            except IndexError or ValueError:
                raise ValueError("Cannot find corresponding epsilon under current condition.")
            Ma_E = solver.root
        else:
            raise ValueError("Either Ma_E or epsilon must be specified.")

        # 根据直接给定或插值计算得到的喷管出口端点马赫数Ma_E计算控制面
        theta_E = self.Ma2thetaE(Ma_E)
        print(f"Bell nozzle config:\n\tMa_E = {Ma_E:.2f}\n\tepsilon_0 = {self._Ma2epsilon(Ma_E):.2f}\n",
              f"\ttheta_E = {180 * theta_E / np.pi:.3f}°", sep='')
        theta_B, line_bd, line_de = self.control_face(Ma_E, n)
        self.field_param['kernel_thetaB'] = theta_B
        print(f"\ttheta_B = {theta_B:.3f}° (max {self.field_param['kernel_thetaB_max']}°)\n",
              f"\tepsilon = {(line_de[1, -1] / self.r_t) ** 2:.2f}\n",
              f"\tdiv_r = {line_de[1, -1]:.3e}\n\tdiv_l = {line_de[0, -1]:.3e}", sep='')

        # 使用特征线法计算BD线和DE线所包围的流场区域
        field = [[tuple(p)] for p in line_bd[: , : -1].T.tolist()] + [line_de.T.tolist()]
        i = len(line_bd[0]) - 2
        while i > 0:
            j = 1
            while j < len(line_de[0]):
                field[i].append(self.chara_line_p(field[i + 1][j], field[i][-1]))
                j += 1
            i -= 1
        self.field['wall'] = field

        # 由特征线流场计算经过BD的流线，即为喷管扩张段型面
        i = 1
        wall_p = [field[0][0]]
        while i < len(field) - 1:
            line = np.array(field[i]).T
            inp = {
                'x': line[0],
                'y': interp1d(line[0], line[1], kind='cubic', fill_value='extrapolate'),
                'U': interp1d(line[0], line[2], kind='cubic'),
                'theta': interp1d(line[0], line[3], kind='cubic')
            }
            try:
                # 预估步
                x = root_scalar(lambda x: np.tan(wall_p[-1][3]) * (x - wall_p[-1][0]) + wall_p[-1][1] - inp['y'](x),
                                bracket=[inp['x'][0], inp['x'][-1]]).root
                # 迭代步
                theta = 0.5 * (wall_p[-1][3] + inp['theta'](x))
                x = root_scalar(lambda x: np.tan(theta) * (x - wall_p[-1][0]) + wall_p[-1][1] - inp['y'](x),
                                bracket=[inp['x'][0], inp['x'][-1]]).root
                wall_p.append([x, inp['y'](x), inp['U'](x), inp['theta'](x)])
            except ValueError:
                # 若扩张比很大，最后一个（或多个）型面点可能会因积分误差过大而舍去
                break
            i += 1
        wall_p.append(field[-1][-1])
        self.wall_p = np.vstack(wall_p)

        print(f"\twall_net = {n}x{len(line_bd[0]) - 1} shape, {len(wall_p)} points")

    def derive(self, Ma_e=None, epsilon=None, throat_theta=30):
        self.net_throat(n=10)
        self.net_kernel(dtheta=0.2, n=5*throat_theta)
        self.net_wall(Ma_E=Ma_e, epsilon=epsilon, n=40)
        # 按比例确定其余尺寸
        self.L_c = self.r_t
        self.theta_c = np.pi * self.field_param['kernel_thetaB'] / 180
        R_e = self.wall_p[-1][1]
        self.domain_w = 18 * R_e
        self.domain_h = 6 * R_e
        self.nozzle_w = 0 * R_e
        self.nozzle_h = 1.2 * R_e

    def generate(self, size=0.1, factor=1.):
        points = {}

        interp = ThirdPoly(self.wall_p[:, 0], self.wall_p[:, 1], np.tan(self.wall_p[:, 3]))
        wall_func = lambda x: (x, interp(x))
        points['wall_div'] = func_space_2d(wall_func,
                                           self.wall_p[-1, 0],
                                           self.wall_p[0, 0],
                                           size * factor, factor=1. / factor)

        # 根据喷管扩张段划分获取其他区域尺寸
        ls_small = np.linalg.norm(points['wall_div'][-2] - points['wall_div'][-1])
        ls_middle = 0.5 * ls_small + 0.5 * np.linalg.norm(points['wall_div'][1] - points['wall_div'][0])
        ls_large = 4 * np.linalg.norm(points['wall_div'][1] - points['wall_div'][0])

        wall_func = lambda t: (np.cos(t) * self.rho_t,
                               self.r_t + self.rho_t + np.sin(t) * self.rho_t)
        points['wall_throat'] = func_space_2d(
            wall_func,
            np.pi * (-90 + self.field_param['kernel_thetaB']) / 180,
            -0.5 * np.pi - self.theta_c,
            ls_small)

        points['wall_inlet'] = offset_space_nd(
            points['wall_throat'][-1],
            points['wall_throat'][-1] - np.array([self.L_c, 0.]),
            ls_small)

        points['inlet'] = offset_space_nd(
            points['wall_inlet'][-1],
            np.array([points['wall_inlet'][-1][0], 0.]),
            ls_small)

        points['axis'] = offset_space_nd(
            points['inlet'][-1],
            np.array([self.domain_w, 0.]),
            ls_small, factor=ls_large / ls_small)

        points['outlet_right'] = offset_space_nd(
            np.array([self.domain_w, 0.]),
            np.array([self.domain_w, self.domain_h]),
            ls_large)

        points['outlet_upper'] = offset_space_nd(
            np.array([self.domain_w, self.domain_h]),
            np.array([- self.nozzle_w, self.domain_h]),
            ls_large)

        points['outlet_left'] = offset_space_nd(
            np.array([- self.nozzle_w, self.domain_h]),
            np.array([- self.nozzle_w, self.nozzle_h]),
            ls_large)

        points['wall_outer'] = offset_space_nd(
            np.array([- self.nozzle_w, self.nozzle_h]),
            np.array([points['wall_div'][0][0], self.nozzle_h]),
            ls_large, factor=ls_middle / ls_large)

        points['wall_end'] = offset_space_nd(
            points['wall_outer'][-1],
            points['wall_div'][0],
            ls_middle)

        self.profile = points
        self.points = np.vstack([points['wall_outer'][: -1],
                                 points['wall_end'][: -1],
                                 points['wall_div'][: -1],
                                 points['wall_throat'][: -1],
                                 points['wall_inlet'][: -1],
                                 points['inlet'][: -1],
                                 points['axis'][: -1],
                                 points['outlet_right'][: -1],
                                 points['outlet_upper'][: -1],
                                 points['outlet_left'][: -1]])
        print(f"Line division info:\n\tls_small = {ls_small: .3e}\n\tls_middle = {ls_middle: .3e}\n\t",
              f"ls_large = {ls_large:.3e}\n\tn_total = {len(self.points)}", sep='')

    def get_profile(self):
        n_wall = sum([len(self.profile[part]) - 1 for part in
                      ['wall_outer', 'wall_end', 'wall_div', 'wall_throat', 'wall_inlet']])
        n_inlet = len(self.profile['inlet']) - 1
        n_axis = len(self.profile['axis']) - 1
        n_outlet = sum([len(self.profile[part]) - 1 for part in
                        ['outlet_right', 'outlet_upper', 'outlet_left']])
        ind = np.cumsum([n_wall, n_inlet, n_axis, n_outlet])
        ind_pair = [0] + np.repeat(ind[: -1], 2).tolist() + [ind[-1]]
        tag = dict([(name, [ind_pair[2 * i], ind_pair[2 * i + 1]])
                    for i, name in enumerate(['shell', 'inlet', 'axis', 'outlet'])])
        return self.points, tag

    def plot_field(self, field=None):
        """以散点图的形式绘制流场数据点"""
        if isinstance(field, type(None)):
            field = []
            if 'throat' in self.field:
                field += self.field['throat']
                if 'kernel' in self.field:
                    if self.field_param['kernel_thetaB'] != 0:
                        ind = int(self.field_param['kernel_thetaB'] / self.field_param['kernel_dtheta'])
                    else:
                        ind = self.field_param['kernel_n'] + 1
                    field += self.field['kernel'][: ind]
                    if 'wall' in self.field:
                        field += self.field['wall']
        if not field:
            print("Please calculate flow field first before plotting.")
            return
        points = []
        for line in field:
            points.extend(line)
        x, y, U, theta = np.array(points).T
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=10, marker='o', c=U)
        #ax.contourf(x, y, U, 150, cmap='RdBu_r', linestyles='dashed', zorder=1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.show()
        plt.pause(0.01)

    def plot_profile(self):
        """绘制喷管型面"""
        if isinstance(self.profile, type(None)):
            print("Please generate plug profile first before plotting.")
            return
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle("Profile of Optimal Bell Nozzle", fontsize=22)
        ax.fill(*self.points.T, color='gray', alpha=0.5)
        for tag, line in self.profile.items():
            if tag == 'inlet':
                ax.plot(*line.T, '.-g', markersize=1, lw=1)
            elif tag.startswith('outlet'):
                ax.plot(*line.T, '.-b', markersize=1, lw=1)
            else:
                ax.plot(*line.T, '.-k', markersize=3, lw=1)
        ax.plot(ax.get_xlim(), [0, 0], '--y', lw=2)
        ax.set_xlabel("x", fontsize=18)
        ax.set_ylabel("y", fontsize=18)
        ax.set_xticks(np.arange(np.ceil(ax.get_xlim()[0]), ax.get_xlim()[1], 1))
        ax.set_ylim(ax.get_ylim()[0], 1.5 * ax.get_ylim()[1])
        ax.set_yticks(np.arange(np.ceil(ax.get_ylim()[0]), ax.get_ylim()[1], 1))
        ax.grid()
        fig.tight_layout()
        fig.show()
        plt.pause(0.01)


if __name__ == '__main__':
    nozzle = CharacteristicsNozzle(r_t=0.2, rho_t=2, axial_sym=True)
    nozzle.derive(epsilon=16, throat_theta=40)
    nozzle.plot_field()
    nozzle.generate(size=0.001, factor=3)
    nozzle.plot_profile()
    profile, tag = nozzle.get_profile()
    profile_to_msh(profile, tag, lc=0., planner=True, save_path='bell.bdf')
    A_inlet = np.pi*profile[tag['inlet'][0]][1]**2


