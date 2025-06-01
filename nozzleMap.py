"""



"""


import numpy as np
from matplotlib import pyplot as plt

__all__ = ['point_transform', 'bezier_profile', 'plot_profile']


def point_transform(normalized_points, l_max=10, l_min=2, h_max=5):
    """
    将网络输出的归一化三元数（取值范围(0, 1)，按行排列）转换为喷管扩张段的贝塞尔曲线控制点，
    每个点包含控制点的x、y坐标和局部曲率，目的是生成合理的喷管型面

    思路参考：https://doi.org/10.1016/j.jcp.2020.110080
    """
    n = len(normalized_points)
    dx, dy, s = normalized_points.T
    dl = max(l_max * dx.mean(), l_min) / n
    x = (np.arange(n) + dx) * dl
    dh = h_max / n
    y = np.cumsum(dy) * dh
    e = s    # e = s / dl
    return np.vstack([x, y, e]).T


def linear_interp(p1, p2, n=10):
    """
    两点线性插值
    """
    simples = np.zeros((n, len(p1)))
    i = 0
    while i < n:
        t = i / n
        simples[i] = (1 - t) * p1 + t * p2
        i += 1
    return simples


def bezier_interp(p1, p2, p3, p4, n=10):
    """
    四点三阶贝塞尔曲线插值
    """
    simples = np.zeros((n, len(p1)))
    i = 0
    while i < n:
        t = i / n
        w1 = (1 - t)**3
        w2 = 3 * (1 - t)**2 * t
        w3 = 3 * (1 - t) * t**2
        w4 = t**3
        simples[i] = w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4
        i += 1
    return simples


def bezier_profile(div_points, throat_point=(0, 1), inlet_point=(-2, 2), sub_n=10):
    """
    根据定位点生成喷管型面，其中扩张段使用分段三阶贝塞尔插值
    """
    k = 0.3    # 收敛段斜率平滑控制
    alpha = 0.5    # 扩张段斜率平滑控制
    axis_point_left = np.array((inlet_point[0], 0))
    inlet_point = np.array(inlet_point)
    throat_point = np.array(throat_point)
    div_e = div_points[: , 2]
    div_points = np.vstack([throat_point,
                            div_points[: , : 2] + throat_point,
                            2 * div_points[-1, : 2] - div_points[-2, : 2] + throat_point])
    outlet_point = div_points[-2]
    axis_point_right = np.array((outlet_point[0], 0))
    simples_block = []
    partition_tag = {}
    # 入口面插值（线性）
    simples_block.append(linear_interp(axis_point_left, inlet_point, sub_n))
    partition_tag["inlet"] = [0, sub_n]
    # 收敛段插值（贝塞尔）
    offset = np.array([k * np.linalg.norm(throat_point - inlet_point), 0])
    simples_block.append(bezier_interp(inlet_point,
                                       inlet_point + offset,
                                       throat_point - offset,
                                       throat_point, sub_n))
    # 扩张段插值（分段贝塞尔）
    offset_a = np.array([k * np.linalg.norm(div_points[1] - div_points[0]), 0])
    for i, point in enumerate(div_points[1: -1]):
        offset_b = div_e[i] * \
                   ((1 - alpha) * (point - div_points[i]) + alpha * (div_points[i + 2] - point))
        simples_block.append(bezier_interp(div_points[i],
                                           div_points[i] + offset_a,
                                           point - offset_b,
                                           point, sub_n))
        offset_a = offset_b
    partition_tag["wall"] = [sub_n, len(div_points) * sub_n]
    # 出口面插值（线性）
    simples_block.append(linear_interp(outlet_point, axis_point_right, sub_n))
    partition_tag["outlet"] = [len(div_points) * sub_n, (len(div_points) + 1) * sub_n]
    # 对称轴插值（线性）
    simples_block.append(linear_interp(axis_point_right, axis_point_left, sub_n))
    partition_tag["axis"] = [(len(div_points) + 1) * sub_n, (len(div_points) + 2) * sub_n]
    control_points = np.vstack([axis_point_left, inlet_point, div_points[: -1], axis_point_right])
    return np.vstack(simples_block), control_points, partition_tag


def plot_profile(profile_points, control_points=None):
    """
    绘制喷管型面
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Profile of Parameterized Nozzle", fontsize=22)
    ax.fill_between(*profile_points.T, facecolor='gray', alpha=0.5)
    ax.plot(*profile_points.T, '.-k', markersize=0.8)
    ax.plot(ax.get_xlim(), [0, 0], '--y', lw=2)
    if not isinstance(control_points, type(None)):
        ax.plot(*control_points.T, 'or')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xticks(np.arange(np.ceil(ax.get_xlim()[0]), ax.get_xlim()[1], 1))
    ax.set_ylim(ax.get_ylim()[0], 1.5 * ax.get_ylim()[1])
    ax.set_yticks(np.arange(np.ceil(ax.get_ylim()[0]), ax.get_ylim()[1], 1))
    ax.grid()
    fig.tight_layout()
    fig.show()


if __name__ == '__main__':
    net_output = np.array([[0.9, 0.4, 0.0],
                           [0.1, 0.2, 0.0],
                           [0.5, 0.1, 0.0]])
    points = point_transform(net_output)
    profile, control, _ = bezier_profile(points, sub_n=50)
    plot_profile(profile, control)


