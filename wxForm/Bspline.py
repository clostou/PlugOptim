import numpy as np
import wx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

from cfd_toolbox.vec import CubicSpline
from plugDesign import ParamProfile2, transform_points
from WinBSpline import MainWindowBase


class DraggablePoints:

    def __init__(self, axes, x, y, callback=None):
        self.axes = axes
        self.canvas = axes.figure.canvas
        self.points = axes.plot(x, y, 'bo', markersize=10, label='Original points', picker=True)[0]
        self.x = x
        self.y = y
        self.current_point = None
        self.callback = (lambda: self.canvas.draw()) if callback is None else callback
        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.click_event_cid = None

    def __len__(self):
        return len(self.x)

    def change_mode(self, mode=0):
        if mode or self.click_event_cid is None:  # add模式
            self.click_event_cid = self.canvas.mpl_connect('button_press_event', self.on_click)
        elif self.click_event_cid is not None:  # pick模式（默认）
            self.canvas.mpl_disconnect(self.click_event_cid)

    def on_click(self, event):
        if event.inaxes:
            self.x = np.append(self.x, event.xdata)
            self.y = np.append(self.y, event.ydata)
            self.update_plot()

    def on_pick(self, event):
        # 检查是否选中了控制点
        if event.artist == self.points:
            self.current_point = event.ind[0]

    def on_motion(self, event):
        # 拖动控制点
        if self.current_point is not None and event.inaxes:
            self.x[self.current_point] = event.xdata
            self.y[self.current_point] = event.ydata
            self.update_plot()

    def on_release(self, event):
        # 释放鼠标
        self.current_point = None

    def update_plot(self):
        # 更新控制点和曲线
        self.points.set_data(self.x, self.y)
        self.callback()

    def reset_plot(self, x, y):
        self.x = x
        self.y = y
        self.update_plot()


class MainWindow(MainWindowBase):

    def __init__(self, parent):
        super(MainWindow, self).__init__(parent)
        self.figure = plt.figure()
        # self.figure.set_facecolor("#ff2e63")
        self.canvas = FigureCanvas(self.fig_panel, -1, self.figure)
        self.fig_sizer.Add(self.canvas, 1, wx.EXPAND)
        self.axes = self.figure.add_subplot(111)

        self.n = 1
        self.theta = 0.
        self.p_start = np.array([0., 1.])
        self.p_end = np.array([1., 0.])
        self.end_point = self.axes.plot(*np.vstack([self.p_start, self.p_end]).T, 'ko', markersize=10)[0]
        self.drag_points = DraggablePoints(self.axes, [], [], callback=self.update_plot)
        self.x_tick = np.linspace(min(self.p_start[0], self.p_end[0]),
                                  max((self.p_start[0], self.p_end[0])), 100)

        self.bsp_1 = self.axes.plot([], [], 'b-.', label='BSpline-1')[0]
        self.bsp_2 = self.axes.plot([], [], 'g-.', label='BSpline-2')[0]
        self.bsp_comp = self.axes.plot([], [], 'r-', label='Composite')[0]
        self.axes.grid()
        self.axes.legend(loc='best')

        self.replot(None)
        self.fig_panel.Layout()
        self.fig_panel.Refresh()

        self.Show(True)

    def update_plot(self):
        self.n = len(self.drag_points) + 2
        self.m_textCtrl3.SetValue(str(self.n))
        if self.n > 3:
            sorted_indices = np.argsort(self.drag_points.x)
            points = np.vstack([self.p_start,
                                np.vstack([self.drag_points.x,
                                           self.drag_points.y]).T[sorted_indices],
                                self.p_end])
            # 在这里选择使用组合插值还是样条插值
            bsp = ParamProfile2(points, self.theta, _weight=0.8)
            self.bsp_1.set_data(self.x_tick, bsp.curve.sample(self.x_tick))
            self.bsp_2.set_data(self.x_tick, bsp.curve_bias.sample(self.x_tick))

            #bsp = CubicSpline(*points.T, type='half-clamped', v0=np.tan(self.theta), vn=0.).sample

            self.bsp_comp.set_data(self.x_tick, bsp(self.x_tick))
        self.canvas.draw()

    def replot(self, event):
        if event:
            n = float(self.m_textCtrl3.GetValue())
            if n < 2:
                n = 2
            elif n > 10:
                n = 10
            else:
                n = int(n)
        else:
            n = self.n
        self.drag_points.reset_plot(np.linspace(self.p_start[0], self.p_end[0], n)[1: -1],
                                    np.linspace(self.p_start[1], self.p_end[1], n)[1: -1])
        self.update_plot()
        self.change_mode_add(event)
        self.m_radioBtn1.SetValue(True)

    def change_slope(self, event):
        self.theta = int(self.m_slider2.GetValue()) / 180 * np.pi
        self.update_plot()

    def change_mode_add(self, event):
        self.m_textCtrl3.Enable(True)
        self.drag_points.change_mode(1)

    def change_mode_move(self, event):
        self.m_textCtrl3.Enable(False)
        self.drag_points.change_mode(0)


def monte_carlo(method=1, round=100_000, control_n=4, l=1, h=1, n=100, only_mono=False):
    """在高维参数空间对参数化曲线做蒙特卡洛采样，以评估参数化曲线的分布"""
    #ParamProfile2(*transform_points([0.5, 0.5, 0.5], l=l, h=h), _weight=0.8).plot()
    if method == 1:  # 非累加映射+组合插值
        by_prop = False
        dim = control_n + 1
        param_profile = lambda p, t: ParamProfile2(p, t, _weight=0.8)
    elif method == 2:  # 累加映射+组合插值
        by_prop = True
        dim = control_n + 2
        param_profile = lambda p, t: ParamProfile2(p, t, _weight=0.8)
    elif method == 3:  # 非累加映射+样条插值
        by_prop = False
        dim = control_n + 1
        param_profile = lambda p, t: CubicSpline(*p.T, type='half-clamped', v0=np.tan(t), vn=0.).sample
    elif method == 4:  # 累加映射+样条插值
        by_prop = True
        dim = control_n + 2
        param_profile = lambda p, t: CubicSpline(*p.T, type='half-clamped', v0=np.tan(t), vn=0.).sample
    else:
        return
    dx, dy = l / n, h / n
    x = np.linspace(dx / 2, l - dx / 2, n)
    x_ind = np.arange(n)
    dist = np.zeros((n, n), dtype=int)
    round_i = 0
    while round_i < round:
        _p = np.random.random(dim)  # 随机生成控制点（归一化）参数
        points, theta = transform_points(_p, l=l, h=h, by_prop=by_prop)
        y = h - param_profile(points, theta)(x)
        is_increasing = np.all(y[1:] > y[:-1])  # 判断插值曲线单调性
        if is_increasing or not only_mono:  # only_mono为True时，只统计单调曲线
            y_ind = (y // dy).astype(int)
            in_range = np.logical_and(y >= 0, y < h)
            dist[y_ind[in_range], x_ind[in_range]] += 1
        round_i += 1
    print("Accept rate: %.2f%%" % (100 * dist.sum() / (round * n)))  # 计算被显示的采样点占比
    plt.imshow(np.log(dist + 1))
    return dist


if __name__ == '__main__':
    app = wx.App(False)
    frame = MainWindow(None)
    app.MainLoop()
    #dist = monte_carlo(method=2, control_n=3, only_mono=False)


