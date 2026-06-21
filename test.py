import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 设置随机种子以确保可重复性
np.random.seed(42)

# 创建初始数据
n_points = 50
x_data = np.random.randn(n_points)
y_data = np.random.randn(n_points)
color_data = np.random.rand(n_points)  # 初始颜色数据

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 初始化散点图
scatter = ax1.scatter([], [], s=50, alpha=0.7, edgecolors='w', linewidth=0.5)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-4, 4)
ax1.set_title('动态散点图')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.grid(True, alpha=0.3)

# 初始化直方图
hist_bins = np.linspace(-4, 4, 20)
hist_counts, hist_bins, hist_patches = ax2.hist([], bins=hist_bins, alpha=0.7, color='steelblue', edgecolor='black')
ax2.set_xlim(-4, 4)
ax2.set_ylim(0, 15)
ax2.set_title('动态直方图')
ax2.set_xlabel('X值')
ax2.set_ylabel('频数')
ax2.grid(True, alpha=0.3, axis='y')

# 添加颜色条
cbar = fig.colorbar(scatter, ax=ax1)
cbar.set_label('颜色值')


def init():
    """初始化函数，设置初始空数据"""
    scatter.set_offsets(np.empty((0, 2)))  # 空点集
    scatter.set_array(np.array([]))  # 空颜色数组

    # 重置直方图
    for patch in hist_patches:
        patch.set_height(0)

    return scatter, *hist_patches


def update(frame):
    """更新函数，每一帧更新数据"""
    # 更新散点图数据 - 添加一些随机变化
    x_new = x_data + 0.1 * np.random.randn(n_points)
    y_new = y_data + 0.1 * np.random.randn(n_points)

    # 更新颜色数据 - 使用正弦函数创建动态颜色变化
    color_new = 0.5 + 0.5 * np.sin(2 * np.pi * frame / 30 + np.linspace(0, 2 * np.pi, n_points))

    # 更新散点图位置和颜色
    scatter.set_offsets(np.column_stack([x_new, y_new]))
    scatter.set_array(color_new)  # 动态修改点的颜色

    # 更新直方图数据
    hist_data = x_new  # 使用x坐标作为直方图数据

    # 计算新的直方图
    counts, bins = np.histogram(hist_data, bins=hist_bins)

    # 更新直方图柱子的高度
    for count, patch in zip(counts, hist_patches):
        patch.set_height(count)

    # 更新直方图y轴限制
    ax2.set_ylim(0, max(counts) * 1.1)

    # 更新标题显示当前帧
    ax1.set_title(f'动态散点图 - 帧 {frame}')
    ax2.set_title(f'动态直方图 - 帧 {frame}')

    return scatter, *hist_patches


# 创建动画
anim = FuncAnimation(
    fig,
    update,
    init_func=init,
    frames=100,  # 动画总帧数
    interval=100,  # 帧间隔（毫秒）
    blit=True  # 使用blitting优化性能
)

plt.tight_layout()
plt.show()

# 如果需要保存动画为GIF或视频，取消以下注释
# anim.save('scatter_hist_animation.gif', writer='pillow', fps=10)
# anim.save('scatter_hist_animation.mp4', writer='ffmpeg', fps=10)