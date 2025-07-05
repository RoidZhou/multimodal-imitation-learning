import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D

def plot_collect_data_trajectory(trajectories, colors, labels):
    fig = plt.figure(figsize=(12, 8), facecolor=None)
    ax = fig.add_subplot(111, projection='3d')

    # 限制轨迹数量不超过颜色和标签长度
    max_trajectories = min(len(trajectories), len(colors), len(labels))
    print(f"Number of trajectories to plot: {max_trajectories}")

    # 绘制每组轨迹
    for idx in range(max_trajectories):
        data = trajectories[idx]
        # 验证数据格式
        print(f"Trajectory {idx} shape: {data.shape}")
        # 绘制轨迹线
        ax.plot(
            data[:, 0], data[:, 1], data[:, 2], linestyle='-',
            color=colors[idx], label=labels[idx], linewidth=2
        )

    # 设置图形属性
    ax.set_xlabel('X (m)', labelpad=19, fontsize=14)
    ax.set_ylabel('Y (m)', labelpad=19, fontsize=14)
    ax.set_zlabel('Z (m)', labelpad=19, fontsize=14)
    ax.view_init(elev=30, azim=45)  # 固定视角
    ax.tick_params(axis='x', pad=10, labelsize=12)
    ax.tick_params(axis='y', pad=10, labelsize=12)
    ax.tick_params(axis='z', pad=10, labelsize=12)
    ax.xaxis.pane.set_facecolor('none')
    ax.yaxis.pane.set_facecolor('none')
    ax.zaxis.pane.set_facecolor('none')
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 加载所有轨迹文件
    action_circle_files = ['action_HDQN_circle_align.npy', 'action_HDQN_circle_contact.npy']  # 假设三个文件
    action_triangle_files = ['action_HDQN_triangle_align.npy', 'action_HDQN_triangle_contact.npy']  # 假设三个文件
    action_square_files = ['action_HDQN_square_align.npy', 'action_HDQN_square_contact.npy']  # 假设三个文件
    trajectories = []
    for f in action_circle_files:
        try:
            trajectories.append(np.load(f))
        except FileNotFoundError:
            print(f"Warning: File {f} not found, skipping.")
            continue

    for f in action_triangle_files:
        try:
            trajectories.append(np.load(f))
        except FileNotFoundError:
            print(f"Warning: File {f} not found, skipping.")
            continue

    for f in action_square_files:
        try:
            trajectories.append(np.load(f))
        except FileNotFoundError:
            print(f"Warning: File {f} not found, skipping.")
            continue
    # 为每组轨迹设置颜色和标签
    colors = ['r', 'g', 'b']
    labels = ['Trajectory 1', 'Trajectory 2', 'Trajectory 3']

    if not trajectories:
        print("Error: No valid trajectory files loaded.")
    else:
        plot_collect_data_trajectory(trajectories, colors, labels)
