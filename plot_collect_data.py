import numpy as np
import matplotlib.pyplot as plt
from numba.cuda.simulator.cudadrv.driver import driver
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize, LinearSegmentedColormap
import math

def euler_to_rot_matrix(roll, pitch, yaw):
    """ 欧拉角转旋转矩阵 (Z-Y-X顺序) """
    return R.from_euler('zyx', [yaw, pitch, roll], degrees=False).as_dcm()


def plot_collect_data_trajectory(trajectories,  colors, labels):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制每组轨迹
    for idx, data in enumerate(trajectories):
        # 绘制轨迹线
        ax.plot(
            data[:, 0], data[:, 1], data[:, 2],
            f'{colors[idx]}-', label=labels[idx], linewidth=1.5, alpha=0.6
        )

        # 在每个点绘制局部坐标系（调整step控制密度）
        step = 1  # 每隔step个点画一个坐标系（1=全部点）
        axis_length = 0.02  # 缩短箭头长度避免重叠

        for i in range(0, len(data), step):
            pos = data[i, :3]
            # end_orn_euler = R.from_quat(data[i, 3:7]).as_euler('xyz', degrees=True)
            roll, pitch, yaw = data[i, 3], data[i, 4], data[i, 5]
            rot_matrix = R.from_euler('zyx', [yaw, pitch, roll],  degrees=True).as_matrix()

            # 绘制X/Y/Z轴（RGB颜色）
            for axis, color in zip(rot_matrix.T, ['r', 'g', 'b']):
                ax.quiver(
                    pos[0], pos[1], pos[2],
                    axis[0], axis[1], axis[2],
                    color=color, length=axis_length,
                    arrow_length_ratio=0.1,
                    linewidth=0.5,
                    alpha=0.7,  # 半透明避免遮挡
                    label=f'{labels[idx]} Frame' if (idx == 0 and i == 0 and axis[0] > 0) else ""
                )

    # 设置图形属性
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Robot 3D Trajectory with Euler Angle Frames')
    ax.legend()
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import Normalize, LinearSegmentedColormap
def normalize_rotation_matrix(rot_matrix):
    """确保旋转矩阵的每个轴向量为单位长度"""
    return np.array([axis/np.linalg.norm(axis) for axis in rot_matrix.T]).T

def plot_phase_adjustment(trajectories):
    for idx, data in enumerate(trajectories):
        fixed_position = data[0, 0:3]
        num_points = len(data)
        time = np.linspace(0, 10, num_points)

        # 提取角度（确保yaw对应Z轴）
        roll, pitch, yaw = data[:, 3], data[:, 4], data[:, 5]
        yaw = np.zeros(len(yaw))
        # ========== 关键修改开始 ==========
        # 强制yaw旋转优先（Z轴旋转）
        # 方法：使用ZXY旋转顺序，确保yaw首先应用
        euler_sequence = 'ZXY'  # 先Z(yaw)，再X(roll)，最后Y(pitch)
        # ========== 关键修改结束 ==========

        fig = plt.figure(figsize=(12, 14))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1,1,1])
        # 可视化参数
        axis_length = 0.5
        total_height = 10
        height_scale = total_height / time[-1]
        colors = ["#1f77b4", "#9467bd"]  # 蓝->紫
        cmap = LinearSegmentedColormap.from_list("custom", colors)
        norm = Normalize(vmin=0, vmax=num_points - 1)

        for i in range(num_points):
            if roll[i] == 0.0:
                roll[:] = 90
                pitch[:] = 90
            z_pos = total_height - time[i] * height_scale
            # 计算旋转矩阵（确保yaw对应Z轴）
            rot = R.from_euler(
                euler_sequence,
                [yaw[i], roll[i], pitch[i]],  # 顺序对应euler_sequence
                degrees=True
            )
            rot_matrix = rot.as_matrix()
            rot_matrix = normalize_rotation_matrix(rot_matrix)
            # 验证Z轴方向（可选）
            z_axis = rot_matrix[:, 2]  # 旋转后的Z轴
            # ========== 关键修改结束 ==========

            # 绘制坐标系（突出Z轴）
            for j, (axis, col) in enumerate(zip(rot_matrix.T, ['r', 'g', 'b'])):
                linewidth = 1 if j == 2 else 1.5  # 加粗Z轴
                # alpha = 0.8

                ax.quiver(
                    0, 0, z_pos, *axis,
                    length=axis_length,
                    color=col,
                    alpha=0.8,
                    normalize=False,
                    arrow_length_ratio=0.15,
                    linewidth=linewidth,
                    label=f'Yaw={yaw[i]:.1f}°' if (i % 10 == 0 and j == 2) else ""
                )

            # 标记关键点
            # if i % 10 == 0:
            #     ax.scatter(0, 0, z_pos, color=cmap(norm(i)), s=20)

        # 参考坐标系（灰色）
        ref_axes = np.eye(3)
        for axis, col in zip(ref_axes.T, ['0.7', '0.7', '0.7']):
            ax.quiver(0, 0, total_height, *axis, length=0.3, color=col, alpha=0.5)
        # set_3d_axes_equal(ax)

        # 图形设置
        ax.set_zlim([0, total_height])
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Steps ↓', rotation=90, labelpad=15)
        # ax.set_title(f'Position Adjustment Trajectory on ', fontsize=12)
        ax.view_init(elev=25, azim=-60)

        # 颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.1)
        cbar.set_label('Steps Progression', labelpad=15)

        plt.tight_layout()
        plt.show()

def set_3d_axes_equal(ax):
    """设置三维坐标轴完全等比例"""
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    center = np.mean(limits, axis=1)
    radius = 0.7 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim([center[0] - radius, center[0] + radius])
    ax.set_ylim([center[1] - radius, center[1] + radius])
    ax.set_zlim([center[2] - radius, center[2] + radius])

def plot_collect_data_forces_torques(force_torque, title):
    for idx, data in enumerate(force_torque):

        """绘制单个轨迹的力和力矩子图"""
        fig, (ax_force, ax_torque) = plt.subplots(2, 1, figsize=(10, 8))

        # 颜色和标签设置
        colors = ['r', 'g', 'b']
        force_labels = ['Fx', 'Fy', 'Fz']
        torque_labels = ['Mx', 'My', 'Mz']

        # ----------------------------
        # 子图1：力分量
        # ----------------------------
        force = data[:, 0:3]
        for i in range(3):
            ax_force.plot(
                force[:, i],
                color=colors[i],
                linewidth=1.5,
                label=force_labels[i]
            )
        ax_force.set_ylabel('Force (N)', fontsize=12)
        # ax_force.set_title(f'{title} - Force Components', fontsize=14)
        ax_force.grid(True, linestyle='--', alpha=0.5)
        ax_force.legend(loc='upper right')

        # ----------------------------
        # 子图2：力矩分量
        # ----------------------------
        torque = data[:, 3:7]
        for i in range(3):
            ax_torque.plot(
                torque[:, i],
                color=colors[i],
                linewidth=1.5,
                label=torque_labels[i]
            )
        ax_torque.set_xlabel('Steps', fontsize=12)
        ax_torque.set_ylabel('Torque (Nm)', fontsize=12)
        # ax_torque.set_title(f'{title} - Torque Components', fontsize=14)
        ax_torque.grid(True, linestyle='--', alpha=0.5)
        ax_torque.legend(loc='upper right')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # 加载所有轨迹文件
    file_names = ["pose.npy", "pose_.npy"]  # 或.csv
    trajectories = [np.load(f) for f in file_names]  # 如果是.csv，用 np.loadtxt(f, delimiter=',')

    # 为每组轨迹设置颜色和标签
    colors = ['y', 'm']  # 黄色、洋红色
    labels = ['Robot A', 'Robot B']


    # plot_collect_data_trajectory(trajectories, colors, labels)
    plot_phase_adjustment(trajectories)

    file_names = ["force.npy", "force_.npy"]  # 或.csv
    trajectories = [np.load(f) for f in file_names]  # 如果是.csv，用 np.loadtxt(f, delimiter=',')

    # 为每组轨迹设置颜色和标签
    labels = ['Robot A', 'Robot B']


    plot_collect_data_forces_torques(trajectories, labels)