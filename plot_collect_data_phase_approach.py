import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D

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
            data[:len(data)-5, 0], data[:len(data)-5, 1], data[:len(data)-5, 2], linestyle='-',
            color=colors[idx], label=labels[idx], linewidth=2
        )

        # 在每个点绘制局部坐标系（调整step控制密度）
        step = 1  # 每隔step个点画一个坐标系（1=全部点）
        axis_length = 0.007  # 缩短箭头长度避免重叠

        for i in range(0, len(data)-5, step):
            pos = data[i, :3]
            # end_orn_euler = R.from_quat(data[i, 3:7]).as_euler('xyz', degrees=True)
            # roll, pitch, yaw = end_orn_euler[0], end_orn_euler[1], end_orn_euler[2]
            # rot_matrix = R.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
            rot_matrix = R.from_quat(data[i, 3:7]).as_matrix()
            quat = data[i, 3:7]
            print(np.linalg.norm(quat))
            print(np.allclose(rot_matrix.T @ rot_matrix, np.eye(3), atol=1e-6))
            # 绘制X/Y/Z轴（RGB颜色）
            c = 1
            for axis, color in zip(rot_matrix.T, ['g', 'r', 'b']):
                ax.quiver(
                    pos[0], pos[1], pos[2],
                    axis[0], axis[1], axis[2],
                    color=color, length=axis_length,
                    arrow_length_ratio=0.1,
                    linewidth=0.5,
                    alpha=0.7,  # 半透明避免遮挡
                    label=f'{labels[c]}' if (idx == 0 and i == 0) else ""
                )
                c += 1

    # 设置图形属性
    # ax.set_xlim([-0.1, 0.5])
    # ax.set_ylim([-0.1, 0.5])
    # ax.set_zlim([0.429999, 0.4301])

    ax.set_xlabel('X (m)', labelpad=15, fontsize=14)
    ax.set_ylabel('Y (m)', labelpad=15, fontsize=14)
    ax.set_zlabel('Z (m)', labelpad=15, fontsize=14)
    ax.view_init(elev=30, azim=45)  # 固定视角
    ax.tick_params(axis='x', pad=10, labelsize=12)
    ax.tick_params(axis='y', pad=10, labelsize=12)
    ax.tick_params(axis='z', pad=10, labelsize=12)
    ax.xaxis.pane.set_facecolor('none')
    ax.yaxis.pane.set_facecolor('none')
    ax.zaxis.pane.set_facecolor('none')
    # ax.set_title('Robot 3D Trajectory with Euler Angle Frames')
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.show()

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
    file_names = ["action_phase_approach.npy"]  # 或.csv
    trajectories = [np.load(f) for f in file_names]  # 如果是.csv，用 np.loadtxt(f, delimiter=',')

    # 为每组轨迹设置颜色和标签
    colors = ['black', 'm']  # 黄色、洋红色More actions
    labels = ['trajectories', 'roll', 'pitch', 'yaw']


    plot_collect_data_trajectory(trajectories, colors, labels)

    # file_names = ["forces2.npy"]  # 或.csv
    # trajectories = [np.load(f) for f in file_names]  # 如果是.csv，用 np.loadtxt(f, delimiter=',')
    #
    # # 为每组轨迹设置颜色和标签
    # labels = ['Robot A', 'Robot B']


    plot_collect_data_forces_torques(trajectories, labels)