import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def validate_and_plot(trajectories, colors, labels):
    """带数据验证的增强版轨迹绘制函数"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # ================== 数据验证阶段 ==================
    print("\n=== 数据验证报告 ===")
    for i, traj in enumerate(trajectories):
        print(f"\n轨迹 {i + 1} ({labels[i]})")
        print(f"总点数: {len(traj)}")
        print("X范围: [{:.4f}, {:.4f}]".format(traj[:, 0].min(), traj[:, 0].max()))
        print("Y范围: [{:.4f}, {:.4f}]".format(traj[:, 1].min(), traj[:, 1].max()))
        print("Z范围: [{:.4f}, {:.4f}]".format(traj[:, 2].min(), traj[:, 2].max()))

        # 检查Z轴异常值
        z_diff = np.abs(traj[:, 2] - traj[0, 2])
        if np.any(z_diff > 10):  # 假设正常移动范围小于10m
            print("警告: 检测到Z轴可能异常！")
            abnormal_idx = np.where(z_diff > 10)[0]
            print(f"异常点索引: {abnormal_idx}")
            print("异常点数据:")
            print(traj[abnormal_idx])

    # ================== 可视化阶段 ==================
    print("\n=== 开始绘图 ===")
    for idx, (data, color, label) in enumerate(zip(trajectories, colors, labels)):
        # 绘制主轨迹
        line = ax.plot(data[:, 0]+idx*0.002, data[:, 1]+idx*0.002, data[:, 2]+idx*0.002,
                       color=color, label=label, linewidth=2.5)

        # 标记关键点
        ax.scatter(data[0, 0], data[0, 1], data[0, 2],
                   color=color, marker='o', s=120) # , label=f'{label} Start'
        ax.scatter(data[-1, 0], data[-1, 1], data[-1, 2],
                   color=color, marker='X', s=150) # , label=f'{label} End'

        # 添加轨迹序号标记
        mid_point = len(data) // 2
        ax.text(data[mid_point, 0], data[mid_point, 1], data[mid_point, 2],
                str(idx + 1), color='black', fontsize=12, ha='center')

    # ================== 坐标轴设置 ==================

    ax.set_xlabel('X (m)', fontsize=14, labelpad=15)
    ax.set_ylabel('Y (m)', fontsize=14, labelpad=15)
    ax.set_zlabel('Z (m)', fontsize=14, labelpad=15)

    # 智能坐标范围设置
    all_points = np.concatenate(trajectories)
    max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                          all_points[:, 1].max() - all_points[:, 1].min(),
                          all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0

    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_zlim([0.42, 0.48])

    # 视角和样式设置
    ax.view_init(elev=25, azim=45)
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.grid(True, linestyle=':', alpha=0.6)

    # 专业级图例
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))  # 去重
    ax.legend(unique_labels.values(), unique_labels.keys(),
              fontsize=12, loc='upper left',
              bbox_to_anchor=(0.05, 0.95))

    plt.tight_layout()
    plt.show()


def load_and_process(files_list):
    """带数据预处理的加载函数"""
    trajectories = []

    for files in files_list:
        shape_data = []
        for f in files:
            try:
                data = np.load(f)
                print(f"\n加载文件: {f}")
                print(f"原始数据形状: {data.shape}")

                # 数据预处理检查
                if data.ndim != 2 or data.shape[1] < 3:
                    print(f"警告: 文件 {f} 数据维度异常，跳过")
                    continue

                # 检查Z轴数据
                z_mean = np.mean(data[:, 2])
                if abs(z_mean) > 100:  # 假设合理Z值范围
                    print(f"警告: 文件 {f} 的Z值异常 (平均Z = {z_mean:.2f})")
                    print("尝试自动修正...")
                    data[:, 2] = data[:, 2] - z_mean  # 中心化处理

                shape_data.append(data)

            except Exception as e:
                print(f"加载 {f} 时出错: {str(e)}")
                continue

        if shape_data:
            # 按时间顺序拼接
            concatenated = np.concatenate(shape_data, axis=0)
            print(f"合并后形状: {concatenated.shape}")
            trajectories.append(concatenated)

    return trajectories


if __name__ == '__main__':
    # 文件配置
    file_groups = [
        ['action_HDQN_circle_approach.npy', 'action_HDQN_circle_align.npy', 'action_HDQN_circle_contact.npy'],
        ['action_HDQN_triangle_approach.npy', 'action_HDQN_triangle_align.npy', 'action_HDQN_triangle_contact.npy'],
        ['action_HDQN_square_approach.npy', 'action_HDQN_square_align.npy', 'action_HDQN_square_contact.npy']
    ]

    # 加载和处理数据
    trajectories = load_and_process(file_groups)

    # 可视化设置
    colors = ['r', 'g', 'b']  # 专业配色
    labels = [
        'Circular Peg Trajectory',
        'Triangular Peg Trajectory',
        'Square Peg Trajectory'
    ]

    if not trajectories:
        print("错误: 没有有效数据可绘制")
    else:
        validate_and_plot(trajectories, colors, labels)