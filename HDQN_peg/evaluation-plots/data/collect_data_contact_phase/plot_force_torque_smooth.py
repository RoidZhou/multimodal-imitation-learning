import glob
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def tflog2pandas(path: str) -> pd.DataFrame:
    """Convert a single TensorFlow log file to pandas DataFrame"""
    DEFAULT_SIZE_GUIDANCE = {
        'compressedHistograms': 1,
        'images': 1,
        'scalars': 0,
        'histograms': 1,
    }
    runlog_data = pd.DataFrame({'metric': [], 'value': [], 'step': []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()['scalars']
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {'metric': [tag] * len(step), 'value': values, 'step': step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r], ignore_index=True)
    except Exception as e:
        print(f'Event file may be corrupted: {path}')
        import traceback
        traceback.print_exc()
    return runlog_data

def smooth_data(data: pd.Series, window_size: int = 20) -> np.ndarray:
    """对数据应用移动平均平滑，并将第20个点设为-1"""
    smoothed = data.rolling(window_size, min_periods=1, center=True).mean().to_numpy()
    # if len(smoothed) >= 20:
    #     smoothed[19] = -0.6  # 第20个点（索引19）设为-1
    #     smoothed[20] = -0.6  # 第20个点（索引19）设为-1
    #     smoothed[21] = -0.6  # 第20个点（索引19）设为-1
    return smoothed

# 自定义排序函数：按 fx, fy, fz 顺序
def sort_log_paths(paths, prefix='force'):
    """按 fx, fy, fz 或 tx, ty, tz 排序日志文件"""
    def get_priority(path):
        basename = os.path.basename(path).lower()
        if f'{prefix}_x' in basename:
            return 0  # fx 或 tx 优先
        elif f'{prefix}_y' in basename:
            return 1  # fy 或 ty 次之
        elif f'{prefix}_z' in basename:
            return 2  # fz 或 tz 最后
        return 3  # 其他文件
    return sorted(paths, key=get_priority)

# Find TensorBoard log files
path = '/home/zhou/autolab/imitation_learning_idp3/HDQN_peg/evaluation-plots/data/collect_data_contact_phase'
force_log_paths = sort_log_paths(glob.glob(os.path.join(path, 'force/*')), prefix='f')
torque_log_paths = sort_log_paths(glob.glob(os.path.join(path, 'torque/*')), prefix='t')

# Hardcode custom legend labels (replace with your desired labels)
force_custom_labels = [
    'Fx', 'Fy', 'Fz', 'Force_Y_Trial2',
    'Force_Z_Trial1', 'Force_Z_Trial2'
]  # Adjust number to match your force log files
torque_custom_labels = [
    'Tx', 'Ty', 'Tz', 'Torque_Y_Trial2',
    'Torque_Z_Trial1', 'Torque_Z_Trial2'
]  # Adjust number to match your torque log files

# Debug: Print log file paths and assigned labels
print("Force log paths and labels:")
for i, p in enumerate(force_log_paths):
    print(f"  {os.path.basename(p)} -> {force_custom_labels[i] if i < len(force_custom_labels) else 'Unlabeled'}")
print("Torque log paths and labels:")
for i, p in enumerate(torque_log_paths):
    print(f"  {os.path.basename(p)} -> {torque_custom_labels[i] if i < len(torque_custom_labels) else 'Unlabeled'}")

# Process force logs
force_logs = pd.DataFrame()
for idx, log_path in enumerate(force_log_paths):
    log = tflog2pandas(log_path)
    if not log.empty:
        # log = log.head(150)  # Limit rows
        log['label'] = force_custom_labels[idx] if idx < len(force_custom_labels) else f'Force_{idx}'
        force_logs = pd.concat([force_logs, log], ignore_index=True)

# Process torque logs
torque_logs = pd.DataFrame()
for idx, log_path in enumerate(torque_log_paths):
    log = tflog2pandas(log_path)
    if not log.empty:
        # log = log.head(150)  # Limit rows
        log['label'] = torque_custom_labels[idx] if idx < len(torque_custom_labels) else f'Torque_{idx}'
        torque_logs = pd.concat([torque_logs, log], ignore_index=True)

# Filter specific metrics and steps
force_metrics = ['force_x', 'force_y', 'force_z']
torque_metrics = ['Torque_x', 'Torque_y', 'Torque_z']

agg_force_logs = force_logs[force_logs['metric'].isin(force_metrics) & (force_logs['step'].between(4,69))].copy()
agg_force_logs = agg_force_logs.reset_index()
agg_torque_logs = torque_logs[torque_logs['metric'].isin(torque_metrics) & (torque_logs['step'].between(4,69))].copy()
agg_torque_logs = agg_torque_logs.reset_index()

# Apply smoothing
window_size = 1
agg_force_logs['smoothed_value'] = agg_force_logs.groupby(['label'])['value'].transform(lambda x: smooth_data(x, window_size))
agg_torque_logs['smoothed_value'] = agg_torque_logs.groupby(['label'])['value'].transform(lambda x: smooth_data(x, window_size))

# 定义颜色映射：按 metric 分配红、绿、蓝
force_palette = {
    'force_x': 'red',
    'force_y': 'green',
    'force_z': 'blue'
}
torque_palette = {
    'Torque_x': 'red',
    'Torque_y': 'green',
    'Torque_z': 'blue'
}

# Create two plots
sns.set(style='whitegrid')
sns.set_context("notebook")

# Plot 1: Force (force_x, force_y, force_z)
fig1 = plt.figure(figsize=(8, 5))
graph1 = sns.lineplot(data=agg_force_logs,
                      x='step',
                      y='smoothed_value',
                      hue='label',
                      style='label',
                      palette={label: force_palette[metric] for label, metric in
                               zip(agg_force_logs['label'], agg_force_logs['metric'])},
                      errorbar=None,
                      alpha=0.7)
graph1.set(xlabel='Step', ylabel='Reward', title='Force Directions')
graph1.set_ylim(-0.2, 0.05)
plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1, 1), shadow=True, ncol=1, prop={'size': 10})
plt.tight_layout(pad=1.0)
plt.savefig('force_reward_smooth.png', dpi=300)
plt.show()

# Plot 2: Torque (torque_x, torque_y, torque_z)
fig2 = plt.figure(figsize=(8, 5))
graph2 = sns.lineplot(data=agg_torque_logs,
                      x='step',
                      y='smoothed_value',
                      hue='label',
                      style='label',
                      palette={label: torque_palette[metric] for label, metric in
                               zip(agg_torque_logs['label'], agg_torque_logs['metric'])},
                      errorbar=None,
                      alpha=0.7)
graph2.set(xlabel='Step', ylabel='Reward', title='Torque Directions')
graph2.set_ylim(-0.002, 0.002)
plt.legend(loc='upper right', fancybox=True, shadow=True, ncol=1, prop={'size': 10})
plt.tight_layout(pad=1.0)
plt.savefig('torque_reward_smooth.png', dpi=300)
plt.show()
print("end")