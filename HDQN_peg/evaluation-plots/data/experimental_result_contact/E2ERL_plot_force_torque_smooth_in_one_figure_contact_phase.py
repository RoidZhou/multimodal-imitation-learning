import glob
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
            step = list(map(lambda x: int(x.step), event_list))  # Force step to integer
            r = {'metric': [tag] * len(step), 'value': values, 'step': step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r], ignore_index=True)
    except Exception as e:
        print(f'Event file may be corrupted: {path}')
        import traceback
        traceback.print_exc()
    return runlog_data

def smooth_data(data: pd.Series, window_size: int = 20) -> np.ndarray:
    """对数据应用移动平均平滑"""
    smoothed = data.rolling(window_size, min_periods=1, center=True).mean().to_numpy()
    return smoothed

def sort_log_paths(paths, prefix='force'):
    """按 fx, fy, fz 或 tx, ty, tz 排序日志文件"""
    def get_priority(path):
        basename = os.path.basename(path).lower()
        if f'{prefix}_x' in basename or f'{prefix}x' in basename:
            return 0  # fx 或 tx 优先
        elif f'{prefix}_y' in basename or f'{prefix}y' in basename:
            return 1  # fy 或 ty 次之
        elif f'{prefix}_z' in basename or f'{prefix}z' in basename:
            return 2  # fz 或 tz 最后
        return 3  # 其他文件
    return sorted(paths, key=get_priority)

# Find TensorBoard log files
path = '/home/zhou/autolab/imitation_learning_idp3/HDQN_peg/evaluation-plots/data/experimental_result_contact/E2ERL/triangle'
force_log_paths = sort_log_paths(glob.glob(os.path.join(path, 'force/*')), prefix='f')
torque_log_paths = sort_log_paths(glob.glob(os.path.join(path, 'torque/*')), prefix='t')

# Hardcode custom legend labels
force_custom_labels = ['Fx', 'Fy', 'Fz']
torque_custom_labels = ['Tx', 'Ty', 'Tz']

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
        log['label'] = force_custom_labels[idx] if idx < len(force_custom_labels) else f'Force_{idx}'
        log['step'] = log['step'].astype(np.int64)  # Ensure step is integer
        force_logs = pd.concat([force_logs, log], ignore_index=True)
    else:
        print(f"Warning: Empty log file {log_path}")

# Process torque logs
torque_logs = pd.DataFrame()
for idx, log_path in enumerate(torque_log_paths):
    log = tflog2pandas(log_path)
    if not log.empty:
        log['label'] = torque_custom_labels[idx] if idx < len(torque_custom_labels) else f'Torque_{idx}'
        log['step'] = log['step'].astype(np.int64)  # Ensure step is integer
        torque_logs = pd.concat([torque_logs, log], ignore_index=True)
    else:
        print(f"Warning: Empty log file {log_path}")

# Debug: Check data before filtering
print(f"force_logs shape: {force_logs.shape}")
print(f"torque_logs shape: {torque_logs.shape}")
print(f"force_logs labels: {force_logs['label'].unique()}")
print(f"torque_logs labels: {torque_logs['label'].unique()}")
print(f"force_logs step dtype: {force_logs['step'].dtype}")
print(f"torque_logs step dtype: {torque_logs['step'].dtype}")

# Filter specific metrics and steps (10 to 200)
force_metrics = ['force_x', 'force_y', 'force_z']
torque_metrics = ['Torque_x', 'Torque_y', 'Torque_z']
agg_force_logs = force_logs[force_logs['metric'].isin(force_metrics) & force_logs['step'].between(10, 270)].copy()
agg_force_logs['display_step'] = (agg_force_logs['step'] - 10).astype(np.int64)  # Map step 10-200 to 0-190
agg_force_logs = agg_force_logs.reset_index(drop=True)
agg_torque_logs = torque_logs[torque_logs['metric'].isin(torque_metrics) & torque_logs['step'].between(10, 270)].copy()
agg_torque_logs['display_step'] = (agg_torque_logs['step'] - 10).astype(np.int64)  # Map step 10-200 to 0-190
agg_torque_logs = agg_torque_logs.reset_index(drop=True)

# Debug: Check filtered data
print(f"agg_force_logs shape: {agg_force_logs.shape}")
print(f"agg_torque_logs shape: {agg_torque_logs.shape}")
print(f"agg_force_logs labels: {agg_force_logs['label'].unique()}")
print(f"agg_torque_logs labels: {agg_torque_logs['label'].unique()}")
print(f"agg_force_logs step range: {agg_force_logs['step'].min()} to {agg_force_logs['step'].max()}")
print(f"agg_torque_logs step range: {agg_torque_logs['step'].min()} to {agg_torque_logs['step'].max()}")
print(f"agg_force_logs display_step dtype: {agg_force_logs['display_step'].dtype}")
print(f"agg_torque_logs display_step dtype: {agg_torque_logs['display_step'].dtype}")

# Apply smoothing
window_size = 1
agg_force_logs['smoothed_value'] = agg_force_logs.groupby(['label'])['value'].transform(lambda x: smooth_data(x, window_size))
agg_torque_logs['smoothed_value'] = agg_torque_logs.groupby(['label'])['value'].transform(lambda x: smooth_data(x, window_size))

# Define color palette based on labels
force_palette = {
    'Fx': 'red',
    'Fy': 'green',
    'Fz': 'blue',
    **{f'Force_{i}': 'gray' for i in range(len(force_log_paths))}  # Fallback for extra labels
}
torque_palette = {
    'Tx': 'red',
    'Ty': 'green',
    'Tz': 'blue',
    **{f'Torque_{i}': 'gray' for i in range(len(torque_log_paths))}  # Fallback for extra labels
}

# Create figure with two subplots
sns.set(style='whitegrid')
sns.set_context("notebook", font_scale=1.4)  # Global font scale increased to 1.4

fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)  # 2行1列，共享 x 轴

# Subplot 1: Force (Fx, Fy, Fz)
sns.lineplot(data=agg_force_logs,
             x='display_step',
             y='smoothed_value',
             hue='label',
             palette=force_palette,
             linestyle='-',
             errorbar=None,
             alpha=0.7,
             ax=axes[0])
axes[0].set_ylabel('Force (N)', fontsize=16)
axes[0].set_xlim(0, 260)
axes[0].set_ylim(-25, 10)
axes[0].xaxis.set_major_locator(ticker.MultipleLocator(40))
axes[0].tick_params(axis='both', labelsize=14)  # Increase tick label size to 14
axes[0].legend(loc='upper right', fancybox=True, bbox_to_anchor=(1, 1), shadow=False, framealpha=0.6,  ncol=1, prop={'size': 14})
axes[0].set_xlabel('')

# Subplot 2: Torque (Tx, Ty, Tz)
sns.lineplot(data=agg_torque_logs,
             x='display_step',
             y='smoothed_value',
             hue='label',
             palette=torque_palette,
             linestyle='-',
             errorbar=None,
             alpha=0.7,
             ax=axes[1])
axes[1].set_xlabel('Steps', fontsize=16)
axes[1].set_ylabel('Torque (N·m)', fontsize=16)
axes[1].set_xlim(0, 260)
axes[1].set_ylim(-0.2, 0.3)
axes[1].xaxis.set_major_locator(ticker.MultipleLocator(40))
axes[1].tick_params(axis='both', labelsize=14)  # Increase tick label size to 14
axes[1].legend(loc='upper right', fancybox=True, bbox_to_anchor=(1, 1), shadow=False, framealpha=0.6,  ncol=1, prop={'size': 14})

# Adjust layout to prevent overlap
plt.tight_layout(pad=3.0)
plt.savefig('force_torque_combined.png', dpi=300)
plt.show()

# Debug: Confirm font sizes and legend position
print("Font sizes set:")
print(f"Axis labels: 16 pt")
print(f"Tick labels: 14 pt")
print(f"Legend: 14 pt")
print(f"Legend position: upper left, bbox_to_anchor=(0.02, 0.98)")
print("end")