import glob
import os
import traceback
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame"""
    DEFAULT_SIZE_GUIDANCE = {
        'compressedHistograms': 1,
        'images': 1,
        'scalars': 0,  # 0 means load all
        'histograms': 1,
    }
    runlog_data = pd.DataFrame({'metric': [], 'value': [], 'step': []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()['scalars']
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = [x.value for x in event_list]
            step = [x.step for x in event_list]
            r = pd.DataFrame({'metric': [tag] * len(step), 'value': values, 'step': step})
            runlog_data = pd.concat([runlog_data, r])
    except Exception:
        print(f'Event file possibly corrupt: {path}')
        traceback.print_exc()
    return runlog_data

def smooth_data_ewm(data: pd.Series, smooth: float = 0.99) -> np.ndarray:
    """使用指数加权移动平均平滑数据"""
    alpha = 1 - smooth
    return data.ewm(alpha=alpha, adjust=True).mean()

PLOT_TRIANGLE = False
PLOT_CIRCLE = False
PLOT_SQUARE = True
path = './data'
legend_labels = ['HRL', 'E2ERL', 'HDPRL']

if PLOT_TRIANGLE:
    param1_log_paths = glob.glob(os.path.join(path, 'triangle/HDPRL/*'))
    param2_log_paths = glob.glob(os.path.join(path, 'triangle/E2ERL/*'))
    param3_log_paths = glob.glob(os.path.join(path, 'triangle/HRL/*'))
if PLOT_CIRCLE:
    param1_log_paths = glob.glob(os.path.join(path, 'circle/E2ERL/*'))
    param2_log_paths = glob.glob(os.path.join(path, 'circle/HRL/*'))
    param3_log_paths = glob.glob(os.path.join(path, 'circle/HDPRL/*'))
if PLOT_SQUARE:
    param1_log_paths = glob.glob(os.path.join(path, 'square/E2ERL/*'))
    param2_log_paths = glob.glob(os.path.join(path, 'square/HRL/*'))
    param3_log_paths = glob.glob(os.path.join(path, 'square/HDPRL/*'))
# create dataframes from TB logs
param1_logs = pd.DataFrame()
for path in param1_log_paths:
    log = tflog2pandas(path)
    if log is not None:
        if param1_logs.shape[0] == 0:
            param1_logs = log
        else:
            param1_logs = pd.concat([param1_logs, log], ignore_index=True)

param2_logs = pd.DataFrame()
for path in param2_log_paths:
    log = tflog2pandas(path)
    if log is not None:
        if param2_logs.shape[0] == 0:
            param2_logs = log
        else:
            param2_logs = pd.concat([param2_logs, log], ignore_index=True)

param3_logs = pd.DataFrame()
for path in param3_log_paths:
    log = tflog2pandas(path)
    if log is not None:
        if param3_logs.shape[0] == 0:
            param3_logs = log
        else:
            param3_logs = pd.concat([param3_logs, log], ignore_index=True)

# query episode rewards
agg_param1_logs = param1_logs.query('metric == "episode_reward" & step <= 100000').copy()
agg_param1_logs = agg_param1_logs.reset_index()
agg_param1_logs['type'] = 'algo_1'

agg_param2_logs = param2_logs.query('metric == "episode_reward" & step <= 1000000').copy()
agg_param2_logs = agg_param2_logs.reset_index()
agg_param2_logs['type'] = 'algo_2'

agg_param3_logs = param3_logs.query('metric == "episode_reward" & step <= 1000000').copy()
agg_param3_logs = agg_param3_logs.reset_index()
agg_param3_logs['type'] = 'algo_3'

# join log files
joint_logs = pd.concat([agg_param1_logs, agg_param2_logs], ignore_index=True)
joint_logs = pd.concat([joint_logs, agg_param3_logs], ignore_index=True)

# 应用平滑处理
joint_logs['smoothed_value'] = joint_logs.groupby('type')['value'].transform(lambda x: smooth_data_ewm(x, smooth=0.99))

# 创建绘图用数据，包含原始和平滑值
plot_data = pd.concat([
    joint_logs[['step', 'value', 'type']].assign(data_type='Raw'),
    joint_logs[['step', 'smoothed_value', 'type']].rename(columns={'smoothed_value': 'value'}).assign(data_type='Smoothed')
], ignore_index=True)

# create figure
fig = plt.figure(figsize=(8, 6))
fig.patch.set_facecolor('white')

# set style and context
sns.set(style='whitegrid')
sns.set_context("notebook")

# 定义颜色
palette = {'algo_1': 'blue', 'algo_2': 'red', 'algo_3': 'green'}

# 分离 Raw 和 Smoothed 数据
raw_data = plot_data[plot_data['data_type'] == 'Raw']
smoothed_data = plot_data[plot_data['data_type'] == 'Smoothed']

# 绘制 Raw 曲线（细线）
graph = sns.lineplot(data=raw_data,
                     x='step',
                     y='value',
                     hue='type',
                     palette=palette,  # 指定颜色
                     linewidth=0.5,
                     errorbar='sd',
                     alpha=0.5,
                     legend=False)

# 绘制 Smoothed 曲线（粗线）
sns.lineplot(data=smoothed_data,
             x='step',
             y='value',
             hue='type',
             palette=palette,  # 指定颜色
             linewidth=2.0,
             errorbar='sd',
             alpha=0.9,
             legend='brief')

# set legend options
handles, labels = graph.get_legend_handles_labels()
fig.tight_layout(pad=2.0)
fig.legend(handles[:4], legend_labels,
           loc='upper right',
           bbox_to_anchor=(0.85, 0.95),
           fancybox=True,
           shadow=True,
           ncol=1,
           prop={'size': 10})

# graph options
graph.legend([], [], frameon=False)
graph.set(xlabel='Timesteps', ylabel='Reward')
graph.set_title('', size=13)
graph.set_ylim(0, 20)
plt.subplots_adjust(right=0.85)
plt.show()
plt.savefig('reward_smooth.png', dpi=300, bbox_inches='tight')
plt.close()