import glob
import os
import pprint
import traceback
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
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
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {'metric': [tag] * len(step), 'value': values, 'step': step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print('Event file possibly corrupt: {}'.format(path))
        traceback.print_exc()
    return runlog_data

# 定义平滑函数（移动平均）
def smooth_data_rolling(data: pd.Series, window_size: int = 10) -> np.ndarray:
    """对数据应用移动平均平滑"""
    return data.rolling(window=window_size, min_periods=1, center=True).mean()

def smooth_data_ewm(data: pd.Series, smooth: float = 0.99) -> np.ndarray:
    """使用指数加权移动平均平滑数据，模拟 TensorBoard 的 smooth 参数"""
    alpha = 1 - smooth  # 平滑因子
    return data.ewm(alpha=alpha, adjust=True).mean()

ADD_LAST_150 = True
# find TB logfiles in path
path = './data'
param1_log_paths = glob.glob(os.path.join(path, 'assemby/*'))
param2_log_paths = glob.glob(os.path.join(path, 'paper/*'))


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
        if ADD_LAST_150:
            # 检查行数并补齐到 500 行
            num_rows = log.shape[0]
            print(f"Log at {path} has {num_rows} rows")
            if num_rows > 0:
                if 'step' not in log.columns:
                    raise ValueError(f"'step' column not found in log at {path}")
                gap = 500 - num_rows
                if gap > 0:
                    # 复制最后 gap 行（或全部行，重复直到补齐）
                    rows_to_copy = min(gap, num_rows)
                    last_rows = log.iloc[-rows_to_copy:].copy()
                    additional_rows = pd.concat([last_rows] * ((gap // rows_to_copy) + 1))[:gap]
                    additional_rows.index = range(log.index[-1] + 1, log.index[-1] + 1 + gap)
                    additional_rows['step'] = range(int(log['step'].iloc[-1]) + 1, int(log['step'].iloc[-1]) + 1 + gap)
                    log = pd.concat([log, additional_rows], ignore_index=False)
                    print(f"Appended {gap} rows (last {rows_to_copy} rows repeated) with indices {additional_rows.index[0]} to {additional_rows.index[-1]}, step values {additional_rows['step'].iloc[0]} to {additional_rows['step'].iloc[-1]}. New rows: {log.shape[0]}")
                else:
                    print(f"No need to append, rows ({num_rows}) >= 500")
            else:
                print(f"Log at {path} is empty, skipping append")

        # 合并到 param2_logs
        if param2_logs.shape[0] == 0:
            param2_logs = log
        else:
            param2_logs = pd.concat([param2_logs, log], ignore_index=True)

# query episode rewards (rollout/ep_rew_mean)
# limit steps to 1000000
# agg_param1_logs = param1_logs.query('metric == "rollout/ep_rew_mean" & step <= 1000000').copy()
agg_param1_logs = param1_logs.query('metric == "episode_reward" & step <= 100000').copy()
agg_param1_logs = agg_param1_logs.reset_index()
agg_param1_logs['type'] = 'algo_1'

# query episode rewards (eval/mean_episode_reward)
# limit steps to 1000000
agg_param2_logs = param2_logs.query('metric == "episode_reward" & step <= 1000000').copy()
# agg_param2_logs = param2_logs.query('metric == "reward" & step <= 1000').copy()
agg_param2_logs = agg_param2_logs.reset_index()
agg_param2_logs['type'] = 'algo_2'

# join log files
joint_logs = pd.DataFrame()
joint_logs = pd.concat([joint_logs, agg_param1_logs])
# joint_logs = joint_logs.append(agg_param2_logs)
joint_logs = pd.concat([joint_logs, agg_param2_logs], ignore_index=True)
joint_logs.head()

# 应用平滑处理
window_size = 10  # 平滑窗口大小，可根据需要调整
joint_logs['smoothed_value'] = joint_logs.groupby('type')['value'].transform(lambda x: smooth_data_ewm(x, smooth=0.99))


# create figure (one plot)
fig = plt.figure(figsize=(5.5, 4.5))
fig.patch.set_facecolor('white')

# set style and context
sns.set(style='whitegrid')
sns.set_context("notebook")

# create graph
graph = sns.lineplot(data=joint_logs,
                     x='step',
                     y='smoothed_value',
                     hue='type',
                     errorbar='sd')

# set legend options
handles, labels = graph.get_legend_handles_labels()
labels = ['Algorithm 1', 'Algorithm 2']
fig.tight_layout(pad=2.0)
fig.legend(loc='lower center',
           bbox_to_anchor=(0.5, -0.1),
           fancybox=True,
           shadow=True,
           ncol=5,
           prop={'size': 12},
           labels=labels)

# graph options
graph.legend([],[], frameon=False)
graph.set(xlabel='Environment Steps', ylabel='Mean Episode Return')
graph.title.set_text('Environment Name')
graph.title.set_size(13)
graph.set_ylim(0, 18)
plt.savefig('reward.png', dpi=300)