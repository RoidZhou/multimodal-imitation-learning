import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d
sns.set()

def smooth(data, wd=2, method='gaussian', sigma=5):
    if not (isinstance(wd, int) and wd > 0):
        raise ValueError('wd must be a positive integer')

    if method == 'moving_average':
        if wd == 1:
            return data
        else:
            weight = np.ones(wd) / wd
            if data.ndim == 1:
                return np.convolve(weight, data, "same")
            elif data.ndim == 2:
                smooth_data = [np.convolve(weight, d, "same") for d in data]
                return np.array(smooth_data)
            else:
                raise ValueError('data must be a one-dimensional or two-dimensional ndarray')

    elif method == 'gaussian':
        if data.ndim == 1:
            return gaussian_filter1d(data, sigma=sigma)
        elif data.ndim == 2:
            smooth_data = [gaussian_filter1d(d, sigma=sigma) for d in data]
            return np.array(smooth_data)
        else:
            raise ValueError('data must be a one-dimensional or two-dimensional ndarray')

    else:
        raise ValueError('Invalid method. Choose "moving_average" or "gaussian".')

def get_data():
    GS1 = pd.read_csv("/HDQN_peg/csv/force_error_z_force_error_z.csv")
    GS2 = pd.read_csv("/HDQN_peg/csv/force_error_z_force_error_z.csv")
    G1 = pd.read_csv("/HDQN_peg/csv/rz_rz.csv")
    G2 = pd.read_csv("/HDQN_peg/csv/rz_rz.csv")


    GS1_array = GS1['Value']
    GS2_array = GS2['Value']
    G1_array = G1['Value']
    G2_array = G2['Value']


    returns1 = np.vstack((GS1_array, GS2_array))
    returns1 = smooth(returns1, 1, sigma=1)
    returns2 = np.vstack((G1_array, G2_array))
    returns2 = smooth(returns2, 1,sigma=1)

    return returns1, returns2

# np.random.seed(11)
# data = get_data()
# label = ['Grid Sensor + Vector Sensor', 'Grid Sensor']
#
# df=[]
# vT1 = pd.read_csv("/home/zhou/autolab/imitation_learning_idp3/HDQN_peg/csv/force_error_z_force_error_z.csv")
# ax = vT1['Step']
#
# for i in range(len(data)):
#     df.append(pd.DataFrame(data[i], columns=ax).melt(var_name='Episode', value_name='Rewards'))
#     df[i]['algo'] = label[i]
# df = pd.concat(df, ignore_index=True)
#
# # 设置全局字体为 Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
#
# # 创建线图，并使用不同的线型
# sns.lineplot(x="Episode", y="Rewards", hue="algo", style="algo", data=df, markers=False)
#
# # 设置图例字体大小为 16
# plt.legend(loc='center right', fontsize=16)
#
# # 设置标题、标签字体为 Times New Roman
# plt.xlabel("Episode", fontsize=16, fontname='Times New Roman')
# plt.ylabel("Rewards", fontsize=16, fontname='Times New Roman')
#
# # 设置坐标刻度字体大小为 16
# plt.tick_params(axis='both', which='major', labelsize=16)
# ax = plt.gca()
# # 设置坐标轴刻度的科学计数法格式
# ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
# ax.ticklabel_format(style='sci', scilimits=(-1, 1000), axis='x')
#
# # 调整科学计数法指数字体大小
# ax.xaxis.get_offset_text().set_fontsize(16)
# ax.yaxis.get_offset_text().set_fontsize(16)
# # 保存为 SVG 格式
# plt.savefig('plot.svg', format='svg')
#
# plt.tight_layout()  # 自动调整布局避免重叠
# # 展示图形
# plt.show()

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv

'''读取csv文件'''


def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append((row[2]))
        x.append((row[1]))
    return x, y


mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

plt.figure()
x2, y2 = readcsv("/home/zhou/autolab/imitation_learning_idp3/HDQN_peg/csv/force_error_z_force_error_z.csv")
plt.plot(x2, y2, color='red', label='Default')
# plt.plot(x2, y2, '.', color='red')

# x, y = readcsv("/home/zhou/autolab/imitation_learning_idp3/HDQN_peg/csv/rz_rz.csv")
# plt.plot(x, y, 'g', label='Without BN')

# x1, y1 = readcsv("scalars2.csv")
# plt.plot(x1, y1, color='black', label='Without DW and PW')
#
# x4, y4 = readcsv("scalars4.csv")
# plt.plot(x4, y4, color='blue', label='Without Residual learning')

plt.xticks(fontsize=4)
plt.yticks(fontsize=4)

plt.ylim(-0.10000014, -0.1)
plt.xlim(0, 100)
plt.xlabel('Steps', fontsize=20)
plt.ylabel('Score', fontsize=20)
plt.legend(fontsize=16)
plt.show()