import os.path
import sys
import numpy as np
import subprocess as sp
from numpy import genfromtxt
from IPython.display import Image
from IPython.display import display
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interpolate
import pandas as pd
import pprint

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams.update({'font.size': 12})
mpl.rcParams.update({'legend.fontsize': 8})
mpl.rcParams.update({'legend.handlelength': 2})

color = []
#color.append('dimgray')
color.append([0.3, 0.3, 0.3])
color.append('dodgerblue')
color.append('royalblue')
color.append('green')
color.append('coral')
color.append('gold')
color.append('lightpink')
color.append('brown')
color.append('red')

def csv_read(file_name):
    try:
        return genfromtxt(file_name, delimiter='\t', skip_header=1)
    except ValueError as e:
        print(file_name)
        raise e

def load_csv_data(file_path, num_elems, seeds=None, index = 2):
    stat_x = np.arange(1, num_elems+1)
    stat_y = []
    for seed in seeds:
        log_path = file_path + f'{int(seed)}/stats.txt'
        log = csv_read(log_path)
        f = open(log_path, 'r')
        y = np.zeros(num_elems)
        for i in range(num_elems):
            y[i] = log[i][index]
        stat_y.append(y)
    return stat_x, np.stack(stat_y, axis=0)

num_runs = 10
home = os.path.expanduser("~")
env='hex'
# file_path = home + '/Github/macarico/tests/VDR_rl/' + env + '/boltzmann/'
# x, y = load_csv_data(file_path, num_elems=10000, seeds = np.arange(1,11))
file_path = home + '/Github/macarico/tests/MC_rl/hex2/boltzmann/'
x, y = load_csv_data(file_path, num_elems=9999, seeds = np.arange(1,11), index=2)
# y = 200*y
# plt.xscale('log')
y_mean = np.mean(y, axis=0)
y_err = 2 * np.std(y, axis=0) / np.sqrt(10)
plt.plot(x, y_mean, alpha=1.0, linewidth=2, color='blue', label='MC')
# plt.plot(x, y_mean, alpha=1.0, linewidth=2, color='blue', label='')
plt.fill_between(x, y_mean - y_err, y_mean + y_err, facecolor='blue', alpha=0.3)

file_path = home + '/Github/macarico/tests/VDR_rl/hex_huber2/boltzmann/'
x, y = load_csv_data(file_path, num_elems=9999, seeds = np.arange(1,11), index=2)
y_mean = np.mean(y, axis=0)
y_err = 2 * np.std(y, axis=0) / np.sqrt(10)
# plt.plot(x, y_mean, alpha=1.0, linewidth=2, color='red', label='Monte-Carlo')
plt.plot(x, y_mean, alpha=1.0, linewidth=2, color='red', label='PREP_huber')
plt.fill_between(x, y_mean - y_err, y_mean + y_err, facecolor='red', alpha=0.3)

# file_path = home + '/Github/macarico/tests/MC_rl/' + env + '/boltzmann/'
# x, y = load_csv_data(file_path, num_elems=10000, seeds = np.arange(1,11))
plt.xscale('log')
file_path = home + '/Github/macarico/tests/MCC_rl/hex/boltzmann/'
x, y = load_csv_data(file_path, num_elems=9999, seeds = np.arange(1,11), index=2)
y_mean = np.mean(y, axis=0)
y_err = 2 * np.std(y, axis=0) / np.sqrt(10)
# plt.plot(x, y_mean, alpha=1.0, linewidth=2, color='red', label='Monte-Carlo')
plt.plot(x, y_mean, alpha=1.0, linewidth=2, color='green', label='MC_Critic')
plt.fill_between(x, y_mean - y_err, y_mean + y_err, facecolor='green', alpha=0.3)

file_path = home + '/Github/macarico/tests/MCC_rl_adv/hex/boltzmann/'
x, y = load_csv_data(file_path, num_elems=9999, seeds = np.arange(1,11), index=2)
y_mean = np.mean(y, axis=0)
y_err = 2 * np.std(y, axis=0) / np.sqrt(10)
# plt.plot(x, y_mean, alpha=1.0, linewidth=2, color='red', label='Monte-Carlo')
plt.plot(x, y_mean, alpha=1.0, linewidth=2, color='cyan', label='MC_critic_adv')
plt.fill_between(x, y_mean - y_err, y_mean + y_err, facecolor='cyan', alpha=0.3)
# plt.ylim([0, 1])
plt.xlabel('Number of episodes')
# plt.xticks(np.arange(1,5), [])
plt.grid(True,which="both",ls="-", lw=0.5)
plt.title('Hex')
plt.legend(fontsize='medium', fancybox=True)
plt.ylabel('Average episode loss (100 ep.)')
plt.show()

