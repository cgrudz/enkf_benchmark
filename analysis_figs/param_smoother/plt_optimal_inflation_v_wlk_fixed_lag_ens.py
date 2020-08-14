import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import Normalize
from matplotlib import rc
from matplotlib.colors import LogNorm
import glob
import ipdb
import math

tanl = 0.05
obs_un = 1.0
ens = 25
lag = 6
wlks = [0.0000, 0.0001, 0.0010, 0.0100]
stat = 'smooth'
methods = ['enks','etks']
versions = ['classic']
markerlist = ['o', 'v', '>', 'X', 'd']

fig = plt.figure()
ax1 = fig.add_axes([.530, .10, .40, .71])
ax0 = fig.add_axes([.050, .10, .40, .71])

line_list = []
name_list = []

for j in range(len(versions)):
    for i in range(len(methods)):
        rmse_list = []
        spread_list = []
        for k in range(len(wlks)):
            version = versions[j]
            method = methods[i]
            wlk = wlks[k]
            
            f = open('processed_' + version + '_smoother_param_rmse_spread_nanl_40000_tanl_0.05_burn_5000_wlk_' + str(wlk).ljust(6,'0') + '.txt', 'rb')
            data = pickle.load(f)
            f.close()
            
            rmse_list.append(data[method + '_' + stat + '_rmse'][lag, ens - 14])
            spread_list.append(data[method + '_' + stat + '_spread'][lag, ens - 14])
            
        l, = ax0.plot(range(4), rmse_list, marker=markerlist[i+j*2], linewidth=2, markersize=10)
        ax1.plot(range(4), spread_list, marker=markerlist[i+j*2], linewidth=2, markersize=10)
        line_list.append(l)
        name_list.append(method + ' ' + version)

ax1.tick_params(
        labelsize=20,
        labelleft=False,
        left=False,
        labelright=True,
        right=True
        )

ax0.tick_params(
        labelsize=20)

if stat == 'param':
    ax1.set_ylim([10e-20,1.0])
    ax0.set_ylim([10e-6,1.0])
    ax0.set_yscale('log')
    ax1.set_yscale('log')

else:
    ax1.set_ylim([0.01,0.25])
    ax0.set_ylim([0.01,0.25])

ax0.tick_params(
        labelsize=22)

ax1.tick_params(
        labelsize=22)

#ax1.set_xlim([-0.1, 4.1])
#ax0.set_xlim([-0.1, 4.1])
ax0.set_xticks(range(4))
ax1.set_xticks(range(4))
ax0.set_xticklabels([str(0), str(0.0001), str(0.001), str(0.01)])
ax1.set_xticklabels([str(0), str(0.0001), str(0.001), str(0.01)])


if stat == 'smooth':
    stat = 'Smoother'

elif stat == 'fore':
    stat = 'Forecast'

elif stat == 'param':
    stat = 'Parameter'

elif stat == 'filter':
    stat = 'Filter'

lag = str(range(1, 52, 5)[lag])

fig.legend(line_list, name_list, fontsize=24, ncol=5, loc='upper center')
plt.figtext(.0575, .83, stat + ' RMSE', horizontalalignment='left', verticalalignment='center', fontsize=24)
plt.figtext(.9225, .83, stat + ' spread', horizontalalignment='right', verticalalignment='center', fontsize=24)
plt.figtext(.50, .03, r'Lag length', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.50, .88, r'Optimally tuned inflation, ensemble size ' + str(ens) + ', lag ' + lag + ', parameter walk std ' +\
        str(wlk), horizontalalignment='center', verticalalignment='center', fontsize=24)

plt.show()
