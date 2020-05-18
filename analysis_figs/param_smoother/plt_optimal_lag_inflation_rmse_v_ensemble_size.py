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
wlk = 0.0100
stat = 'param'
methods = ['enks','etks']
versions = ['classic', 'hybrid']
markerlist = ['o', 'v', '>', 'X', 'd']

fig = plt.figure()
ax1 = fig.add_axes([.530, .10, .40, .71])
ax0 = fig.add_axes([.050, .10, .40, .71])


def find_optimal_values(data, method, stat):
    smooth_rmse = data[method + '_smooth_rmse']
    smooth_spread = data[method + '_smooth_spread']
    stat_rmse = data[method + '_' + stat + '_rmse']
    stat_spread = data[method + '_' + stat + '_spread']

    rmse_min_vals = np.amin(smooth_rmse, axis=0)
    rmse_vals = np.zeros(len(rmse_min_vals))
    spread_vals = np.zeros(len(rmse_min_vals))

    for i in range(len(rmse_min_vals)):
        if math.isnan(rmse_min_vals[i]):
            spread_vals[i] = math.nan
            
        else:
            indx = smooth_rmse[:, i] == rmse_min_vals[i]
            rmse_vals[i] = stat_rmse[indx, i]
            spread_vals[i] = stat_spread[indx, i]


    return [rmse_vals, spread_vals]


line_list = []
name_list = []

for j in range(len(versions)):
    for i in range(len(methods)):
        version = versions[j]
        method = methods[i]
        
        f = open('processed_' + version + '_smoother_param_rmse_spread_nanl_40000_tanl_0.05_burn_5000_wlk_' + str(wlk).ljust(6,'0') + '.txt', 'rb')
        data = pickle.load(f)
        f.close()
        
        [rmse, spread] = find_optimal_values(data, method, stat)
        
        l, = ax0.plot(range(14, 42), rmse, marker=markerlist[i+j*2], linewidth=2, markersize=10)
        ax1.plot(range(14, 42),spread, marker=markerlist[i+j*2], linewidth=2, markersize=10)
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
    ax1.set_ylim([0.05,0.5])
    ax0.set_ylim([0.05,0.5])

ax1.set_xlim([13.5, 42.5])
ax0.set_xlim([13.5, 42.5])

ax0.tick_params(
        labelsize=22)

ax1.tick_params(
        labelsize=22)

if stat == 'smooth':
    stat = 'Smoother'

elif stat == 'fore':
    stat = 'Forecast'

elif stat == 'param':
    stat = 'Parameter'

elif stat == 'filter':
    stat = 'Filter'

fig.legend(line_list, name_list, fontsize=24, ncol=5, loc='upper center')
plt.figtext(.0575, .83, stat + ' RMSE', horizontalalignment='left', verticalalignment='center', fontsize=24)
plt.figtext(.9225, .83, stat + ' spread', horizontalalignment='right', verticalalignment='center', fontsize=24)
plt.figtext(.50, .03, r'Ensemble size', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.50, .88, r'Optimally tuned inflation and lag, parameter walk std ' + str(wlk), horizontalalignment='center', verticalalignment='center', fontsize=24)

plt.show()
