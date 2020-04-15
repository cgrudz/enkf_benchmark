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
wlk = 0.01
stat = 'param'
methods = ['enkf', 'etkf', 'enks', 'etks', 'ienkf']
markerlist = ['o', 'v', '>', 'X', 'd']

fig = plt.figure()
ax1 = fig.add_axes([.530, .10, .40, .76])
ax0 = fig.add_axes([.050, .10, .40, .76])


f = open('processed_param_rmse_spread_diffusion_000_param_wlk_' + str(wlk).ljust(4,'0') + '_nanl_40000_tanl_0.05_burn_5000.txt', 'rb')
data = pickle.load(f)
f.close()


def find_optimal_values(data, method, stat):
    anal_rmse = data[method + '_anal_rmse']
    anal_spread = data[method + '_anal_spread']
    stat_rmse = data[method + '_' + stat + '_rmse']
    stat_spread = data[method + '_' + stat + '_spread']

    rmse_min_vals = np.amin(anal_rmse, axis=0)
    rmse_vals = np.zeros(len(rmse_min_vals))
    spread_vals = np.zeros(len(rmse_min_vals))

    for i in range(len(rmse_min_vals)):
        if math.isnan(rmse_min_vals[i]):
            spread_vals[i] = math.nan
            
        else:
            indx = anal_rmse[:, i] == rmse_min_vals[i]
            rmse_vals[i] = stat_rmse[indx, i]
            spread_vals[i] = stat_spread[indx, i]


    return [rmse_vals, spread_vals]


line_list = []
for i in range(5):
    
    method = methods[i]
    [rmse, spread] = find_optimal_values(data, method, stat)
    
    l, = ax0.plot(range(14, 43), rmse, marker=markerlist[i], linewidth=2, markersize=10)
    ax1.plot(range(14, 43),spread, marker=markerlist[i], linewidth=2, markersize=10)
    line_list.append(l)

ax1.tick_params(
        labelsize=20,
        labelleft=False,
        left=False,
        labelright=True,
        right=True
        )

ax0.tick_params(
        labelsize=20)

#ax1.set_ylim([0.10,1.0])
#ax0.set_ylim([0.10,1.0])
ax1.set_ylim([10e-20,1.0])
ax0.set_ylim([10e-6,1.0])
ax1.set_xlim([13.5, 42.5])
ax0.set_xlim([13.5, 42.5])
ax0.set_yscale('log')
ax1.set_yscale('log')

if stat == 'anal':
    stat = 'Analysis'

elif stat == 'fore':
    stat = 'Forecast'

ax0.tick_params(
        labelsize=22)

ax1.tick_params(
        labelsize=22)

if stat == 'anal':
    stat = 'Analysis'

elif stat == 'fore':
    stat == 'Forecast'

elif stat == 'param':
    stat = 'Parameter'

fig.legend(line_list, methods, fontsize=24, ncol=5, loc='upper center')
plt.figtext(.2525, .88, stat + ' RMSE', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.7225, .88, stat + ' spread', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.50, .03, r'Ensemble size', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.50, .88, r'Parameter walk std ' + str(wlk), horizontalalignment='center', verticalalignment='center', fontsize=24)

plt.show()
