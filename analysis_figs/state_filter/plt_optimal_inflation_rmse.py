import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import Normalize
from matplotlib import rc
from matplotlib.colors import LogNorm
import glob
#import ipdb
import math

tanl = 0.05
obs_un = 1.0
methods = ['enkf', 'etkf', 'enks', 'etks', 'ienkf']
markerlist = ['o', 'v', '>', 'X', 'd']

fig = plt.figure()
ax1 = fig.add_axes([.530, .10, .40, .76])
ax0 = fig.add_axes([.050, .10, .40, .76])


f = open('processed_rmse_spread_diffusion_000_nanl_40000_tanl_0.05_burn_5000.txt', 'rb')
tmp = pickle.load(f)
f.close()


def find_optimal_values(rmse, spread):

    rmse_min_vals = np.amin(rmse, axis=0)
    spread_vals = np.zeros(len(rmse_min_vals))

    for i in range(len(rmse_min_vals)):
        if math.isnan(rmse_min_vals[i]):
            spread_vals[i] = math.nan
            
        else:
            indx = rmse[:, i] == rmse_min_vals[i]
            spread_vals[i] = spread[indx, i]


    return [rmse_min_vals, spread_vals]


line_list = []
for i in range(5):
    
    method = methods[i]
    rmse = tmp[method + '_rmse']
    spread = tmp[method + '_spread']
    [rmse, spread] = find_optimal_values(rmse, spread)
    
    l, = ax0.plot(range(14, 43), rmse, marker=markerlist[i], linewidth=2, markersize=10)
    ax1.plot(range(14, 43),spread, marker=markerlist[i], linewidth=2, markersize=10)
    line_list.append(l)

ax1.tick_params(
        labelsize=20,
        labelleft=False,
        left=False)

ax0.tick_params(
        labelsize=20)

ax1.set_ylim([0.10,1.0])
ax0.set_ylim([0.10,1.0])
ax1.set_xlim([13.5, 42.5])
ax0.set_xlim([13.5, 42.5])
ax0.set_yscale('log')
ax1.set_yscale('log')

fig.legend(line_list, methods, fontsize=24, ncol=5, loc='upper center')
plt.figtext(.2525, .88, 'Analysis RMSE', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.7225, .88, 'Analysis spread', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.50, .03, r'Ensemble size', horizontalalignment='center', verticalalignment='center', fontsize=24)

plt.show()
