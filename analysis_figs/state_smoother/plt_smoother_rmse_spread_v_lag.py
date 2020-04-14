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
method_list = ['etks']
analysis = ['filter', 'smooth']
markerlist = ['o','v']

fig = plt.figure()
ax1 = fig.add_axes([.530, .10, .40, .76])
ax0 = fig.add_axes([.050, .10, .40, .76])


f = open('processed_shift_equal_lag_False_smoother_state_rmse_spread_nanl_40000_tanl_0.05_burn_5000.txt', 'rb')
tmp = pickle.load(f)
f.close()


def find_optimal_values(rmse, spread):

    rmse_min_vals = np.amin(rmse, axis=1)
    spread_vals = np.zeros(len(rmse_min_vals))

    for i in range(len(rmse_min_vals)):
        if math.isnan(rmse_min_vals[i]):
            spread_vals[i] = math.nan
            
        else:
            indx = rmse[i, :] == rmse_min_vals[i]
            spread_vals[i] = spread[i, indx]


    return [rmse_min_vals, spread_vals]


line_list = []
j = 0
for meth in method_list:
    for anal in analysis:   
        rmse = tmp[meth + '_' + anal + '_rmse']
        spread = tmp[meth + '_' + anal + '_spread']
        [rmse, spread] = find_optimal_values(rmse, spread)
        
        l, = ax0.plot(range(1, 52, 5), rmse, marker=markerlist[j], linewidth=2, markersize=10)
        ax1.plot(range(1, 52, 5),spread, marker=markerlist[j], linewidth=2, markersize=10)
        line_list.append(l)

        j+=1


ax1.tick_params(
        labelsize=20,
        labelleft=False,
        left=False)

ax0.tick_params(
        labelsize=20)

ax1.set_ylim([0.01,0.3])
ax0.set_ylim([0.01,0.3])
ax1.set_xlim([0.5, 51.5])
ax0.set_xlim([0.5, 51.5])
#ax0.set_yscale('log')
#ax1.set_yscale('log')

fig.legend(line_list, ['ETKS Filter', 'ETKS Smoother'], fontsize=24, ncol=5, loc='upper center')
plt.figtext(.2525, .88, 'RMSE', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.7225, .88, 'Spread', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.50, .03, r'Lag length', horizontalalignment='center', verticalalignment='center', fontsize=24)

plt.show()
