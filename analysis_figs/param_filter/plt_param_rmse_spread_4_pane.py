import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import Normalize
from matplotlib import rc
from matplotlib.colors import LogNorm
import glob
import ipdb
from matplotlib.colors import LogNorm

method = 'etks'
tanl = 0
wlk = 0.01

f = open('./processed_param_rmse_spread_diffusion_000_param_wlk_' + str(wlk).ljust(4, '0') + '_nanl_40000_tanl_0.05_burn_5000.txt', 'rb')
tmp = pickle.load(f)
f.close()


state_rmse = tmp[method + '_state_rmse']
state_spread = tmp[method + '_state_spread']
param_rmse = tmp[method + '_param_rmse']
param_spread = tmp[method + '_param_spread']


fig = plt.figure()
ax2 = fig.add_axes([.935, .520, .01, .427])
ax5 = fig.add_axes([.935, .080, .01, .427])

ax1 = fig.add_axes([.504, .08, .425, .427])
ax0 = fig.add_axes([.070, .08, .425, .427])
ax4 = fig.add_axes([.504, .520, .425, .427])
ax3 = fig.add_axes([.070, .520, .425, .427])


color_map_state = sns.color_palette("husl", 101)
color_map_params = sns.color_palette("cubehelix", 100)


sns.heatmap(state_rmse, linewidth=0.5, ax=ax3, cbar_ax=ax2, vmin=0.01, vmax=1.0, cmap=color_map_state)
sns.heatmap(state_spread, linewidth=0.5, ax=ax4, vmin=0.01, vmax=1.0, cmap=color_map_state, cbar=False)
sns.heatmap(param_rmse, linewidth=0.5, ax=ax0, cbar_ax=ax5,  vmin=0.00001, vmax=1.0, cmap=color_map_params, norm=LogNorm())
sns.heatmap(param_spread, linewidth=0.5, ax=ax1, vmin=0.00001, vmax=1.0, cmap=color_map_params, cbar=False, norm=LogNorm())


ax2.tick_params(
        labelsize=20)

ax5.tick_params(
        labelsize=20)

ax1.tick_params(
        labelsize=20,
        labelleft=False,
        left=False,
        right=True)

ax0.tick_params(
        labelsize=20)

ax2.tick_params(
        labelsize=20,
        labelleft=False,
        labelright=True,
        right=True,
        left=False)

ax3.tick_params(
        labelsize=20,
        labelbottom=False)

ax4.tick_params(
        labelleft=False,
        labelbottom=False)

x_labs = []
for i in range(14,44,3):
    x_labs.append(str(i))

y_labs = []
y_vals = np.linspace(1.0, 1.2, 21)
for i in range(len(y_vals)):
    if i % 6 == 0:
        y_labs.append(str(np.around(y_vals[i],2)))
    else:
        y_labs.append('')


y_labs = y_labs[::-1]

ax1.set_xticks(range(0,29,3))
ax0.set_xticks(range(0,29,3))
ax1.set_xticklabels(x_labs, ha='left')
ax0.set_xticklabels(x_labs, ha='left')
ax1.set_ylim([21,0])
ax0.set_ylim([21,0])
ax0.set_yticks(range(0,22))
ax0.set_yticklabels(y_labs, va='center')
ax3.set_yticks(range(0,22))
ax3.set_yticklabels(y_labs, va='center')
ax1.set_yticks(range(0,22))
#ax1.set_yticklabels(y_labs, va='top', rotation='90')
plt.figtext(.2, .96, 'Analysis RMSE', horizontalalignment='center', verticalalignment='center', fontsize=22)
plt.figtext(.8, .96, 'Analysis spread', horizontalalignment='center', verticalalignment='center', fontsize=22)
plt.figtext(.03, .7335, r'State vector', horizontalalignment='left', verticalalignment='center', fontsize=22, rotation='90')
plt.figtext(.03, .2935, r'Param vector', horizontalalignment='left', verticalalignment='center', fontsize=22, rotation='90')
plt.figtext(.02, .52, r'Inflation level', horizontalalignment='right', verticalalignment='center', fontsize=22, rotation='90')
plt.figtext(.50, .02, r'Number of samples', horizontalalignment='center', verticalalignment='center', fontsize=22)
plt.figtext(.5, .965, method + '- Observation interval ' + str(tanl) + ' Param wlk ' + str(wlk), horizontalalignment='center', verticalalignment='bottom', fontsize=24)


plt.show()
