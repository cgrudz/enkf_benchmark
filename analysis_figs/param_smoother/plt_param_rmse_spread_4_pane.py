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
version = 'classic'
tanl = 0
nanl = 45000
burn = 5000
wlk = 0.0100
diff = 0.1
stat = 'filter'

f = open('./processed_'+ version + '_smoother_param_rmse_spread_nanl_40000_tanl_0.05_burn_5000_wlk_' + str(wlk).ljust(6, '0') + '_diff_' + str(diff)+ '.txt', 'rb')
tmp = pickle.load(f)
f.close()


smooth_rmse = tmp[method + '_' + stat + '_rmse']
smooth_spread = tmp[method + '_' + stat + '_spread']
param_rmse = tmp[method + '_param_rmse']
param_spread = tmp[method + '_param_spread']

fig = plt.figure()

ax0 = fig.add_axes([.070, .08, .425, .427])
ax1 = fig.add_axes([.504, .08, .425, .427])
ax5 = fig.add_axes([.935, .08, .01, .427])

ax3 = fig.add_axes([.070, .520, .425, .427])
ax4 = fig.add_axes([.504, .520, .425, .427])
ax2 = fig.add_axes([.935, .520, .01, .427])


color_map_state = sns.color_palette("husl", 101)
color_map_params = sns.color_palette("cubehelix", 100)


sns.heatmap(smooth_rmse, linewidth=0.5, ax=ax3, cbar_ax=ax2, vmin=0.01, vmax=0.3, cmap=color_map_state)
sns.heatmap(smooth_spread, linewidth=0.5, ax=ax4, vmin=0.01, vmax=0.3, cmap=color_map_state, cbar=False)
sns.heatmap(param_rmse, linewidth=0.5, ax=ax0, cbar_ax=ax5,  vmin=0.00001, vmax=1.0, cmap=color_map_params, norm=LogNorm())
sns.heatmap(param_spread, linewidth=0.5, ax=ax1, vmin=0.00001, vmax=1.0, cmap=color_map_params, cbar=False, norm=LogNorm())


ax2.tick_params(
        labelsize=20)

ax5.tick_params(
        labelsize=20)

ax1.tick_params(
        labelsize=20,
        labelleft=False,
        left=False)

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
y_vals = range(1, 52, 5)
j = 0
for i in range(len(y_vals)):
    if j % 2 == 0:
        y_labs.append(str(np.around(y_vals[i],2)))
    else:
        y_labs.append('')
    
    j+=1


y_labs = y_labs[::-1]

ax1.set_xticks(np.arange(0,29,3) + 0.5)
ax0.set_xticks(np.arange(0,29,3) + 0.5)
ax3.set_xticks(np.arange(0,29,3) + 0.5)
ax4.set_xticks(np.arange(0,29,3) + 0.5)
ax1.set_xticklabels(x_labs, ha='center')
ax0.set_xticklabels(x_labs, ha='center')
ax0.set_ylim([11,0])
ax1.set_ylim([11,0])
ax3.set_ylim([11,0])
ax4.set_ylim([11,0])
ax0.set_yticks(np.arange(0,11) + 0.5)
ax0.set_yticklabels(y_labs, va='center')
ax3.set_yticks(np.arange(0,11) + 0.5)
ax3.set_yticklabels(y_labs, va='center')
ax1.set_yticks(np.arange(0,11) + 0.5)
if stat == 'smooth':
    stat = 'Smoother'

elif stat == 'filter':
    stat = 'Filter'

elif stat == 'fore':
    stat = 'Forecast'

if method == 'enks':
    method = 'EnKS'

elif method == 'etks':
    method = 'ETKS'

plt.figtext(.2, .96, stat + ' RMSE', horizontalalignment='center', verticalalignment='center', fontsize=22)
plt.figtext(.8, .96, stat+ ' spread', horizontalalignment='center', verticalalignment='center', fontsize=22)
plt.figtext(.03, .7335, r'State vector', horizontalalignment='left', verticalalignment='center', fontsize=22, rotation='90')
plt.figtext(.03, .2935, r'F parameter', horizontalalignment='left', verticalalignment='center', fontsize=22, rotation='90')
plt.figtext(.02, .52, r'Lag length', horizontalalignment='right', verticalalignment='center', fontsize=22, rotation='90')
plt.figtext(.2, .02, r'Smoother RMSE', horizontalalignment='center', verticalalignment='center', fontsize=22)
plt.figtext(.8, .02, r'Smoother spread', horizontalalignment='center', verticalalignment='center', fontsize=22)
plt.figtext(.50, .02, r'Ensemble size', horizontalalignment='center', verticalalignment='center', fontsize=22)
plt.figtext(.5, .965, method + ' ' + version + ' - parameter walk std ' + str(wlk) + ' diffusion ' + str(diff),
        horizontalalignment='center', verticalalignment='bottom', fontsize=24)


plt.show()
