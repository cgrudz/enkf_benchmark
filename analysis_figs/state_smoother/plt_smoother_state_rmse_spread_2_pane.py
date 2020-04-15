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
stat = 'fore'
tanl = 0.05
shift = False

f = open('./processed_shift_equal_lag_' + str(shift) + '_smoother_state_rmse_spread_nanl_40000_tanl_0.05_burn_5000.txt', 'rb')
tmp = pickle.load(f)
f.close()


rmse = tmp[method + '_' + stat + '_rmse']
spread = tmp[method + '_' + stat + '_spread']

fig = plt.figure()
ax3 = fig.add_axes([.460, .13, .02, .70])
ax2 = fig.add_axes([.940, .13, .02, .70])
ax1 = fig.add_axes([.530, .13, .390, .70])
ax0 = fig.add_axes([.060, .13, .390, .70])


color_map = sns.color_palette("husl", 101)
max_scale = 0.30
min_scale = 0.01


sns.heatmap(rmse, linewidth=0.5, ax=ax0, cbar_ax=ax3, vmin=min_scale, vmax=max_scale, cmap=color_map)
sns.heatmap(spread, linewidth=0.5, ax=ax1, cbar_ax=ax2, vmin=min_scale, vmax=max_scale, cmap=color_map)


ax2.tick_params(
        labelsize=20)

ax1.tick_params(
        labelsize=20,
        labelleft=False)

ax0.tick_params(
        labelsize=20)

ax3.tick_params(
        labelsize=20,
        labelleft=False,
        labelright=True,
        right=True,
        left=False)
ax2.tick_params(
        labelsize=20,
        labelleft=False,
        labelright=True,
        right=True,
        left=False)


x_labs = []
for i in range(14,42,3):
    x_labs.append(str(i))

y_labs = []
y_vals = np.arange(1,52, 5)
for i in range(len(y_vals)):
    if i % 1 == 0:
        y_labs.append(str(y_vals[i]))
    else:
        y_labs.append('')


y_labs = y_labs[::-1]

ax1.set_xticks(range(0,28,3))
ax0.set_xticks(range(0,28,3))
ax1.set_xticklabels(x_labs)
ax0.set_xticklabels(x_labs)
ax1.set_ylim([9,1])
ax0.set_ylim([9,1])
ax0.set_yticks(range(11))
ax0.set_yticklabels(y_labs, va='bottom')
ax1.set_yticks(range(11))

if stat == 'smooth':
    stat = 'Smoother'

elif stat == 'filter':
    stat = 'Filter'

elif stat == 'fore':
    stat = 'Forecast'

plt.figtext(.2525, .87, stat + ' RMSE', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.7225, .87, stat + ' spread', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.015, .52, r'Lag length', horizontalalignment='center', verticalalignment='center', fontsize=24, rotation='90')
plt.figtext(.50, .04, r'Ensemble size', horizontalalignment='center', verticalalignment='center', fontsize=24)
plt.figtext(.5, .95, method + ' ' + stat +  ' optimally tuned inflation, shift equal lag ' + str(shift),
        horizontalalignment='center', verticalalignment='center', fontsize=24)


plt.show()
