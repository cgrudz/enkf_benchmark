import numpy as np
import pickle
import glob
import ipdb

tanl = 0.05
nanl = 40000
burn = 5000
diffusion = 0
wlk = 0.01
shift = 1

method_list = ['etks']
data = {
        'etks_filter_rmse': np.zeros([21, 29]),
        'etks_filter_spread': np.zeros([21, 29]),
        'etks_smooth_rmse': np.zeros([21, 29]),
        'etks_smooth_spread': np.zeros([21, 29]),
        'etks_param_rmse': np.zeros([21, 29]),
        'etks_param_spread': np.zeros([21, 29]),
       }

def process_data(fnames):
    # loop columns
    for j in range(29):        
        #loop rows
        for i in range(21):
            f = open(fnames[i + j*21], 'rb')
            tmp = pickle.load(f)
            f.close()
            
            filter_state_rmse = tmp['filt_rmse']
            filter_state_spread = tmp['filt_spread']
            smooth_state_rmse = tmp['anal_rmse']
            smooth_state_spread = tmp['anal_spread']
            param_rmse = tmp['param_rmse']
            param_spread = tmp['param_spread']

            data[method + '_filter_rmse'][20 - i, j] = np.mean(filter_state_rmse[burn: nanl+burn])
            data[method + '_filter_spread'][20 - i, j] = np.mean(filter_state_spread[burn: nanl+burn])
            data[method + '_smooth_rmse'][20 - i, j] = np.mean(smooth_state_rmse[burn: nanl+burn])
            data[method + '_smooth_spread'][20 - i, j] = np.mean(smooth_state_spread[burn: nanl+burn])
            data[method + '_param_rmse'][20 - i, j] = np.mean(param_rmse[burn: nanl+burn])
            data[method + '_param_spread'][20 - i, j] = np.mean(param_spread[burn: nanl+burn])

for method in method_list:
    fnames = sorted(glob.glob('../smoother_param_data/data/' + method + '/*diffusion_' + str(diffusion).zfill(3) + \
                              '*_param_wlk_' + str(wlk).ljust(4, '0') + '_nanl_' + str(nanl+burn) +  '_tanl_' + str(tanl) + '*' ))

    process_data(fnames)


f = open('./processed_param_rmse_spread_diffusion_' + str(diffusion).zfill(3) +\
         '_param_wlk_' + str(wlk).ljust(4, '0') + '_nanl_' + str(nanl) +\
         '_tanl_' + str(tanl) + '_burn_' + str(burn) + '.txt', 'wb')

pickle.dump(data, f)
f.close()
