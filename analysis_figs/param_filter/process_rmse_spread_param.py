import numpy as np
import pickle
import glob
import ipdb

tanl = 0.05
nanl = 40000
burn = 5000
diffusion = 0
wlk = 0.01

method_list = ['enkf', 'etkf', 'enks', 'etks', 'ienkf']
data = {
        'enkf_state_rmse': np.zeros([21, 29]),
        'enkf_state_spread': np.zeros([21, 29]),
        'enkf_param_rmse': np.zeros([21, 29]),
        'enkf_param_spread': np.zeros([21, 29]),
        'etkf_state_rmse': np.zeros([21, 29]),
        'etkf_state_spread': np.zeros([21, 29]),
        'etkf_param_rmse': np.zeros([21, 29]),
        'etkf_param_spread': np.zeros([21, 29]),
        'enks_state_rmse': np.zeros([21, 29]),
        'enks_state_spread': np.zeros([21, 29]),
        'enks_param_rmse': np.zeros([21, 29]),
        'enks_param_spread': np.zeros([21, 29]),
        'etks_state_rmse': np.zeros([21, 29]),
        'etks_state_spread': np.zeros([21, 29]),
        'etks_param_rmse': np.zeros([21, 29]),
        'etks_param_spread': np.zeros([21, 29]),
        'ienkf_state_rmse': np.zeros([21, 29]),
        'ienkf_state_spread': np.zeros([21, 29]),
        'ienkf_param_rmse': np.zeros([21, 29]),
        'ienkf_param_spread': np.zeros([21, 29]),
       }

def process_data(fnames):
    # loop columns
    for j in range(29):        
        #loop rows
        for i in range(21):
            f = open(fnames[i + j*21], 'rb')
            tmp = pickle.load(f)
            f.close()
            
            ana_state_rmse = tmp['state_anal_rmse']
            ana_state_spread = tmp['state_anal_spread']
            ana_param_rmse = tmp['param_anal_rmse']
            ana_param_spread = tmp['param_anal_spread']

            data[method + '_state_rmse'][20 - i, j] = np.mean(ana_state_rmse[burn: nanl+burn])
            data[method + '_state_spread'][20 - i, j] = np.mean(ana_state_spread[burn: nanl+burn])
            data[method + '_param_rmse'][20 - i, j] = np.mean(ana_param_rmse[burn: nanl+burn])
            data[method + '_param_spread'][20 - i, j] = np.mean(ana_param_spread[burn: nanl+burn])

for method in method_list:
    fnames = sorted(glob.glob('../filter_param_data/data/' + method + '/*diffusion_' + str(diffusion).zfill(3) + \
                              '*_param_wlk_' + str(wlk).ljust(4, '0') + '_nanl_' + str(nanl+burn) +  '_tanl_' + str(tanl) + '*' ))

    process_data(fnames)


f = open('./processed_param_rmse_spread_diffusion_' + str(diffusion).zfill(3) +\
         '_param_wlk_' + str(wlk).ljust(4, '0') + '_nanl_' + str(nanl) +\
         '_tanl_' + str(tanl) + '_burn_' + str(burn) + '.txt', 'wb')

pickle.dump(data, f)
f.close()
