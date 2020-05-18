import sys
import pickle
from smoother_exps import classic_state, classic_param, hybrid_state, hybrid_param
from filter_exps import filter_state, filter_param

########################################################################################################################
# Classic smoother parameter estimation
########################################################################################################################
## FUNCTIONALIZED EXPERIMENT CALL OVER PARAMETER MAP
j = int(sys.argv[1])
f = open('./data/input_data/benchmark_smoother_param.txt', 'rb')
data = pickle.load(f)
args = data[j]
f.close()

classic_param(args)
########################################################################################################################
