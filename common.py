import pickle
import glob

########################################################################################################################
# common auxilliary functions 

def picopen(fname):
    f = open(fname, 'rb')
    tmp = pickle.load(f)
    f.close()

    return tmp

def picwrite(data, fname):
    f = open(fname, 'wb')
    pickle.dump(data, f)
    f.close()
