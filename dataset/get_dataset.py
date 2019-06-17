import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

CACHE_LOCATION = 'dataset/data_cache'
OUTPUT_LOCATION = 'dataset/data'



def get_synthetic_data( dim, n, flag='sign'):
    w = pd.Series(np.random.randint(2, size=dim+1))
    if flag == 'sign':
        w = 2*w-1

    # generate random features
    # X = pd.DataFrame([2*np.random.randint(2, size=dim)-1 for _ in range(n)])## x_i \in {-1,1}
    X = pd.DataFrame([np.random.randint(2, size=dim) for _ in range(n)]) ## x_i \in {0,1}
    X = X.assign(intercept=pd.DataFrame([1] * n))
    X.columns = range(X.shape[1])

    y = np.sign(np.dot(X, w))
    y = pd.Series([b if b != 0 else -1 for b in y])

    return X.values, y.values, w

def preprocess(dataset_name):
    print("preprocessing dataset {}".format(dataset_name))
    cmd = "python dataset/{}.py {} {}".format(dataset_name, CACHE_LOCATION, OUTPUT_LOCATION)
    os.system(cmd)

def get_dataset(dataset_name):
    if dataset_name[:4] == "sync":
        param = dataset_name[5:].split(",")
        dim = int(param[0])
        data_size = int(param[1])
        X,y,w = get_synthetic_data(dim, data_size)
        return X, y

    dataset_location = "{}".format(OUTPUT_LOCATION)
    if not os.path.isfile(os.path.join(dataset_location, '{}_processed_x.npy'.format(dataset_name))):
        preprocess(dataset_name)
    features = np.load(os.path.join(dataset_location, '{}_processed_x.npy'.format(dataset_name)))
    features = features.astype(float)
    labels = np.load(os.path.join(dataset_location, '{}_processed_y.npy'.format(dataset_name)))
    labels = labels.astype(float)
    return features, labels
