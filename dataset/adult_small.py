import os
import sys
import csv
import numpy as np
from sklearn import preprocessing
from utils.utils_download import download_extract
from utils.utils_preprocessing import convert_to_binary, normalize_rows, format_output
import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/'
db_name = "adult_small"
FILENAME_X = '{}_processed_x.npy'.format(db_name)
FILENAME_Y = '{}_processed_y.npy'.format(db_name)

# preprocess implemented in numpy

def preprocess(cache_location, output_location):

    np.random.seed(10000019)
    download_extract(url, cache_location, 'adult.data')
    download_extract(url, cache_location, 'adult.test')

    raw_train_set = csv.reader(open(os.path.join(cache_location, 'adult.data')))

    list_train_set = []
    for row in raw_train_set:
        if ' ?' not in row:
            list_train_set.append(row)

    np_train_set = np.array(list_train_set[:-2])

    raw_test_set = csv.reader(open(os.path.join(cache_location, 'adult.test')))

    list_test_set = []
    for row in raw_test_set:
        if ' ?' not in row:
            list_test_set.append(row)

    np_test_set = np.array(list_test_set[1:-2])
    # np_set = np.vstack((np_train_set, np_test_set))
    np_set = np.vstack((np_train_set ))


    symbolic_cols = []
    continuous_cols = []
    label_cols = []
    le = preprocessing.LabelEncoder()
    continuous_pos = [0, 2, 4, 10, 11, 12]
    # use_col = [5,7,8,9,10,11,12]
    use_col = [0, 2, 4, 5, 8, 9, 10, 11, 12]

    for i in range(np.shape(np_set)[1]):
        col = np_set[:, i]
        if i in continuous_pos:
            if i not in use_col : continue
            col = np.array([int(j) for j in col])
            col = col/col.max()
            continuous_cols.append(col)
        elif i != np.shape(np_set)[1]-1:
            if i not in use_col : continue
            col = le.fit_transform(col)
            symbolic_cols.append(col)
        else:
            for j in range(col.shape[0]):
                if col[j] == ' <=50K' or col[j] == ' <=50K.':
                    col[j] = -1
                else:
                    col[j] = 1
            col = np.array([int(j) for j in col])
            label_cols = col.tolist()

    symbolic_cols = convert_to_binary(symbolic_cols)

    combined_data = np.column_stack(symbolic_cols+continuous_cols)
    final_data = combined_data

    all_data = np.column_stack([final_data, label_cols])
    np.random.shuffle(all_data)


    np.save(os.path.join(output_location, FILENAME_X), all_data[:, :-1])
    np.save(os.path.join(output_location, FILENAME_Y), all_data[:, -1])

if __name__=="__main__":
    assert len(sys.argv)==3, "takes two arguments"
    preprocess(sys.argv[1], sys.argv[2])
