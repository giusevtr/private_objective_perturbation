import os
import sys
import csv
import numpy as np
from sklearn import preprocessing
from utils.utils_download import download_extract
from utils.utils_preprocessing import convert_to_binary, normalize_rows, format_output
import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/'
# use_col = [0, 2, 4, 5, 8, 9, 10, 11, 12]
# use_col = [0, 2, 4, 9, 10, 11, 12]
# use_col = [5,7,8,9,10,11,12]

use_col = [0,2,4,10,11,12]
use_col = [
# 0 , # age: continuous.
1 , # workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# 2 , # fnlwgt: continuous.
3 , # education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# 4 , # education-num: continuous.
5 , # marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
6 , # occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
7 , # relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
8 , # race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
9 , # sex: Female, Male.
# 10, # capital-gain: continuous.
# 11, # capital-loss: continuous.
# 12, # hours-per-week: continuous.
# 13, # native-country
]
db_name = "adult_{}".format("-".join([str(x) for x in use_col]))
# db_name = "adult_small"
FILENAME_X = '{}_processed_x.npy'.format(db_name)
FILENAME_Y = '{}_processed_y.npy'.format(db_name)

# preprocess implemented in numpy

def preprocess(cache_location="dataset/data_cache", output_location="dataset/data"):

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
    np_set = np.vstack((np_train_set, np_test_set))
    # np_set = np.vstack((np_train_set ))


    symbolic_cols = []
    continuous_cols = []
    label_cols = []
    le = preprocessing.LabelEncoder()
    continuous_pos = [0, 2, 4, 10, 11, 12]

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

    if symbolic_cols:
        symbolic_cols = convert_to_binary(symbolic_cols)

    combined_data = np.column_stack(symbolic_cols+continuous_cols)
    final_data = combined_data

    all_data = np.column_stack([final_data, label_cols])
    np.random.shuffle(all_data)

    ## Subsample
    # all_data = all_data[:30000, :]
    # all_data = all_data[:15000, :]
    # all_data = all_data[:7000, :]

    print("Saving")
    print(FILENAME_X)
    print(FILENAME_Y)
    print("rows = ", all_data.shape[0])
    print("cols = ", all_data.shape[1])
    np.save(os.path.join(output_location, FILENAME_X), all_data[:, :-1])
    np.save(os.path.join(output_location, FILENAME_Y), all_data[:, -1])

if __name__=="__main__":
    if len(sys.argv) == 3:
        preprocess(sys.argv[1], sys.argv[2])
    else:
        preprocess()
