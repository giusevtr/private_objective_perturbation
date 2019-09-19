import os
import sys
import csv
import numpy as np
from sklearn import preprocessing
from utils.utils_download import download_extract
from utils.utils_preprocessing import convert_to_binary, normalize_rows, format_output
import pandas as pd
import sklearn
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


    np.save(os.path.join(output_location, FILENAME_X), all_data[:, :-1])
    np.save(os.path.join(output_location, FILENAME_Y), all_data[:, -1])

if __name__=="__main__":
    if len(sys.argv) == 3:
        preprocess(sys.argv[1], sys.argv[2])
    else:
        preprocess()
