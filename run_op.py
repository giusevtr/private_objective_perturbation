
from __future__ import print_function
import sys
import numpy as np
import pandas as pd
import itertools
import time

from gurobipy import *
from solvers.OP_MIP import OP_MIP
from dataset.get_dataset import get_dataset


RESULTS_DIR="results"


def create_directory(directory_name):
    try:
      os.stat(directory_name)
    except:
      os.mkdir(directory_name)

def save_output(dataset_name, eps, acc, seconds):
    create_directory(RESULTS_DIR)
    result_path = os.path.join(RESULTS_DIR, "{}.csv".format(dataset_name))

    with  open(result_path, mode='a+') as f:
        row = "{},{:.2f},{:.2f}\r\n".format(eps, acc, seconds)
        f.write(row)


if __name__=="__main__":
    assert len(sys.argv) == 2, "provide dataset name"
    dataset_name = sys.argv[1]
    run_average = 10
    epsilon_schedule = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]

    # X,y = get_dataset("sync_5,10000")
    X,y = get_dataset(dataset_name)
    N = X.shape[0]

    solver = OP_MIP(X, y, 1)
    for eps in epsilon_schedule:
        delta =  1.0 / N**2
        acc_sum = 0
        time_sum = 0
        for _ in range(run_average):
            star_time = time.time()
            solver.set_noise(eps, delta)
            w_start, acc = solver.get_solution()
            acc_sum += acc
            elapsed_time = time.time()-star_time
            time_sum += elapsed_time
            save_output(dataset_name, eps, acc, elapsed_time)

        print("====================================")
        print("average:")
        print("eps: {}\tacc: {:.2f}\ttime: {:.2f}".format(eps, acc_sum / run_average, time_sum / run_average))
        print("====================================")
