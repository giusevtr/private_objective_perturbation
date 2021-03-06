
from __future__ import print_function
import sys
import numpy as np
import pandas as pd
import itertools
import time
from sklearn.linear_model import LogisticRegression
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
    if len(sys.argv) == 4:
        dataset_name = sys.argv[1]
        tau = float(sys.argv[2])
        run_average = int(sys.argv[3])
    else:
        dataset_name = input("Enter dataset name: ")
        tau = float(input("Enter tau: "))
        run_average = int(input("Enter run average: "))
    # epsilon_schedule = [1]
    # epsilon_schedule = [0.001, 0.1, 0.3, 0.5, 0.7, 1]
    # epsilon_schedule = [0.001, 0.01, 0.1]
    epsilon_schedule = [0.01, 0.1]

    # X,y = get_dataset("sync_5,10000")
    X,y = get_dataset(dataset_name)
    N = X.shape[0]
    D = X.shape[1]

    # classifier = LogisticRegression(penalty='none', fit_intercept=False)
    classifier = LogisticRegression(C=100000000000000, fit_intercept=False)
    classifier.fit(X, y)
    predicted_labels = classifier.predict(X)
    print(predicted_labels)
    eq = np.equal(y, predicted_labels)
    eq = eq.astype(float)
    accuracy = np.mean(eq)
    print("Scikit-learn classifier got accuracy {0}".format(accuracy))

    print("N = ", N)
    print("D = ", D)
    print("tau = ", tau)

    solver = OP_MIP(X, y, dataset_name, tau)
    for eps in epsilon_schedule:
        print("epsilon =========>>", eps)
        print("eps acc sec")
        delta =  1.0 / N**2
        acc_sum = 0
        time_sum = 0
        for _ in range(run_average):
            solver.set_noise(eps, delta)
            w_start, acc = solver.get_solution()
            acc_sum += acc
            time_sum += solver.run_time
            # print()
            print("{} {:.3f} {:.3f}s".format(eps, acc, solver.run_time), end="{}\n".format("                                           "))
            save_output(dataset_name, eps, acc, solver.run_time)

        print("average:")
        print("eps: {}\tacc: {:.3f}\ttime: {:.3f}".format(eps, acc_sum / run_average, time_sum / run_average))
        print("========================================================================")
