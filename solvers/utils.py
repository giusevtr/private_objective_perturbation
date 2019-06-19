import sys
import numpy as np
from gurobipy import *
import tempfile

def get_max_X_l2_norm(X):
    N = X.shape[0]
    max_x_l2_norm = 0
    for i in range(N):
        xnorm = np.linalg.norm(X[i, :])
        max_x_l2_norm = max(max_x_l2_norm, xnorm)
    return max_x_l2_norm

def MIP_progress_bar(model, where):
    bar_len = 20
    if where == GRB.Callback.MIP:
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        gap = abs(objbst - objbnd)/ (1.0 + abs(objbst))
        sys.stdout.write("[")
        for i in range(bar_len):
            c = "-" if i < bar_len*(1-gap) else " "
            sys.stdout.write(c)
        sec =  model.cbGet(GRB.Callback.RUNTIME)
        sys.stdout.write("]\tgap={:.2f}%\ttime={:.2f}\r".format(100*gap,sec))
        sys.stdout.flush()

def verify_MIP(model, w_star, X, y):
    N,D = X.shape
    assert model.getAttr(GRB.Attr.Status) == GRB.OPTIMAL, "no optimal solution found"

    print("objective value: ", model.objVal)
    print("w_star = ", w_star.round(4))
    mistakes = 0
    regu_term = 0
    debug_sqrt_x = 0
    debug_r = 0
    for v in model.getVars():
        if v.varName[:2] =="r_":
            temp1,temp2 = v.varName[3:-1].split(',')
            i = int(temp2)
            debug_r += v.x * W_tau[i]**2

        if v.varName[:2] =="q_":
            i = int(v.varName[2:])
            debug_sqrt_x += v.x * sqrt_x[i]
            regu_term += v.x*sqrt_y[i]

        if v.varName[0] =="e":
            mistakes += v.x
            i = int(v.varName[2:])
            p = np.dot(w_star, X.iloc[i])
            assert np.abs(p) < DOT_LIM, "dot error. p = {}".format(p)
            if y.iloc[i]>0:
                if v.x < 0.5:## e_i = 0
                    assert p>THRESHOLD, "constraint failed: y = {}, p = {}".format(y.iloc[i], p)
                else: ## e_i = 1
                    assert p<=THRESHOLD, "constraint failed: y = {}, p = {}".format(y.iloc[i], p)
            else:
                if v.x < 0.5:## e_i = 0
                    assert p<=THRESHOLD, "constraint failed: y = {}, p = {}".format(y.iloc[i], p)
                else:## e_i = 1
                    assert p>THRESHOLD, "constraint failed: y = {}, p = {}".format(y.iloc[i], p)
                # print("w*x_{} = {:5f}\ty={}".format(i, p, y.iloc[i]))
    print("residual_term^2 = \t{:.4f}".format(regu_term**2))
    norm2 = np.linalg.norm(w_star)**2
    print("||w_start||_2^2 = \t{:.4f}".format(norm2))
    print("mistakes = {}".format(np.round(mistakes)))


    print("{:10s}|{:10d}".format('mistakes', mistakes))
    print("{:10s}|{:10d}".format('residual', residual_term))
    print("{:10s}|{:10d}".format('norm2', norm2))

    assert np.abs(debug_sqrt_x-debug_r)<1e-6, "l2_norm problem"
    assert np.abs(regu_term**2 + norm2 - l2_diameter**2)<1e-6, "diff = {}".format(regu_term**2 + norm2 - l2_diameter**2 )


    print("======================== Done debug")
    # if model.getAttr(GRB.Attr.Status) == GRB.OPTIMAL:


def save_solution(model, dataset_name, tau):
    sol_dir = "solvers/solutions/{}_{}".format(dataset_name, tau)
    create_directory('solvers/solutions')
    create_directory(sol_dir)
    tf = tempfile.NamedTemporaryFile(dir=sol_dir, suffix=".mst", delete=False)
    model.write(tf.name)
    # print("saving ", tf.name)


def load_solution(model, dataset_name, tau):
    sol_dir = "solvers/solutions/{}_{}".format(dataset_name, tau)
    create_directory('solvers/solutions')
    create_directory(sol_dir)
    for fn in os.listdir(sol_dir):
        if fn.endswith('.mst') or fn.endswith('.sol'):
            path = os.path.join(sol_dir, fn)
            # print("loading ", fn)
            # GRBread(model, fn)
            model.read(path)


def create_directory(directory_name):
    try:
      os.stat(directory_name)
    except:
      os.mkdir(directory_name)
