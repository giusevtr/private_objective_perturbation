import numpy as np
import sys,os
from gurobipy import *
from solvers.utils import *
import time

## MIP solver
class OP_MIP:
    def __init__(self, X, y, dataset_name, tau):
        _eps = 0.00001
        self.X = X
        self.y = y
        self.dataset_name = dataset_name
        self.tau = tau
        self.INT_TOL = 1e-5
        self.N, self.D = X.shape
        self.max_x_l2_norm = get_max_X_l2_norm(X)
        self.l2_diameter = np.sqrt(self.D)
        # self.l2_diameter = self.D
        self.DOT_LIM = self.l2_diameter * self.max_x_l2_norm +1e-3
        self.THRESHOLD = self.DOT_LIM*self.INT_TOL

        print("DOT_LIM = ", self.DOT_LIM)
        print("THRESHOLD = ", self.THRESHOLD)
        ########################################################
        ## state variables
        ########################################################
        self.run_time = 0

        ########################################################
        ## Discretized decision space
        ########################################################
        B = self.l2_diameter
        self.W_tau = np.sort(np.append(-np.arange(self.tau, B+1e-9,self.tau), np.arange(0, B+1e-9, self.tau)))
        self.W_tau_size = self.W_tau.shape[0]
        self.W_tau_sq = np.arange(0, B**2+1e-9, self.tau**2)
        self.W_tau_sq_size = self.W_tau_sq.shape[0]

        print("W_tau.size = ",      self.W_tau_size)
        print("W_tau_sq.size = ",   self.W_tau_sq_size)
        ########################################################
        ## Gurobi Parameters
        ########################################################
        self.e = {}
        self.w = {}
        self.n = {}
        self.r = {}
        self.q = {}
        self.noise_vector = {}
        self.model = Model("ERM")
        self.model.setParam("OutputFlag", 0)
        # model.setParam("TimeLimit", 5*60)
        # model.setParam("MIPFocus", 2)
        self.model.setParam("IntFeasTol", self.INT_TOL) ## Default value: 1e-5

        ########################################################
        ## Define variables
        ########################################################

        # print("w_tau = ", self.W_tau)
        # print("w_tau_sq = ", self.W_tau_sq)
        self.sqrt_x = [x for x in self.W_tau_sq] ## All possible values that ||w||^2 can take
        self.sqrt_y = [np.sqrt(self.l2_diameter**2-x+1e-9) for x in self.W_tau_sq] ## sqrt{D^2-||w||^2}

        for i in range(self.W_tau_sq_size):
            self.q[i] = self.model.addVar(vtype=GRB.BINARY, name="q_{}".format(i))
        for i in range(self.N):
            self.e[i] = self.model.addVar(vtype=GRB.BINARY, name="e_{}".format(i))
        for i in range(self.D):
            self.w[i] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=-self.l2_diameter, ub=self.l2_diameter, name="w_{}".format(i))
        for k in range(self.D):
            for i in range(self.W_tau_size):
                self.r[k, i] = self.model.addVar(vtype=GRB.BINARY, name="r_({},{})".format(k,i))
        self.model.update()


        ########################################################
        ## Constraints
        ########################################################
        ## Binary variables
        self.model.addConstr(quicksum(self.q[i] for i in range(self.W_tau_sq_size)) == 1)

        ## Define w_i
        for i in range(self.D):
            self.model.addConstr(quicksum(self.r[i, j] for j in range(self.W_tau_size)) == 1)
            self.model.addConstr(quicksum(self.W_tau[j]*self.r[i, j] for j in range(self.W_tau_size)) == self.w[i])

        self.model.addConstr(quicksum(self.sqrt_x[i]*self.q[i] for i in range(self.W_tau_sq_size)) ==
                                quicksum(self.W_tau[j]**2*self.r[i,j] for i in range(self.D) for j in range(self.W_tau_size))  )

        ## 0/1 loss constraints
        for i in range(self.N):
            if self.y[i] >0:
                self.model.addConstr(quicksum(self.w[j]*self.X[i][j] for j in range(self.D)) + self.DOT_LIM*self.e[i]  >= 2*self.THRESHOLD + 1e-9)
            else:
                self.model.addConstr(quicksum(self.w[j]*self.X[i][j] for j in range(self.D)) <= self.DOT_LIM*self.e[i])

        ########################################################
        ## Objective
        ########################################################
        self.set_tuning()

    def set_noise(self, eps, delta):
        c = 7*np.sqrt(np.log(1.0 / delta))*self.l2_diameter**2/(2*self.tau**2*eps)
        # c = 7*np.sqrt(np.log(1.0 / delta))*self.l2_diameter/(2*self.tau**2*eps)
        # c = 7*np.sqrt(np.log(1.0 / delta))/(2*self.tau**2*eps)
        noise = np.random.normal(0, c , self.D+1)
        ## update objective
        obj1 = quicksum(self.e[i] for i in range(self.N))
        obj2 = quicksum(noise[i]*self.w[i] for i in range(self.D))/self.l2_diameter
        obj3 = noise[self.D] * quicksum(self.sqrt_y[i]*self.q[i] for i in range(self.W_tau_sq_size))/self.l2_diameter
        self.model.setObjective(obj1-obj2-obj3, GRB.MINIMIZE)
        # self.model.setObjective(obj1, GRB.MINIMIZE)

    ##
    def get_solution(self):
        load_solution(self.model, self.dataset_name, self.tau)
        star_time = time.time()
        self.model.optimize(MIP_progress_bar)
        self.run_time = time.time() - star_time
        w_start = np.array([self.w[j].X for j in range(self.D)])
        verify_MIP(self, self.model, w_start, debug=False)
        y_pred = np.sign(np.dot(self.X, w_start)-self.THRESHOLD).astype(int)
        y_pred = np.array([b if b !=0 else -1 for b in y_pred])
        acc = (self.y == y_pred).sum()/len(self.y)
        save_solution(self.model, self.dataset_name, self.tau)
        return w_start, acc

    def set_tuning(self):
        # Tune
        prm_file_path = "tuning/tune0.prm"
        exists = os.path.isfile(prm_file_path)
        if os.path.isfile(prm_file_path):
            print("reading parameter file ", prm_file_path)
            self.model.read(prm_file_path)
        else:
            self.model.tune()
            print("tune results ============> ", self.model.tuneResultCount)
            for i in range(self.model.tuneResultCount):
                self.model.getTuneResult(i)
                self.model.write('tuning/tune'+str(i)+'.prm')
