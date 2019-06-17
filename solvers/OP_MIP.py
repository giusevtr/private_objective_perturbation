import numpy as np
import sys,os
from gurobipy import *
from solvers.utils import  get_max_X_l2_norm, MIP_progress_bar, verify_MIP

## MIP solver
class OP_MIP:
    def __init__(self, X, y, tau):
        _eps = 0.00001
        self.X = X
        self.y = y
        self.tau = tau
        INT_TOL = 1e-5
        self.N, self.D = X.shape
        self.max_x_l2_norm = get_max_X_l2_norm(X)
        self.l2_diameter =np.sqrt(self.D)
        DOT_LIM = self.l2_diameter * self.max_x_l2_norm +1e-3
        self.THRESHOLD = DOT_LIM*INT_TOL

        ########################################################
        ## Discretized decision space
        ########################################################
        B = self.l2_diameter/tau
        self.W_tau = np.sort(np.append(-np.arange(self.tau, B+1e-9,self.tau), np.arange(0, B+1e-9, self.tau)))
        self.W_tau_size = self.W_tau.shape[0]
        self.W_tau_sq = np.arange(0, B**2+1e-9, self.tau**2)
        self.W_tau_sq_size = self.W_tau_sq.shape[0]

        ########################################################
        ## Gurobi Parameters
        ########################################################
        self.e = {}
        self.w = {}
        self.n = {}
        self.r = {}
        self.q = {}
        self.model = Model("ERM")
        self.model.setParam("OutputFlag", 0)
        # model.setParam("TimeLimit", 5*60)
        # model.setParam("MIPFocus", 2)
        self.model.setParam("IntFeasTol", INT_TOL) ## Default value: 1e-5

        ########################################################
        ## Define variables
        ########################################################
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

        self.model.addConstr(    quicksum(self.sqrt_x[i]*self.q[i] for i in range(self.W_tau_sq_size)) ==
                                quicksum(self.W_tau[j]**2*self.r[i,j] for i in range(self.D) for j in range(self.W_tau_size))  )

        ## 0/1 loss constraints
        for i in range(self.N):
            if self.y[i] >0:
                self.model.addConstr(quicksum(self.w[j]*self.X[i][j] for j in range(self.D)) + DOT_LIM*self.e[i]  >= 2*self.THRESHOLD + 1e-9)
            else:
                self.model.addConstr(quicksum(self.w[j]*self.X[i][j] for j in range(self.D)) <= DOT_LIM*self.e[i])

    def set_noise(self, eps, delta):
        c = 7*np.sqrt(np.log(1.0 / delta))*self.l2_diameter**2/(self.tau**2 * eps)
        noise = np.random.normal(0, c , self.D+1)
        ## update objective
        obj1 = quicksum(self.e[i] for i in range(self.N))
        obj2 = quicksum(noise[i]*self.w[i] for i in range(self.D))/self.l2_diameter
        obj3 = noise[self.D] * quicksum(self.sqrt_y[i]*self.q[i] for i in range(self.W_tau_sq_size))/self.l2_diameter
        self.model.setObjective(obj1-obj2-obj3, GRB.MINIMIZE)

    ##
    def get_solution(self):
        self.model.optimize(MIP_progress_bar)
        w_start = np.array([self.w[j].X for j in range(self.D)])
        y_pred = np.sign(np.dot(self.X, w_start)-self.THRESHOLD).astype(int)
        y_pred = np.array([b if b !=0 else -1 for b in y_pred])
        acc = (self.y == y_pred).sum()/len(self.y)
        return w_start, acc
