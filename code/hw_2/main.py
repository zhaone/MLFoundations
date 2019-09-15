import numpy as np
import sympy

from decision_stump import *


def compute_bound(N):
    dvc = 50
    delta = 0.05

    def origiVC():
        return np.sqrt(8/N*np.log(4*pow(2*N, dvc)/delta))

    def rpVC():
        r1 = np.sqrt(2*np.log(2.0*N*pow(N, dvc))/N)
        r2 = np.sqrt(2/N*np.log(1/delta))
        return r1+r2+1/N

    def pvVC():
        epsilon = sympy.symbols('epsilon')
        r1 = sympy.solve(
            epsilon-sympy.sqrt(1/N*sympy.log(6*((2*N)**dvc)/delta)), epsilon)
        return r1

    def dVC():
        epsilon = sympy.symbols('epsilon')
        r1 = sympy.solve(epsilon-sympy.sqrt(0.5/N*(4*epsilon*(1+epsilon) +
                                                   np.log(4)+dvc*sympy.log(N**2.0)-np.log(delta))), epsilon)
        return r1

    def vVC():
        return np.sqrt(16/N*(np.log(2)+dvc*np.log(N*1.0)-np.log(np.sqrt(delta))))
    # print("origiVC:", origiVC(),
    #         "rpVC", rpVC(),
    #         "pvVC", pvVC(),
    #         "dVC", dVC(),
    #         "vVC", vVC())

def as_3():
    def calc_err(N, d_vc=10, delta=0.05):
        # DO NOT use np.power(), for N*10 is so big that it will overflow
        return np.sqrt((8/N)*np.log(4*pow(2*N, d_vc)/delta))

    candidates = range(420000, 520000, 20000)
    errs = []
    epsilon = 0.05
    for N in candidates:
        errs.append(np.abs(calc_err(N)-epsilon))
    return errs

def as_4():
    return compute_bound(10000)


def as_5():
    return compute_bound(5)

def as_17():
    total_err=0
    for _ in range(5000):
        feas, labels = genData(-1, 1)
        s, theta, err = decision_stump(feas, labels)
        total_err += err
    return total_err/5000


def as_18():
    total_err = 0
    s = 0
    theta = 0
    for _ in range(5000):
        feas, labels = genData(-1, 1)
        tmp_s, tmp_theta, err = decision_stump(feas, labels)
        s += tmp_s
        theta += tmp_theta
    s = np.sign(s/5000)
    theta /= 5000

    for _ in range(5000):
        feas, labels = genData(-1, 1)
        tmp_err = eval(s, theta, feas, labels)
        total_err += tmp_err
    return total_err/5000

def as_19():
    dataset = np.loadtxt('./code/hw_2/hw2_train.dat',
                         delimiter=' ', dtype=np.float32)
    feas = dataset[:, :-1]
    labels = dataset[:, -1]
    s, theta, col, err_in = decision_stump_multi_d(feas, labels)
    return s, theta, col, err_in

def as_20():
    s, theta, col, err_in = as_19()
    dataset = np.loadtxt('./code/hw_2/hw2_test.dat',
                         delimiter=' ', dtype=np.float32)
    feas = dataset[:, col]
    labels = dataset[:, -1]
    err_out = eval(s, theta, feas, labels)
    return err_out

if __name__ == "__main__":
    print(as_5())
