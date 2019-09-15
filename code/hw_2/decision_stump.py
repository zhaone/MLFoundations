import numpy as np
from copy import copy as cp

def genData(s, e, size=20, noise_prob=0.2):
    assert e > s, "internal end points err, end point should >= start point"
    gap = (e-s)/size
    feas = np.arange(s+gap/2, e+gap/2, gap)
    labels = np.sign(feas)

    neg_labels = -labels
    mask = np.random.ranf(size) < 0.2
    labels[mask] = neg_labels[mask]

    return feas, labels


def eval(s, theta, feas, labels):
    pred = s*np.sign(feas-theta)
    pred[pred == 0] = -1
    err = np.sum(pred != labels)/feas.shape[0]
    return err


def decision_stump(feas, labels):
    best_s = None
    best_theta = None
    min_err = np.inf
    # sort
    arg = np.argsort(feas)
    feas = feas[arg]
    labels = labels[arg]
    # find min err dic
    for s in [-1, 1]:
        for i in range(feas.shape[0]):
            if i == 0:
                theta = feas[i] - 0.1
            elif i == feas.shape[0]-1:
                theta = feas[-1] + 0.1
            else:
                theta = (feas[i] + feas[i+1])/2
            tmp_err = eval(s, theta, feas, labels)
            if tmp_err < min_err:
                best_s = cp(s)
                best_theta = cp(theta)
                min_err = tmp_err
    return best_s, best_theta, min_err


def decision_stump_multi_d(feas, labels):
    best_s, best_theta, best_col = (None, None, None)
    min_err = np.inf
    for col in range(feas.shape[1]):
        fea = feas[:, col]
        s, theta, err = decision_stump(fea, labels)
        if err < min_err:
            best_s, best_theta, best_col, min_err = (
                cp(s), cp(theta), cp(col), cp(err))
    return best_s, best_theta, best_col, min_err
