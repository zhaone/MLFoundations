import numpy as np
import copy

def trans_sample(dataset, shuffle=False):
    # process dataset
    if shuffle:
        np.random.shuffle(dataset)
    # get feas and labels
    feas = dataset[:, :-1]
    bias = np.ones((len(feas), 1))
    feas = np.hstack((bias, feas))
    labels = dataset[:, -1].reshape((-1, 1))
    return feas, labels

# do an  evaluation for dataset
def test(feas, labels, w):
    test_out = np.sign(np.dot(feas, w.T))
    test_out[test_out == 0] = -1
    mask = test_out == np.squeeze(labels)
    right_num = np.sum(mask)
    acc = right_num/feas.shape[0]
    return acc

def pla(train_set, lr=1, shuffle=False, iter_num=np.inf):
    feas, labels = trans_sample(train_set, shuffle=shuffle)
    # begin to train
    update_time = 0
    w = np.zeros(feas.shape[1])
        
    while update_time <= iter_num:
        conti = False
        for fea, label in zip(feas, labels):
            out = np.sign(np.dot(fea,w.T))
            out = -1 if out == 0 else out
            if out != label:
                conti = True
                w = w + lr * fea * label
                update_time += 1
        if not conti:
            return w, update_time
    return w, update_time


def pocket(train_set, lr=1, shuffle=False, iter_num=np.inf):
    feas, labels = trans_sample(train_set, shuffle=shuffle)
    # begin to train
    update_time = 0
    w = np.zeros(feas.shape[1])
    best_w = None
    best_acc = 0
    best_update_id = None # this is for debug
    # train pocket
    while update_time <= iter_num:
        conti = False
        for fea, label in zip(feas, labels):
            out = np.sign(np.dot(fea, w.T))
            out = -1 if out == 0 else out
            if out != label:
                conti = True
                # update
                w += lr * fea * label
                tmp_acc = test(feas, labels, w)
                update_time += 1
                # if new w works better, do an update
                if tmp_acc > best_acc:
                    # note! deep copy, not just a reference
                    best_w = copy.copy(w)
                    best_acc = tmp_acc
                    best_update_id = update_time
                if update_time == iter_num:
                    break
        if not conti:
            break
    return best_w, best_acc
