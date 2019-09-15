import numpy as np 

from PLA import pla, pocket, trans_sample, test

def as_15():
    train_set = np.loadtxt('./code/hw_1/hw1_15_train.dat',
                           delimiter=' ', dtype=np.float32)
    w, update_time = pla(train_set)
    print('w:', w)
    print('update_time:', update_time)


def as_16():
    train_set = np.loadtxt('./code/hw_1/hw1_15_train.dat',
                           delimiter=' ', dtype=np.float32)
    total_update = 0
    for _ in range(2000):
        w, update_time = pla(train_set, shuffle=True)
        total_update += update_time
    print('average_update_time:', total_update/2000)


def as_17():
    train_set = np.loadtxt('./code/hw_1/hw1_15_train.dat',
                           delimiter=' ', dtype=np.float32)
    total_update = 0
    for _ in range(2000):
        w, update_time = pla(train_set, lr=0.5, shuffle=True)
        total_update += update_time
    print('average_update_time:', total_update/2000)


def as_18():
    train_set = np.loadtxt('./code/hw_1/hw1_18_train.dat',
                           delimiter=' ', dtype=np.float32)
    test_set = np.loadtxt('./code/hw_1/hw1_18_test.dat',
                           delimiter=' ', dtype=np.float32)
    # do test
    test_feas, test_labels = trans_sample(test_set)
    total_acc = 0
    for i in range(2000):
        w, _ = pocket(train_set, shuffle=True, iter_num=50)
        test_acc = test(test_feas, test_labels, w)
        total_acc += test_acc
    print('average_acc:', total_acc/2000)


def as_19():
    train_set = np.loadtxt('./code/hw_1/hw1_18_train.dat',
                           delimiter=' ', dtype=np.float32)
    test_set = np.loadtxt('./code/hw_1/hw1_18_test.dat',
                          delimiter=' ', dtype=np.float32)
    # do test
    test_feas, test_labels = trans_sample(test_set)
    total_acc = 0
    for i in range(2000):
        w, _ = pla(train_set, shuffle=True, iter_num=50)
        test_acc = test(test_feas, test_labels, w)
        total_acc += test_acc
    print('average_acc:', total_acc/2000)


def as_20():
    train_set = np.loadtxt('./code/hw_1/hw1_18_train.dat',
                           delimiter=' ', dtype=np.float32)
    test_set = np.loadtxt('./code/hw_1/hw1_18_test.dat',
                          delimiter=' ', dtype=np.float32)
    # do test
    test_feas, test_labels = trans_sample(test_set)
    total_acc = 0
    for i in range(2000):
        w, _ = pocket(train_set, shuffle=True, iter_num=100)
        test_acc = test(test_feas, test_labels, w)
        total_acc += test_acc
    print('average_acc:', total_acc/2000)

if __name__ == "__main__":
    as_20()
