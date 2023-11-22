import numpy as np


def pla(xin, yin, max_it):
    [n_x, d_x] = xin.shape
    it = 0
    w_pla = np.zeros(d_x)

    while it < max_it:
        misclassified = False
        for k in range(n_x):
            if yin[k] * np.dot(w_pla, xin[k]) <= 0:
                misclassified = True
                w_pla += 0.001 * yin[k] * xin[k]
        if not misclassified:
            break
        it += 1

    return w_pla


def statistic(xin, yin, win):
    n_x = len(xin)
    wrong_cases = 0
    for i in range(n_x):
        w0 = np.sign(np.dot(win[0], xin[i].T))
        w1 = np.sign(np.dot(win[1], xin[i].T))
        w2 = np.sign(np.dot(win[2], xin[i].T))
        if w0 == 1 and w1 == 1:
            y_pred = 0
        elif w0 == -1 and w2 == 1:
            y_pred = 1
        elif w1 == -1 and w2 == -1:
            y_pred = 2
        else:
            y_pred = -1
        if yin[i] != y_pred:
            wrong_cases += 1

    accuracy = 1 - wrong_cases / n_x
    return wrong_cases, accuracy


def find_w(train_set_in):
    d_x = np.shape(train_set_in)[1]
    w = np.zeros([3, d_x])
    for i in range(2):
        for j in range(i + 1, 3):
            x_train = np.zeros([60, d_x])
            x_train[0:30] = train_set_in[30 * i:30 * (i + 1)]
            x_train[30:60] = train_set_in[30 * j:30 * (j + 1)]
            y_train = np.zeros([60, 1])
            y_train[0:30] = np.ones([30, 1])
            y_train[30:60] = -1 * np.ones([30, 1])
            w[i + j - 1] = pla(x_train, y_train, 1000)
            """
            w[0]: 正为0类，负为1类;
            w[1]: 正为0类，负为2类;
            w[2]: 正为1类，负为2类;
            """
    return w
