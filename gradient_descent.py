import numpy as np

"""----------------数据归一化----------------"""


def normalization(data):
    out = np.copy(data)
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)
    R = max - min
    d_x = data.shape[1]
    if d_x == 1:
        out[:, 0] = (data[:, 0] - min[0]) / R[0]
    else:
        for i in range(data.shape[1] - 1):
            out[:, i] = (data[:, i] - min[i]) / R[i]

    return out, min, max


"""----------------w逆归一化----------------"""


def norm_inverse(w, data_range):
    min1 = data_range[0]
    max1 = data_range[1]
    min2 = data_range[2]
    max2 = data_range[3]
    if len(w[0]) == 3:
        d = w[0][0]
        e = w[0][1]
        f = w[0][2]
        a = d / (max1[0] - min1[0])
        b = e / (max1[1] - min1[1])
        c = -d * min1[0] / (max1[0] - min1[0]) - e * min1[1] / (max1[1] - min1[1]) + f / 2
        w[0][0] = a
        w[0][1] = b
        w[0][2] = c
    elif len(w[0]) == 2:
        d = w[0][0]
        e = w[0][1]
        a = d * (max2[0] - min2[0]) / (max1[0] - min1[0])
        b = -d * min1[0] * (max2[0] - min2[0]) / (max1[0] - min1[0]) + min2[0] + e * (max2[0] - min2[0])
        w[0][0] = a
        w[0][1] = b

    return w


"""----------------洗牌----------------"""


def shuffle(xin, yin, batch_size):
    x_in = np.copy(xin)
    y_in = np.copy(yin)
    x_out = []
    y_out = []
    for i in range(len(x_in) - 1, len(x_in) - 1 - batch_size, -1):
        p = np.random.random_integers(0, i)
        x_in[i], x_in[p] = x_in[p], x_in[i]
        y_in[i], y_in[p] = y_in[p], y_in[i]
        x_out = x_in[len(x_in) - 1 - batch_size:len(x_in) - 1]
        y_out = y_in[len(x_in) - 1 - batch_size:len(x_in) - 1]
    return x_out, y_out


"""----------------计算损失函数梯度----------------"""


def deriv_loss(x, y, w):
    n_x = len(x)
    temp = np.dot(x, w) - y
    temp = np.dot(temp.T, x)
    d_loss = 2 / n_x * temp
    d_loss_abs = d_loss.tolist()
    d_loss_abs = np.linalg.norm(d_loss_abs)
    return d_loss, d_loss_abs


"""----------------不同的梯度下降法----------------"""


def preprocess(xin, yin):   # 预处理
    x = np.copy(xin)
    y = np.copy(yin)
    x, min_x, max_x = normalization(x)
    y, min_y, max_y = normalization(y)
    loss_t = []
    hist = []
    # d_loss_record = []
    [n_x, d_x] = x.shape
    w = np.zeros((1, d_x))
    w = w.T
    hist.append([w[0][0], w[1][0]])
    # d_loss, d_loss_abs = deriv_loss(x, y, w)
    # d_loss_record.append([d_loss[0][0], d_loss[0][1]])
    loss = np.sum((np.dot(x, w) - y) ** 2) / n_x
    loss_t.append([0, loss])
    range = [min_x, max_x, min_y, max_y]

    return x, y, w, n_x, hist, loss_t, range


"""----------------梯度下降----------------"""


def gradient_descent(xin, yin, eta, max_it):
    x, y, w, n_x, hist, loss_t, data_range = preprocess(xin, yin)
    d_loss, d_loss_abs = deriv_loss(x, y, w)

    it = 0
    while d_loss_abs > 0.000000001:
        it += 1
        if it == max_it + 1:
            break
        w = w - eta * d_loss.T
        hist.append([w[0][0], w[1][0]])
        d_loss, d_loss_abs = deriv_loss(x, y, w)
        # d_loss_record.append([d_loss[0][0],d_loss[0][1]])
        loss = np.sum((np.dot(x, w) - y) ** 2) / n_x
        loss_t.append([it, loss])
    w = w.T
    w = w.tolist()
    w = norm_inverse(w, data_range)

    return w, it - 1, loss_t, hist


def gradient_descent_epoch(xin, yin, eta, batch, epoch):
    x, y, w, n_x, hist, loss_t, data_range = preprocess(xin, yin)
    d_loss, d_loss_abs = deriv_loss(x, y, w)
    terminate = False
    it = 0
    loss_min = loss_t[0]
    for j in range(epoch):
        if terminate:
            break
        xe, ye = x, y  # 不进行洗牌
        # xe, ye = shuffle(x, y)        # 进行洗牌
        for k in range(int(n_x / batch)):
            xb = xe[k * batch:(k + 1) * batch]
            yb = ye[k * batch:(k + 1) * batch]
            d_loss, d_loss_abs = deriv_loss(xb, yb, w)
            loss = np.sum((np.dot(x, w) - y) ** 2) / n_x
            loss_t.append([it, loss])
            if d_loss_abs <= 0.000000001:
                terminate = True
                break
            it += 1
            w = w - eta * d_loss.T
            hist.append([w[0], w[1], w[2]])
    w = w.T
    w = w.tolist()
    w = norm_inverse(w, data_range)

    return w, it - 1, loss_t, hist


"""----------------随机梯度下降----------------"""


def sgd(xin, yin, eta, max_it, batch_size):
    x, y, w, n_x, hist, loss_t, data_range = preprocess(xin, yin)
    xb, yb = shuffle(x, y, batch_size)
    d_loss, d_loss_abs = deriv_loss(xb, yb, w)

    it = 0
    while d_loss_abs > 0.000000001:
        it += 1
        if it == max_it + 1:
            break
        w = w - eta * d_loss.T
        hist.append([w[0][0], w[1][0]])
        xb, yb = shuffle(x, y, batch_size)
        d_loss, d_loss_abs = deriv_loss(xb, yb, w)

        loss = np.sum((np.dot(x, w) - y) ** 2) / len(x)
        loss_t.append([it, loss])
    w = w.T
    w = w.tolist()
    w = norm_inverse(w, data_range)
    return w, it - 1, loss_t, hist


"""----------------Adagrad----------------"""


def adagrad(xin, yin, eta, max_it, epsilon):
    x, y, w, n_x, hist, loss_t, data_range = preprocess(xin, yin)
    d_loss, d_loss_abs = deriv_loss(x, y, w)
    d_loss_sum = d_loss ** 2
    it = 0
    while d_loss_abs > 0.000000001:
        it += 1
        if it == max_it + 1:
            break
        sigma = (1 / it * d_loss_sum) ** 0.5 + epsilon
        w = w - eta / (it * sigma.T) * d_loss.T
        hist.append([w[0][0], w[1][0]])
        d_loss, d_loss_abs = deriv_loss(x, y, w)
        d_loss_sum = d_loss_sum + d_loss ** 2
        loss = np.sum((np.dot(x, w) - y) ** 2) / n_x
        loss_t.append([it, loss])
    w = w.T
    w = w.tolist()

    w = norm_inverse(w, data_range)

    return w, it - 1, loss_t, hist


"""----------------rms_drop----------------"""


def rms_drop(xin, yin, eta, max_it, epsilon, alpha):
    x, y, w, n_x, hist, loss_t, data_range = preprocess(xin, yin)
    d_loss, d_loss_abs = deriv_loss(x, y, w)
    d_loss_sum = d_loss ** 2
    it = 0
    sigma = []
    while d_loss_abs > 0.000000001:
        it += 1
        if it == max_it + 1:
            break
        if it == 1:
            sigma = abs(d_loss)
        else:
            sigma = (alpha * sigma ** 2 + (1 - alpha) * d_loss ** 2) ** 0.5 + epsilon
        w = w - eta / sigma.T * d_loss.T
        hist.append([w[0][0], w[1][0]])
        d_loss, d_loss_abs = deriv_loss(x, y, w)
        d_loss_sum = d_loss_sum + d_loss ** 2
        loss = np.sum((np.dot(x, w) - y) ** 2) / n_x
        loss_t.append([it, loss])
    w = w.T
    w = w.tolist()

    w = norm_inverse(w, data_range)

    return w, it - 1, loss_t, hist


"""----------------Momentum----------------"""


def momentum(xin, yin, eta, max_it, lamda):
    x, y, w, n_x, hist, loss_t, data_range = preprocess(xin, yin)
    d_loss, d_loss_abs = deriv_loss(x, y, w)
    m = 0
    it = 0
    while d_loss_abs > 0.000000001:
        it += 1
        if it == max_it + 1:
            break
        m = lamda * m - eta * d_loss.T
        w = w + m
        hist.append([w[0][0], w[1][0]])
        d_loss, d_loss_abs = deriv_loss(x, y, w)
        loss = np.sum((np.dot(x, w) - y) ** 2) / n_x
        loss_t.append([it, loss])
    w = w.T
    w = w.tolist()
    w = norm_inverse(w, data_range)

    return w, it - 1, loss_t, hist


"""----------------Adam----------------"""


def adam(xin, yin, eta, max_it, alpha, beta1, beta2, epsilon):
    x, y, w, n_x, hist, loss_t, data_range = preprocess(xin, yin)
    d_loss, d_loss_abs = deriv_loss(x, y, w)
    m = v = 0
    it = 0
    while d_loss_abs > 0.000000001:
        it += 1
        if it == max_it + 1:
            break
        m = (beta1 * m - (1 - beta1) * d_loss.T) / (1 - beta1 ** it)
        v = (beta2 * v + (1 - beta2) * d_loss.T ** 2 )/ (1 - beta2 ** it)
        w = w - alpha * m / (v ** 0.5 + epsilon)
        w = w - eta * d_loss.T
        hist.append([w[0][0], w[1][0]])
        d_loss, d_loss_abs = deriv_loss(x, y, w)
        # d_loss_record.append([d_loss[0][0],d_loss[0][1]])
        loss = np.sum((np.dot(x, w) - y) ** 2) / n_x
        loss_t.append([it, loss])
    w = w.T
    w = w.tolist()
    w = norm_inverse(w, data_range)

    return w, it - 1, loss_t, hist
