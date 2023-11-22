import numpy as np
import matplotlib.pyplot as plt


def softmax(xin, w):
    d_y = len(w)
    s = np.zeros(d_y)
    sum_s = 0
    for j in range(d_y):
        s[j] = np.dot(w[j], xin.T)
        s[j] = np.exp(s[j])
        sum_s = sum_s + s[j]
    y_hat = s / sum_s
    return y_hat


def statistic(xin, yin, win):
    n_x = len(xin)
    wrong_cases = 0
    pred = []
    for j in range(n_x):
        y_pred = softmax(xin[j], win)
        class_pred = np.argmax(y_pred)
        pred.append(class_pred)
        if yin[j] != class_pred:
            wrong_cases += 1

    accuracy = 1 - wrong_cases / n_x
    return wrong_cases, accuracy, pred


def stat_progress(x_train, y_train, x_test, y_test, w_epoch, loss_epoch):
    len_hist = len(loss_epoch)
    acc_epoch = []
    train_pred = []
    test_pred = []
    train_error = -1
    test_error = -1
    for j in range(len_hist):
        if j != len_hist - 1:
            train_acc_temp = statistic(x_train, y_train, w_epoch[j])[1]
            test_acc_temp = statistic(x_test, y_test, w_epoch[j])[1]
        else:
            train_error, train_acc_temp, train_pred = statistic(x_train, y_train, w_epoch[j])
            test_error, test_acc_temp, test_pred = statistic(x_test, y_test, w_epoch[j])
        acc_epoch.append([train_acc_temp, test_acc_temp])

    acc_epoch = np.array(acc_epoch)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylim([0, loss_epoch[0] + 0.2])
    ax1.set_ylabel('loss')
    ax2 = ax1.twinx()  # this is the important function
    ax2.set_xlim([-0.5, len_hist - 0.5])
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    if len_hist > 20:
        ax1.plot(loss_epoch, c='r', label="loss")
        ax2.plot(acc_epoch[:, 0], 'g', label="acc_train")
        ax2.plot(acc_epoch[:, 1], 'b', label="acc_test")
    else:
        ax1.plot(loss_epoch, c='r', marker='.', label="loss")
        ax2.plot(acc_epoch[:, 0], 'g', marker='.', label="acc_train")
        ax2.plot(acc_epoch[:, 1], 'b', marker='.', label="acc_test")
    fig.legend(loc="right", bbox_to_anchor=(1, 0.75), bbox_transform=ax1.transAxes)
    plt.show()

    print("训练集错误个数：", train_error, "，正确率：", acc_epoch[len_hist - 1, 0])
    print("测试集错误个数：", test_error, "，正确率：", acc_epoch[len_hist - 1, 1])

    return train_pred, test_pred


def normalize(data):
    out = np.copy(data)
    min_data = np.min(data, axis=0)
    max_data = np.max(data, axis=0)
    r = max_data - min_data
    d_x = data.shape[1]
    for i in range(d_x - 1):  # 对于增广的最后一列不做归一化，防止除0
        if r[i] != 0:
            out[:, i] = (data[:, i] - min_data[i]) / r[i]
    range_norm = [min_data, max_data]
    return out, range_norm


def norm_align(data, data_range):
    data_aligned = 0 * data
    d_data = data.shape[1]
    r = data_range[1][:d_data - 1] - data_range[0][:d_data - 1]
    data_aligned[:, :d_data - 1] = (data[:, :d_data - 1] - data_range[0][:d_data - 1]) / r
    data_aligned[:, d_data - 1] = data[:, d_data - 1]
    return data_aligned


def loss(xin, yin, w_span_in):
    loss_t = 0
    n_x, d_x = np.shape(xin)
    d_y = np.shape(yin)[1]
    win = np.copy(w_span_in)
    win = win.reshape(d_y, d_x)
    for j in range(n_x):
        prob = softmax(xin[j], win)
        loss_t += np.dot(yin[j], np.log(prob).T)
    loss_t = -loss_t / n_x
    return loss_t


def deriv_loss(x, y, w):  # 对多个类别的w同时求梯度
    n_x = len(x)
    d_y = len(w)
    y_pred = np.zeros([n_x, d_y])
    for j in range(n_x):
        y_pred[j] = softmax(x[j], w)
    y_diff = y_pred - y
    d_loss = 0 * w
    for k in range(n_x):
        d_loss += np.outer(y_diff[k], x[k])
    d_loss_abs = 0
    for j in range(d_y):
        d_loss_abs_separate = d_loss[j].tolist()
        d_loss_abs += np.linalg.norm(d_loss_abs_separate)
    d_loss = d_loss / n_x
    d_loss_abs = d_loss_abs / n_x
    return d_loss, d_loss_abs


def gradient_descent(xin, yin, eta, batch, epoch):
    x = np.copy(xin)
    y = np.copy(yin)
    n_x, d_x = x.shape
    d_y = y.shape[1]

    w = np.random.normal(loc=0, scale=0.01, size=(d_y, d_x))
    hist = []
    w_span = np.copy(w)
    hist.append(w_span.reshape(1, -1)[0])
    temp = loss(x, y, w.reshape(1, -1))
    loss_epoch = [temp]
    w_epoch = [w]
    "----------"
    terminate = False
    it = 0
    for j in range(epoch):
        # print("epoch:", j)
        if terminate:
            break
        xe, ye = x, y  # 不洗牌
        rounds = int(n_x / batch)
        for k in range(rounds):
            xb = xe[k * batch:(k + 1) * batch]
            yb = ye[k * batch:(k + 1) * batch]
            d_loss, d_loss_abs = deriv_loss(xb, yb, w)
            if d_loss_abs <= 0.000000001:
                terminate = True
                break
            it += 1
            w = w - eta * d_loss
        w_epoch.append(w)
        temp = loss(x, y, w)
        loss_epoch.append(temp)
    return w, it - 1, w_epoch, loss_epoch


def adam(xin, yin, eta, max_it, alpha, beta1, beta2, epsilon):
    x = np.copy(xin)
    y = np.copy(yin)
    n_x, d_x = x.shape
    d_y = y.shape[1]

    w = np.zeros([d_y, d_x])
    hist = []
    w_span = np.copy(w)
    hist.append(w_span.reshape(1, -1)[0])
    # d_loss, d_loss_abs = deriv_loss(x, y, w)
    temp = loss(x, y, w.reshape(1, -1))
    loss_hist = [temp]

    m = np.zeros([d_y, d_x])
    v = np.zeros([d_y, d_x])
    it = 0
    d_loss_abs = 1
    while d_loss_abs > 0.000000001:
        it += 1
        if it == max_it + 1:
            break
        d_loss, d_loss_abs = deriv_loss(x, y, w)
        m = (beta1 * m + (1 - beta1) * d_loss) / (1 - beta1 ** it)
        v = (beta2 * v + (1 - beta2) * d_loss ** 2) / (1 - beta2 ** it)
        w = w - alpha * m / (v ** 0.5 + epsilon)
        # w = w - eta * d_loss.T
        w_span = np.copy(w)
        hist.append(w_span.reshape(1, -1)[0])
        d_loss, d_loss_abs = deriv_loss(x, y, w)
        # d_loss_record.append([d_loss[0][0],d_loss[0][1]])
        temp = loss(x, y, w_span)
        loss_hist.append(temp)

    return w, it - 1, loss_hist


def find_w(train_set_in, y_train_set_in, categories, eta, batch, epoch):
    n_x, d_x = np.shape(train_set_in)
    y_one_hot = np.zeros([n_x, categories])
    for j in range(categories):
        for n in range(n_x):
            if y_train_set_in[n][0] == j:
                y_one_hot[n][j] = 1
    w, iteration, w_epoch, loss_epoch = gradient_descent(train_set_in, y_one_hot, eta, batch, epoch)

    return w, w_epoch, loss_epoch
