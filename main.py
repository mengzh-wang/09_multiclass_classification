import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os.path import join
from read_mnist_dataset import MnistDataloader, show_images
import ovo_classify as ovo
import softmax


def iris_classification():
    iris_file = open("iris.csv", 'r')
    iris_table = pd.read_csv("iris.csv")
    iris = np.loadtxt(iris_file, delimiter=',', skiprows=1, usecols=[1, 2, 3, 4])
    augment = np.ones([150, 1])
    iris = np.hstack([iris, augment])  # 数据增广
    nx, dx = np.shape(iris)
    y = np.zeros([nx, 1])
    for i in range(nx):
        if iris_table["Species"][i] == "setosa":
            y[i][0] = 0
        elif iris_table["Species"][i] == "versicolor":
            y[i][0] = 1
        elif iris_table["Species"][i] == "virginica":
            y[i][0] = 2

    train_set = np.zeros([90, dx])
    y_train_set = np.zeros([90, 1])
    test_set = np.zeros([60, dx])
    y_test_set = np.zeros([60, 1])
    for i in range(3):
        temp = np.random.permutation(iris[50 * i:50 * (i + 1)])
        train_set[30 * i:30 * (i + 1)] = temp[0:30]
        y_train_set[30 * i:30 * (i + 1)] = np.ones([30, 1]) * i
        test_set[20 * i:20 * (i + 1)] = temp[30:50]
        y_test_set[20 * i:20 * (i + 1)] = np.ones([20, 1]) * i

    """-----------------算法性能测试-----------------"""
    print("一对一（OVO）算法：")
    w_ovo = ovo.find_w(train_set)
    wrong_cases_train_ovo, acc_train_ovo = ovo.statistic(train_set, y_train_set, w_ovo)
    wrong_cases_test_ovo, acc_rate_test_ovo = ovo.statistic(test_set, y_test_set, w_ovo)
    print("训练集错误个数：", wrong_cases_train_ovo, "，正确率：", acc_train_ovo)
    print("测试集错误个数：", wrong_cases_test_ovo, "，正确率：", acc_rate_test_ovo)

    print("softmax算法：")
    train_set_norm, range_norm = softmax.normalize(train_set)
    w_softmax, w_epoch_softmax, loss_epoch_softmax = softmax.find_w(train_set_norm, y_train_set, categories=3, eta=0.6,
                                                                    batch=90, epoch=600)
    test_set_aligned = softmax.norm_align(test_set, range_norm)
    softmax.stat_progress(train_set_norm, y_train_set, test_set_aligned, y_test_set, w_epoch_softmax,
                          loss_epoch_softmax)
    '''wrong_cases_train_softmax, acc_train_softmax = softmax.statistic(train_set_norm, y_train_set, w_softmax)[0:2]
    wrong_cases_test_softmax, acc_test_softmax = softmax.statistic(test_set_aligned, y_test_set, w_softmax)[0:2]
    print("训练集错误个数：", wrong_cases_train_softmax, "，正确率：", acc_train_softmax)
    print("测试集错误个数：", wrong_cases_test_softmax, "，正确率：", acc_test_softmax)
    plt.figure()
    plt.plot(loss_epoch_softmax)
    plt.show()'''
    print('Iris Test Finished.')


# iris_classification()


def mnist_classification():
    input_path = '.\\mnist_dataset'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                       test_labels_filepath)
    (img_train, label_train), (img_test, label_test) = mnist_dataloader.load_data()

    x_train = np.array(img_train)
    x_train = x_train.reshape(60000, 784)
    x_train = x_train / 255
    y_train = np.array(label_train)
    y_train = y_train.reshape(60000, 1)
    x_test = np.array(img_test)
    x_test = x_test.reshape(10000, 784)
    x_test = x_test / 255
    y_test = np.array(label_test)
    y_test = y_test.reshape(10000, 1)

    w_softmax, w_epoch_softmax, loss_epoch_softmax = softmax.find_w(x_train, y_train, categories=10, eta=0.03,
                                                                    batch=256, epoch=10)
    train_pred, test_pred = softmax.stat_progress(x_train, y_train, x_test, y_test, w_epoch_softmax, loss_epoch_softmax)

    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = np.random.randint(1, 10000)
        images_2_show.append(img_test[r])
        str_2_show = 'No.' + str(r) + '=' + str(label_test[r]) + ', y_s=' + str(test_pred[r])
        titles_2_show.append(str_2_show)

    show_images(images_2_show, titles_2_show)

    print('MNIST Test Finished.')


mnist_classification()
