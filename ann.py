#!/usr/bin/python
# -*- coding: utf-8 -*-

## 这一份代码是以前写的，by 胡

import numpy as np
import struct
import matplotlib.pyplot as plt

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False


def get_xp(x):
    if GPU_AVAILABLE and cp is not None and isinstance(x, cp.ndarray):
        return cp
    return np


def to_gpu(x):
    if GPU_AVAILABLE and cp is not None and not isinstance(x, cp.ndarray):
        return cp.asarray(x)
    return x


def to_cpu(x):
    if GPU_AVAILABLE and cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x


def col_vec(x, xp):
    x = xp.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


# define activation function
def tanh(x):
    xp = get_xp(x)
    return xp.tanh(x)

def sigmoid(x):
    xp = get_xp(x)
    return 1.0 / (1 + xp.exp(-1.0 * x))

def RELU(x):
    xp = get_xp(x)
    return xp.maximum(0, x)

def RELU3(x):
    return RELU(x) * RELU(x) * RELU(x)


# define derivatives
def tanh_deriv(x):
    y = tanh(x)
    return 1.0 - y * y

def sigmoid_deriv(x):
    s = sigmoid(x)
    return (1.0 - s) * s

def RELU_deriv(x):
    xp = get_xp(x)
    return (x >= 0).astype(xp.float32)

def RELU3_deriv(x):
    return 3.0 * RELU(x) * RELU(x)


class NeuralNetwork:
    def __init__(self, layers, activation, opt_alg, use_gpu=True):
        """
        layers: layers of NNs
        activation: activation function name
        opt_alg: optimization algorithm name
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np

        # 保存名称，后面画图直接用
        self.activation_name = activation
        self.opt_alg_name = opt_alg

        # 记录训练过程
        self.precision_history = []
        self.perf_history = []

        if activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv
        elif activation == 'RELU':
            self.activation = RELU
            self.activation_deriv = RELU_deriv
        elif activation == 'RELU3':
            self.activation = RELU3
            self.activation_deriv = RELU3_deriv
        else:
            raise ValueError("Unsupported activation")

        if opt_alg == 'GD':
            self.opt = self.GD
        elif opt_alg == 'SGD':
            self.opt = self.SGD
        elif opt_alg == 'ADAM':
            self.opt = self.ADAM
            self.mw = []
            self.mtheta = []
            self.vw = []
            self.vtheta = []
            for i in range(len(layers) - 1):
                self.mw.append(self.xp.zeros((layers[i + 1], layers[i]), dtype=self.xp.float32))
                self.mtheta.append(self.xp.zeros((layers[i + 1], 1), dtype=self.xp.float32))
                self.vw.append(self.xp.zeros((layers[i + 1], layers[i]), dtype=self.xp.float32))
                self.vtheta.append(self.xp.zeros((layers[i + 1], 1), dtype=self.xp.float32))
        else:
            raise ValueError("Unsupported optimization algorithm")

        self.weights = []
        self.thetas = []
        for i in range(len(layers) - 1):
            w_cpu = 2 * np.random.rand(layers[i + 1], layers[i]).astype(np.float32) - 1
            t_cpu = 2 * np.random.rand(layers[i + 1], 1).astype(np.float32) - 1

            if self.use_gpu:
                self.weights.append(cp.asarray(w_cpu))
                self.thetas.append(cp.asarray(t_cpu))
            else:
                self.weights.append(w_cpu)
                self.thetas.append(t_cpu)

        self.layers = layers

    def propagation(self, x, k):
        xp = self.xp
        temp = xp.asarray(x, dtype=xp.float32)
        if temp.ndim == 1:
            temp = temp.reshape(-1, 1)

        for i in range(len(self.weights)):
            temp = self.activation(xp.dot(self.weights[i], temp) + self.thetas[i])
        z = k * temp
        return z

    def backpropagation(self, x, error):
        xp = self.xp
        n_w = len(self.weights)
        z = []
        K = []
        dweights = []
        dthetas = []
        delta = []

        x = xp.asarray(x, dtype=xp.float32)
        error = xp.asarray(error, dtype=xp.float32)

        for i in range(n_w):
            if i == 0:
                z.append(xp.dot(self.weights[i], x) + self.thetas[i])
                K.append(x)
                delta.append(x)
            else:
                act_prev = self.activation(z[i - 1])
                z.append(xp.dot(self.weights[i], act_prev) + self.thetas[i])
                K.append(act_prev)
                delta.append(act_prev)

        for i in range(n_w - 1, -1, -1):
            if i == n_w - 1:
                delta[i] = error * self.activation_deriv(z[i])
            else:
                delta[i] = xp.dot(self.weights[i + 1].T, delta[i + 1]) * self.activation_deriv(z[i])

        for i in range(n_w):
            dweights.append(xp.dot(delta[i], K[i].T))
            dthetas.append(delta[i])

        return dweights, dthetas

    def GD(self, X, Y, k, learning_rate, epochs):
        xp = self.xp
        perf = 0.0
        ddweights = []
        ddthetas = []
        layers = self.layers

        for i in range(len(layers) - 1):
            ddweights.append(xp.zeros((layers[i + 1], layers[i]), dtype=xp.float32))
            ddthetas.append(xp.zeros((layers[i + 1], 1), dtype=xp.float32))

        sample_num = X.shape[1]

        for j in range(sample_num):
            input = X[:, j:j+1]
            output = self.propagation(input, k)
            y = Y[:, j:j+1]
            error = y - output
            perf += float(to_cpu(0.5 * xp.sum(error * error)))

            dweights, dthetas = self.backpropagation(input, error)
            for i in range(len(self.weights)):
                ddweights[i] += k * (1.0 / sample_num) * learning_rate * dweights[i]
                ddthetas[i] += k * (1.0 / sample_num) * learning_rate * dthetas[i]

        for i in range(len(self.weights)):
            self.weights[i] += ddweights[i]
            self.thetas[i] += ddthetas[i]

        return perf

    def SGD(self, X, Y, k, learning_rate, epochs):
        xp = self.xp
        perf = 0.0
        ddweights = []
        ddthetas = []
        layers = self.layers

        for i in range(len(layers) - 1):
            ddweights.append(xp.zeros((layers[i + 1], layers[i]), dtype=xp.float32))
            ddthetas.append(xp.zeros((layers[i + 1], 1), dtype=xp.float32))

        rand_X = xp.arange(X.shape[1])
        xp.random.shuffle(rand_X)

        n = max(1, int(X.shape[1] / 100))
        for j in range(n):
            idx = int(to_cpu(rand_X[j]))
            input = X[:, idx:idx+1]
            output = self.propagation(input, k)
            y = Y[:, idx:idx+1]
            error = y - output
            perf += float(to_cpu(0.5 * xp.sum(error * error)))

            dweights, dthetas = self.backpropagation(input, error)
            for i in range(len(self.weights)):
                ddweights[i] += k * (1.0 / n) * learning_rate * dweights[i]
                ddthetas[i] += k * (1.0 / n) * learning_rate * dthetas[i]

        for i in range(len(self.weights)):
            self.weights[i] += ddweights[i]
            self.thetas[i] += ddthetas[i]

        return perf

    def ADAM(self, X, Y, k, learning_rate, epochs):
        xp = self.xp
        perf = 0.0
        layers = self.layers

        beta2 = 0.999
        beta1 = 0.9
        epsilon = 1e-8

        ddweights = []
        ddthetas = []
        for i in range(len(layers) - 1):
            ddweights.append(xp.zeros((layers[i + 1], layers[i]), dtype=xp.float32))
            ddthetas.append(xp.zeros((layers[i + 1], 1), dtype=xp.float32))

        rand_X = xp.arange(X.shape[1])
        xp.random.shuffle(rand_X)
        n = max(1, int(X.shape[1] / 10))

        for j in range(n):
            idx = int(to_cpu(rand_X[j]))
            input = X[:, idx:idx+1]
            output = self.propagation(input, k)
            y = Y[:, idx:idx+1]
            error = y - output
            perf += float(to_cpu(0.5 * xp.sum(error * error)))

            dweights, dthetas = self.backpropagation(input, error)
            for i in range(len(self.weights)):
                ddweights[i] += k * (1.0 / n) * dweights[i]
                ddthetas[i] += k * (1.0 / n) * dthetas[i]

        for j in range(len(self.weights)):
            self.mw[j] = beta1 * self.mw[j] + (1 - beta1) * ddweights[j]
            self.vw[j] = beta2 * self.vw[j] + (1 - beta2) * (ddweights[j] * ddweights[j])
            self.mtheta[j] = beta1 * self.mtheta[j] + (1 - beta1) * ddthetas[j]
            self.vtheta[j] = beta2 * self.vtheta[j] + (1 - beta2) * (ddthetas[j] * ddthetas[j])

            mwHat = self.mw[j] / (1 - beta1 ** epochs)
            vwHat = self.vw[j] / (1 - beta2 ** epochs)
            mtHat = self.mtheta[j] / (1 - beta1 ** epochs)
            vtHat = self.vtheta[j] / (1 - beta2 ** epochs)

            self.weights[j] += learning_rate * mwHat / (xp.sqrt(vwHat) + epsilon)
            self.thetas[j] += learning_rate * mtHat / (xp.sqrt(vtHat) + epsilon)

        return perf

    def train(self, X, Y, k, learning_rate, epochs):
        xp = self.xp
        X = xp.asarray(X, dtype=xp.float32)
        Y = xp.asarray(Y, dtype=xp.float32)

        self.precision_history = []
        self.perf_history = []

        for i in range(epochs):
            perf = self.opt(X, Y, k, learning_rate, i + 1)
            predict = self.propagation(X, 1)
            true = int(to_cpu(xp.sum(xp.argmax(Y, axis=0) == xp.argmax(predict, axis=0))))
            precision = 1.0 * true / Y.shape[1]

            self.perf_history.append(perf)
            self.precision_history.append(precision)

            print('perf:', perf, 'epochs:', i + 1, 'predict_true:', true, 'precision:', precision)

    def plot_precision(self, save_path=None):
        epochs = np.arange(1, len(self.precision_history) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, self.precision_history)
        plt.xlabel("epoch")
        plt.ylabel("Precision")
        plt.title(f"Precision after {len(self.precision_history)} epoch,with {self.activation_name} and {self.opt_alg_name}")

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
if __name__ == "__main__":
    num = 500
    test_num = 100
    data_root = './data/FashionMNIST/raw'

    filename = data_root + '/train-images-idx3-ubyte'
    binfile = open(filename , 'rb')
    buf = binfile.read()

    index = 0
    X = np.zeros((784, num), dtype=np.float32)

    magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)
    index += struct.calcsize('>IIII')
    for i in range(num):
        X[:, i] = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
    X = np.array(X, dtype=np.float32) / 255.0

    filename_test = data_root + '/t10k-images-idx3-ubyte'
    binfile_test = open(filename_test, 'rb')
    buf_test = binfile_test.read()

    index_test = 0
    testX = np.zeros((784, test_num), dtype=np.float32)
    magic, numImages , numRows , numColumns = struct.unpack_from('>IIII', buf_test, index_test)
    index_test += struct.calcsize('>IIII')
    for i in range(test_num):
        testX[:, i] = struct.unpack_from('>784B', buf_test, index_test)
        index_test += struct.calcsize('>784B')
    testX = np.array(testX, dtype=np.float32) / 255.0

    filename_tar = data_root + '/train-labels-idx1-ubyte'
    binfile_tar = open(filename_tar, 'rb')
    buf_tar = binfile_tar.read()

    index_tar = 8
    Y = np.zeros((10, num), dtype=np.float32)
    for i in range(num):
        k = struct.unpack_from('>1B', buf_tar, index_tar)[0]
        Y[k][i] = 1
        index_tar += struct.calcsize('>1B')
    Y = np.array(Y, dtype=np.float32)

    filename_test_tar = data_root + '/t10k-labels-idx1-ubyte'
    binfile_test_tar = open(filename_test_tar, 'rb')
    buf_test_tar = binfile_test_tar.read()

    index_test_tar = 8
    testY = np.zeros((10, test_num), dtype=np.float32)
    for i in range(test_num):
        k = struct.unpack_from('>1B', buf_test_tar, index_test_tar)[0]
        testY[k][i] = 1
        index_test_tar += struct.calcsize('>1B')
    testY = np.array(testY, dtype=np.float32)

    layers = np.array([X.shape[0], 128, 64, Y.shape[0]])
    Net = NeuralNetwork(layers, 'sigmoid', 'GD', use_gpu=True)

    Net.train(X, Y, 1, 0.1,10000)

    Net.plot_precision("sigmoid_GD.png")

    testX_dev = to_gpu(testX) if GPU_AVAILABLE else testX
    testY_dev = to_gpu(testY) if GPU_AVAILABLE else testY

    output = Net.propagation(testX_dev, 1)
    xp = cp if GPU_AVAILABLE else np
    true = int(to_cpu(xp.sum(xp.argmax(testY_dev, axis=0) == xp.argmax(output, axis=0))))

    print("test correct:", true)
    print("test precision:", true / testX.shape[1])