#-----------
# 数据加载
#-----------
import numpy as np
from keras.utils import to_categorical

def load_data():
    """
    训练集：55000
    测试集：10000
    验证集：5000
    path: mnist.npz
    """
    f = np.load('../data/mnist.npz')
    train_x, train_y = f["x_train"], f["y_train"]
    test_x, test_y = f["x_test"], f["y_test"]
    # >>> 分一点数据到验证集
    np.random.seed(12305)  # 设置同一个随机种子
    state = np.random.get_state()
    np.random.shuffle(train_x)
    np.random.set_state(state)
    np.random.shuffle(train_y)
    val_x, val_y = train_x[:5000], train_y[:5000]
    train_x, train_y = train_x[5000:], train_y[5000:]
    # <<<
    train_x, val_x, test_x = (
        train_x.reshape((-1, 28, 28, 1)).astype("float32") / 255,
        val_x.reshape((-1, 28, 28, 1)).astype("float32") / 255,
        test_x.reshape((-1, 28, 28, 1)).astype("float32") / 255,
    )
    train_y, val_y, test_y = to_categorical(train_y), to_categorical(val_y), to_categorical(test_y)
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


#----------------------------
# callback logging function
#----------------------------
from keras.callbacks import Callback
import numpy as np

class logger(Callback):
    def __init__(self, output_path):
        self.output_path = output_path

    def on_train_begin(self, logs={}):
        self.log = {}

    def on_batch_end(self, batch, logs={}):
        for i in logs.keys():
            if i == 'size' or i == 'batch':
                continue

            if i not in self.log:
                self.log[i] = []

        for i in logs.keys():
            if i == 'size' or i == 'batch':
                continue

            self.log[i].append(logs.get(i))

    def on_epoch_end(self, epoch, logs={}):
        for i in logs.keys():
            if i not in self.log:
                self.log[i] = []

        for i in logs.keys():
            self.log[i].append(logs.get(i))

    def on_train_end(self, logs={}):
        np.save(self.output_path, self.log)


#-----------------
# model function
#-----------------
import sys

sys.path.append("../..")
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Softmax
from keras.layers import Flatten
from keras.models import Model
from src.conv import Conv2D as RQConv2D

from src.regularizer import rq_reg
from keras.regularizers import l1
from keras.regularizers import l2


def rq_model(reg_rate):
    KERNEL_REG = lambda n: rq_reg(reg_rate, n)

    inputs = Input((28, 28, 1))
    net = Conv2D(4, 3, padding="same")(inputs)
    net = RQConv2D(8, 3, padding="same", activation="relu", kernel_regularizer=KERNEL_REG(8))(net)
    net = AveragePooling2D()(net)
    net = RQConv2D(16, 3, padding="same", activation="relu", kernel_regularizer=KERNEL_REG(16))(net)
    net = AveragePooling2D()(net)
    net = RQConv2D(8, 3, padding="same", activation="relu", kernel_regularizer=KERNEL_REG(8))(net)
    net = AveragePooling2D()(net)
    net = Dropout(0.5)(net)
    net = Flatten()(net)
    net = Dense(10)(net)
    net = Softmax()(net)
    return Model(inputs=inputs, outputs=net)


def l1_model(reg_rate):
    KERNEL_REG = lambda n: l1(reg_rate)

    inputs = Input((28, 28, 1))
    net = Conv2D(4, 3, padding="same")(inputs)
    net = RQConv2D(8, 3, padding="same", activation="relu", kernel_regularizer=KERNEL_REG(8))(net)
    net = AveragePooling2D()(net)
    net = RQConv2D(16, 3, padding="same", activation="relu", kernel_regularizer=KERNEL_REG(16))(net)
    net = AveragePooling2D()(net)
    net = RQConv2D(8, 3, padding="same", activation="relu", kernel_regularizer=KERNEL_REG(8))(net)
    net = AveragePooling2D()(net)
    net = Dropout(0.5)(net)
    net = Flatten()(net)
    net = Dense(10)(net)
    net = Softmax()(net)
    return Model(inputs=inputs, outputs=net)


def l2_model(reg_rate):
    KERNEL_REG = lambda n: l2(reg_rate)

    inputs = Input((28, 28, 1))
    net = Conv2D(4, 3, padding="same")(inputs)
    net = RQConv2D(8, 3, padding="same", activation="relu", kernel_regularizer=KERNEL_REG(8))(net)
    net = AveragePooling2D()(net)
    net = RQConv2D(16, 3, padding="same", activation="relu", kernel_regularizer=KERNEL_REG(16))(net)
    net = AveragePooling2D()(net)
    net = RQConv2D(8, 3, padding="same", activation="relu", kernel_regularizer=KERNEL_REG(8))(net)
    net = AveragePooling2D()(net)
    net = Dropout(0.5)(net)
    net = Flatten()(net)
    net = Dense(10)(net)
    net = Softmax()(net)
    return Model(inputs=inputs, outputs=net)


def none_model(reg_rate):
    KERNEL_REG = lambda n: None

    inputs = Input((28, 28, 1))
    net = Conv2D(4, 3, padding="same")(inputs)
    net = RQConv2D(8, 3, padding="same", activation="relu", kernel_regularizer=KERNEL_REG(8))(net)
    net = AveragePooling2D()(net)
    net = RQConv2D(16, 3, padding="same", activation="relu", kernel_regularizer=KERNEL_REG(16))(net)
    net = AveragePooling2D()(net)
    net = RQConv2D(8, 3, padding="same", activation="relu", kernel_regularizer=KERNEL_REG(8))(net)
    net = AveragePooling2D()(net)
    net = Dropout(0.5)(net)
    net = Flatten()(net)
    net = Dense(10)(net)
    net = Softmax()(net)
    return Model(inputs=inputs, outputs=net)
