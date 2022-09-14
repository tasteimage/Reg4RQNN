import keras
import sys
sys.path.append("..")
import src.util as util
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--epoch_num", type=int, default=100, help="训练轮数")
parser.add_argument("--batch_size", type=int, default=32, help="小批量大小")
parser.add_argument("--learning_rate", type=float, default=0.05, help="学习率")
parser.add_argument("--reg_rate", type=float, default=0.0001, help="正则化系数")
parser.add_argument("--train", action="store_true", help="是否训练")
parser.add_argument("--test", action="store_true", help="是否测试")
args = parser.parse_args()


#------------
# parameter
#------------
# 是否训练
TRAIN = args.train
# 是否测试
TEST = args.test
# 小批量的大小
BATCH_SIZE = args.batch_size
# 训练轮数
EPOCH_NUM = args.epoch_num
# 正则化系数
REG_RATE = args.reg_rate
# 模型参数需要保存的位置
WEIGHT_PATH = './cache/weight.h5'
# 训练日志需要保存的位置
LOG_PATH = './cache/log.npy'
# 学习率
LEARNING_RATE = args.learning_rate


#-----------
# function
#-----------
# 模型函数
MODEL_FUNC = lambda: util.l1_model(REG_RATE)
# 日志函数
LOG_FUNC = util.logger(LOG_PATH)
# 数据函数
DATA_FUNC = lambda: util.load_data()
# 损失函数
LOSS_FUNC = keras.losses.categorical_crossentropy
# 更新函数
OPT_FUNC = keras.optimizers.Adadelta(LEARNING_RATE)
# 评价函数
METRIC_FUNC = [ keras.metrics.categorical_accuracy ]


if __name__ == "__main__":
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = DATA_FUNC()

    model = MODEL_FUNC()
    model.compile(
        loss=LOSS_FUNC,
        optimizer=OPT_FUNC,
        metrics=METRIC_FUNC
    )

    if TRAIN:
        checkpoint = keras.callbacks.ModelCheckpoint(
            WEIGHT_PATH,
            verbose=1,
            save_best_only=True,
        )
        model.summary()
        model.fit(
            train_x,
            train_y,
            batch_size=BATCH_SIZE,
            epochs=EPOCH_NUM,
            validation_data=(val_x, val_y),
            callbacks=[checkpoint, LOG_FUNC],
        )

    if TEST:
        # 重载参数+评估
        model.load_weights(WEIGHT_PATH)
        model.summary()
        loss, accuracy = model.evaluate(test_x, test_y, verbose=1)
        print("loss: %.4f\taccuracy: %.4f" % (loss, accuracy))
