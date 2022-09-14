from keras import backend as K
import tensorflow as tf


def rq_reg(alpha, filters):
    """
    此方案效果不错
    """
    def inner(weight_matrix):
        z1 = weight_matrix[:, :, :, :filters]
        z2 = weight_matrix[:, :, :, filters:]
        z_module = (z1 * z1 - z2 * z2) ** 2  # 复模长
        return alpha * K.sum(K.concatenate((z_module, z_module), axis=-1))

    return inner
