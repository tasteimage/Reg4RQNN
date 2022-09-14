"""
退化四元全连接层，加上通道选择之后的批量归一化
"""
from keras.layers import Layer, InputSpec
from keras import backend as K
from keras import initializers, regularizers, constraints, activations
from tensorflow import real, imag, concat
from keras.initializers import Zeros
import numpy as np
import tensorflow as tf


class Dense(Layer):
    # 初始化方法
    def __init__(
        self,
        units,  # 神经元个数
        activation=None,  # 激活函数
        use_bias=True,  # 是否使用偏置
        bias_initializer="zeros",  # 偏置的初始化方法
        kernel_regularizer=None,  # 权值矩阵的正则化方法
        bias_regularizer=None,  # 偏置的正则化方法
        kernel_constraint=None,  # 权值矩阵的约束项
        bias_constraint=None,  # 偏置的约束项
        **kwargs
    ):
        super(Dense, self).__init__()

        # 显式成员设置
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # 隐式成员设定
        self.input_spec = InputSpec(ndim=2)  # 输入张量的维数
        self.supports_masking = True

    # 定义层内需要的变量
    def build(self, input_shape):
        assert len(input_shape) == 2

        input_dim = input_shape[-1] // 4
        kernel_shape = (input_dim, self.units)

        #  s = 1.0 / input_dim
        # TODO: 初始化最优为平均分布[-0.25, 0.25]
        a = np.random.uniform(-0.25, 0.25, kernel_shape)
        b = np.random.uniform(-0.25, 0.25, kernel_shape)
        c = np.random.uniform(-0.25, 0.25, kernel_shape)
        d = np.random.uniform(-0.25, 0.25, kernel_shape)

        def init_z1(shape, dtype=None, **kwargs):
            return a + b * 1j

        def init_z2(shape, dtype=None, **kwargs):
            return c + d * 1j

        self.z1_kernel = self.add_weight(
            shape=kernel_shape, initializer=init_z1, name="z1_kernel", regularizer=self.kernel_regularizer, constraint=self.kernel_constraint,
        )

        self.z2_kernel = self.add_weight(
            shape=kernel_shape, initializer=init_z2, name="z2_kernel", regularizer=self.kernel_regularizer, constraint=self.kernel_constraint,
        )

        # 偏置

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(4 * self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=2, axes={-1: 4 * input_dim})
        self.built = True

    # 描述调用时的计算行为
    def call(self, inputs):
        z1, z2 = self.z1_kernel, self.z2_kernel
        a = (real(z1) + real(z2)) / 2
        b = (imag(z1) + imag(z2)) / 2
        c = (real(z1) - real(z2)) / 2
        d = (imag(z1) - imag(z2)) / 2

        mat1 = concat((a, -b, c, -d), axis=-1)
        mat2 = concat((b, a, d, c), axis=-1)
        mat3 = concat((c, -d, a, -b), axis=-1)
        mat4 = concat((d, c, b, a), axis=-1)

        mat = concat((mat1, mat2, mat3, mat4), axis=0)

        output = inputs @ mat

        if self.use_bias:
            output = K.bias_add(output, self.bias)

        if self.activation:
            output = self.activation(output)

        if self.kernel_regularizer:
            output = self.kernel_regularizer(output)

        return output

    # 供调试时使用 可以在不调用运算的情况下获得给定输入行列数时输出的形状
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = 4 * self.units
        return tuple(output_shape)

    # 供调试时使用 方便开发者查看层定义时的参数设置情况
    def get_config(self):
        config = {
            "units": self.units,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "kernel_constraint": self.kernel_constraint,
            "bias_constraint": self.bias_constraint,
        }

        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
