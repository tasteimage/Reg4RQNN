"""
退化四元卷积层
"""
from numpy.random import uniform
from numpy import concatenate
from keras.initializers import Initializer
from keras.initializers import _compute_fans
from keras.layers import Layer
from keras.layers import InputSpec  # 确定网络层输入数据的形状、类型、大小
from keras.utils import conv_utils
from keras import initializers, regularizers, constraints, activations
from keras import backend as K
from tensorflow import imag, real, concat, image


class Init(Initializer):
    """
    权重初始化方法：得到卷积网络的权重，其中的criterion参数表示初始化使用的是he策略还是
    glorot策略（又称为Xavier策略）。这两种初始化方法效果不同，使用的时候需要酌情考量
    """

    def __init__(
        self,
        kernel_size,  # 卷积核形状
        input_dim,  # 输入数据维数
        weight_dim,  # 权重的维数（1维、2维、3维）
        nb_filters=None,  # 卷积核个数 number of filters
        criterion="he",  # 初始化策略
        seed=None,  # 随机数种子
    ):
        # 保证维数合法
        assert len(kernel_size) == weight_dim
        assert weight_dim in {0, 1, 2, 3}

        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.criterion = criterion

    def __call__(self, shape, dtype=None):
        if self.nb_filters:
            kernel_shape = tuple(self.kernel_size) + (self.input_dim, self.nb_filters)
        else:
            # nb_filters为None，则默认卷积核数为1
            kernel_shape = (self.input_dim, self.kernel_size[-1])

        fan_in, fan_out = _compute_fans(kernel_shape)

        # 以下是两种不同的初始化策略
        if self.criterion == "glorot":
            s = 1.0 / (fan_in + fan_out)
        elif self.criterion == "he":
            s = 1.0 / fan_in
        else:
            raise ValueError("Invaild criterion: " + self.criterion)

        # >>> 不同的退化四元数初始化方法
        a = uniform(-s, s, kernel_shape)
        b = uniform(-s, s, kernel_shape)
        c = uniform(-s, s, kernel_shape)
        d = uniform(-s, s, kernel_shape)
        z1 = a + b * 1j
        z2 = c + d * 1j

        return concatenate([z1, z2], axis=-1)
        # <<<

    def get_config(self):
        # 反馈内部变量情况
        return {
            "kernel_size": self.kernel_size,
            "nb_filter": self.nb_filters,
            "input_dim": self.input_dim,
            "weight_dim": self.weight_dim,
            "criterion": self.criterion,
        }


class Conv(Layer):
    def __init__(
        self,
        rank,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        data_format=None,  # 'channels_last' or 'channels_first'
        dilation_rate=1,
        activation=None,
        activity_regularizer=None,
        use_bias=True,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs
    ):
        super(Conv, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, "kernel_size"
        )  # e.g. Conv2D(32, kernel_size = 3): 3 -> (3, 3)
        self.strides = conv_utils.normalize_tuple(strides, rank, "strides")
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, "dilation_rate")
        self.activation = activations.get(activation)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

        # 数据存储的时候是通道先还是通道后
        if rank == 1:
            self.data_format = "channels_last"
        else:
            self.data_format = K.normalize_data_format(data_format)

    def build(self, input_shape):
        if self.data_format == "channels_first":
            channel_axis = 1
        else:
            channel_axis = -1

        if input_shape[channel_axis] is None:
            raise ValueError("The channel dimension of the inputs should be defined. Found `None`.")

        input_dim = input_shape[channel_axis] // 4  # 四个通道是在一维上合并在一起的
        self.kernel_shape = self.kernel_size + (input_dim, self.filters)  # 计算出卷积核的形状
        # 利用前面定义的初始化方法获得卷积核
        self.kernel = self.add_weight(
            name="kernel",
            shape=self.kernel_shape,
            initializer=Init(
                kernel_size=self.kernel_size,
                input_dim=input_dim,
                weight_dim=self.rank,
                nb_filters=self.filters,
            ),
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )
        # 偏置
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(4 * self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )
        else:
            self.bias = None
        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim * 4})
        self.built = True

    def call(self, inputs):
        # >>> 定义退化四元数的计算过程
        if self.rank == 1:
            z1 = self.kernel[:, :, : self.filters]
            z2 = self.kernel[:, :, self.filters :]
            convFunc = K.conv1d
        elif self.rank == 2:
            z1 = self.kernel[:, :, :, : self.filters]
            z2 = self.kernel[:, :, :, self.filters :]
            convFunc = K.conv2d
        elif self.rank == 3:
            z1 = self.kernel[:, :, :, :, : self.filters]
            z2 = self.kernel[:, :, :, :, self.filters :]
            convFunc = K.conv3d

        a = (real(z1) + real(z2)) / 2
        b = (imag(z1) + imag(z2)) / 2
        c = (real(z1) - real(z2)) / 2
        d = (imag(z1) - imag(z2)) / 2

        mat1 = concat((a, -b, c, -d), axis=-2)
        mat2 = concat((b, a, d, c), axis=-2)
        mat3 = concat((c, -d, a, -b), axis=-2)
        mat4 = concat((d, c, b, a), axis=-2)
        mat = concat((mat1, mat2, mat3, mat4), axis=-1)
        # <<<

        output = convFunc(
            inputs,
            mat,
            strides=self.strides[0] if self.rank == 1 else self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate[0] if self.rank == 1 else self.dilation_rate,
        )

        # 加偏置
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format=self.data_format)

        # 加激活函数
        if self.activation:
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_last":
            space = input_shape[1:-1]
        elif self.data_format == "channels_first":
            space = input_shape[2:]

        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i],
            )
            new_space.append(new_dim)

        if self.data_format == "channels_last":
            return (input_shape[0],) + tuple(new_space) + (4 * self.filters,)
        elif self.data_format == "channels_first":
            return (input_shape[0],) + (4 * self.filters,) + tuple(new_space)

    def get_config(self):
        config = {
            "rank": self.rank,
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
            "dilation_rate": self.dilation_rate,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super(Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Conv1D(Conv):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding="valid",
        dilation_rate=1,
        activation=None,
        use_bias=True,
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(Conv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format="channels_last",
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

    def get_config(self):
        config = super(Conv1D, self).get_config()
        config.pop("rank")
        return config


class Conv2D(Conv):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(Conv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.pop("rank")
        return config


class Conv3D(Conv):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1, 1),
        activation=None,
        use_bias=True,
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(Conv3D, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

    def get_config(self):
        config = super(Conv3D, self).get_config()
        config.pop("rank")
        return config


def psnr_pred(y_true, y_pred):
    return image.psnr(y_true, y_pred, max_val=1.0)
