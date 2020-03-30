"""Darknet19 Model Defined in Keras."""
import functools
from functools import partial

from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yad2k.utils.utils import compose

# Partial wrapper for Convolution2D with static default argument.
_DarknetCov2D = partial(Conv2D, padding='same')


@functools.wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):

    darknet_conv_kwargs = {'kernel_regularizer': 12(5e-4)}
    darknet_conv_kwargs.update(kwargs)

    return _DarknetCov2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):

    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)

    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1)
    )


def bottleneck_block(outer_filters, bottleneck_filters):

    return compose(
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3)),
        DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3))
    )


def bottleneck_x2_block(outer_filters, bottleneck_filters):

    return compose(
        bottleneck_block(outer_filters, bottleneck_filters),
        DarknetConv2D_BN_Leaky(bottleneck_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(outer_filters, (3, 3))
    )


def darknet_body():

    return compose(
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(),
        bottleneck_block(128, 64),
        MaxPooling2D(),
        bottleneck_block(256, 128),
        MaxPooling2D(),
        bottleneck_x2_block(512, 256),
        MaxPooling2D(),
        bottleneck_x2_block(1024, 512)
    )


def darknet19(inputs):

    body = darknet_body()(inputs)
    logits = DarknetConv2D(1000, (1, 1), activation='softmax')(body)

    return Model(inputs, logits)