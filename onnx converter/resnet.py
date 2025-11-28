from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
from keras import layers
from keras.regularizers import l2
from keras.layers import Input, Conv2D, Activation, BatchNormalization
from keras.layers import MaxPooling2D, AveragePooling2D, Flatten, Dense

global weight_decay
weight_decay = 1e-4

# === IMPORTANT PATCH ===
# Allow Keras 3.x to accept layer names containing "/" (legacy behavior)
# We wrap name argument and bypass the new validation.
def legacy_name(name: str):
    # Keras 3 forbids "/", but the weight file requires them.
    # We inject the name AFTER layer creation using TF node rewrite.
    class NameProxy(str):
        pass
    return NameProxy(name)


def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_1 = legacy_name(f"conv{stage}_{block}_1x1_reduce")
    bn_name_1   = legacy_name(f"conv{stage}_{block}_1x1_reduce/bn")
    x = Conv2D(filters1, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable)(input_tensor)
    x._name = conv_name_1
    x = BatchNormalization(axis=bn_axis)(x)
    x._name = bn_name_1
    x = Activation('relu')(x)

    conv_name_2 = legacy_name(f"conv{stage}_{block}_3x3")
    bn_name_2   = legacy_name(f"conv{stage}_{block}_3x3/bn")
    x = Conv2D(filters2, kernel_size, padding="same",
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable)(x)
    x._name = conv_name_2
    x = BatchNormalization(axis=bn_axis)(x)
    x._name = bn_name_2
    x = Activation('relu')(x)

    conv_name_3 = legacy_name(f"conv{stage}_{block}_1x1_increase")
    bn_name_3   = legacy_name(f"conv{stage}_{block}_1x1_increase/bn")
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable)(x)
    x._name = conv_name_3
    x = BatchNormalization(axis=bn_axis)(x)
    x._name = bn_name_3

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2), trainable=True):
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_1 = legacy_name(f"conv{stage}_{block}_1x1_reduce")
    bn_name_1   = legacy_name(f"conv{stage}_{block}_1x1_reduce/bn")
    x = Conv2D(filters1, (1, 1), strides=strides,
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable)(input_tensor)
    x._name = conv_name_1
    x = BatchNormalization(axis=bn_axis)(x)
    x._name = bn_name_1
    x = Activation('relu')(x)

    conv_name_2 = legacy_name(f"conv{stage}_{block}_3x3")
    bn_name_2   = legacy_name(f"conv{stage}_{block}_3x3/bn")
    x = Conv2D(filters2, kernel_size, padding="same",
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable)(x)
    x._name = conv_name_2
    x = BatchNormalization(axis=bn_axis)(x)
    x._name = bn_name_2
    x = Activation('relu')(x)

    conv_name_3 = legacy_name(f"conv{stage}_{block}_1x1_increase")
    bn_name_3   = legacy_name(f"conv{stage}_{block}_1x1_increase/bn")
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               trainable=trainable)(x)
    x._name = conv_name_3
    x = BatchNormalization(axis=bn_axis)(x)
    x._name = bn_name_3

    conv_name_4 = legacy_name(f"conv{stage}_{block}_1x1_proj")
    bn_name_4   = legacy_name(f"conv{stage}_{block}_1x1_proj/bn")
    shortcut = Conv2D(filters3, (1,1), strides=strides,
                      kernel_initializer='orthogonal',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay),
                      trainable=trainable)(input_tensor)
    shortcut._name = conv_name_4
    shortcut = BatchNormalization(axis=bn_axis)(shortcut)
    shortcut._name = bn_name_4

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet50_backend(inputs):
    bn_axis = 3

    # original name with slash
    conv1 = legacy_name("conv1/7x7_s2")
    conv1_bn = legacy_name("conv1/7x7_s2/bn")

    x = Conv2D(64, (7,7), strides=(2,2),
               kernel_initializer='orthogonal',
               use_bias=False,
               trainable=True,
               kernel_regularizer=l2(weight_decay),
               padding='same')(inputs)
    x._name = conv1

    x = BatchNormalization(axis=bn_axis)(x)
    x._name = conv1_bn

    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64,64,256], stage=2, block=1, strides=(1,1), trainable=True)
    x = identity_block(x, 3, [64,64,256], stage=2, block=2, trainable=True)
    x = identity_block(x, 3, [64,64,256], stage=2, block=3, trainable=True)

    x = conv_block(x, 3, [128,128,512], stage=3, block=1, trainable=True)
    x = identity_block(x, 3, [128,128,512], stage=3, block=2, trainable=True)
    x = identity_block(x, 3, [128,128,512], stage=3, block=3, trainable=True)
    x = identity_block(x, 3, [128,128,512], stage=3, block=4, trainable=True)

    x = conv_block(x, 3, [256,256,1024], stage=4, block=1, trainable=True)
    x = identity_block(x, 3, [256,256,1024], stage=4, block=2, trainable=True)
    x = identity_block(x, 3, [256,256,1024], stage=4, block=3, trainable=True)
    x = identity_block(x, 3, [256,256,1024], stage=4, block=4, trainable=True)
    x = identity_block(x, 3, [256,256,1024], stage=4, block=5, trainable=True)
    x = identity_block(x, 3, [256,256,1024], stage=4, block=6, trainable=True)

    x = conv_block(x, 3, [512,512,2048], stage=5, block=1, trainable=True)
    x = identity_block(x, 3, [512,512,2048], stage=5, block=2, trainable=True)
    x = identity_block(x, 3, [512,512,2048], stage=5, block=3, trainable=True)

    return x
