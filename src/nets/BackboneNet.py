from functools import wraps

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate,
                                     Conv2D, Layer, MaxPooling2D, SeparableConv2D,
                                     ZeroPadding2D,GlobalAveragePooling2D, Dense, Multiply)
from tensorflow.keras.regularizers import l2
from utils.utils import compose
from nets.attention import se_block, cbam_block, eca_block, ca_block, bam_block

attention = [se_block, cbam_block, eca_block, ca_block, bam_block]

class SiLU(Layer):
    def __init__(self, **kwargs):
        super(SiLU, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.sigmoid(inputs)

    def get_config(self):
        config = super(SiLU, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

class Focus(Layer):
    def __init__(self):
        super(Focus, self).__init__()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 2 if input_shape[1] != None else input_shape[1], input_shape[2] // 2 if input_shape[2] != None else input_shape[2], input_shape[3] * 4)

    def call(self, x): 
        return tf.concat(
            [x[...,  ::2,  ::2, :],   
             x[..., 1::2,  ::2, :],   
             x[...,  ::2, 1::2, :],  
             x[..., 1::2, 1::2, :]],  
             axis=-1
        )

@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02), 'kernel_regularizer' : l2(kwargs.get('weight_decay', 5e-4))}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2, 2) else 'same'   
    try:
        del kwargs['weight_decay']
    except:
        pass
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def Darknet_SeparableConv2D_BN_SiLU(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    if "name" in kwargs.keys():
        no_bias_kwargs['name'] = kwargs['name'] + '.conv'
    return compose(
        SeparableConv2D(*args, **no_bias_kwargs),
        BatchNormalization(momentum = 0.97, epsilon = 0.001, name = kwargs['name'] + '.bn'),
        SiLU())
       
def DarknetConv2D_BN_SiLU(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    if "name" in kwargs.keys():
        no_bias_kwargs['name'] = kwargs['name'] + '.conv'
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(momentum = 0.97, epsilon = 0.001, name = kwargs['name'] + '.bn'),
        SiLU())

def Bottleneck(x, out_channels, shortcut=True, weight_decay=5e-4, name = ""):
    y = compose(
            DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.cv1'),
            DarknetConv2D_BN_SiLU(out_channels, (3, 3), weight_decay=weight_decay, name = name + '.cv2'))(x)
            # Darknet_SeparableConv2D_BN_SiLU(out_channels, (1, 1), padding="same", name = name + '.cv1'),
            # Darknet_SeparableConv2D_BN_SiLU(out_channels, (3, 3), padding="same", name = name + '.cv2'))(x)
    if shortcut:
        y = Add()([x, y])
    y = attention[4](y, name = name + '.cam')
    return y


# 轻量级C3+残差边
def C3(x, num_filters, num_blocks, n=4, shortcut= True, expansion=0.5, weight_decay=5e-4, name=""):
    hidden_channels = int(num_filters * expansion)
    x_1 = Darknet_SeparableConv2D_BN_SiLU(hidden_channels, (1, 1), name = name + '.cv1')(x)
    x_2 = Darknet_SeparableConv2D_BN_SiLU(hidden_channels, (1, 1), name = name + '.cv2')(x)
    for i in range(n):
        # x_1 = Bottleneck(x_1, hidden_channels, shortcut=shortcut, weight_decay=weight_decay,name = name + '.m.' + str(i))
        x_3 = Bottleneck(x_1 , hidden_channels, shortcut=shortcut, name = name + '.m.' + str(i))
    
    route = Concatenate()([x_1, x_2, x_3])
    return DarknetConv2D_BN_SiLU(num_filters, (1, 1), name = name + '.cv3')(route)

def SPPBottleneck(x, out_channels, weight_decay=5e-4, name = ""):
    x = DarknetConv2D_BN_SiLU(out_channels // 2, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x)
    maxpool1 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    maxpool3 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
    x = Concatenate()([x, maxpool1, maxpool2, maxpool3])
    x = DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)
    return x
    
def resblock_body(x, num_filters, num_blocks, expansion=0.5, shortcut=True, last=False, weight_decay=5e-4, name = ""):
    x = ZeroPadding2D(((1, 0),(1, 0)))(x)
    x = DarknetConv2D_BN_SiLU(num_filters, (3, 3), strides = (2, 2), weight_decay=weight_decay, name = name + '.0')(x)
    if last:
        x = SPPBottleneck(x, num_filters, weight_decay=weight_decay, name = name + '.1')
    return C3(x, num_filters, num_blocks, shortcut=shortcut, expansion=expansion, weight_decay=weight_decay, name = name + '.1' if not last else name + '.2')

def concen(x1, channels):
    x1_att =  Conv2D(channels,(1,1),activation='relu')(x1)
    x1_att = GlobalAveragePooling2D()(x1_att)
    x1_att = Dense(channels,activation='softmax')(x1_att)
    x1_att = tf.expand_dims(tf.expand_dims(x1_att, axis=1),axis=1)
    x1_att = tf.tile(x1_att, [1, x1_att.shape[1], x1_att.shape[2], 1])
    x1_weight = Multiply()([x1,x1_att])
    return x1_weight

def darknet_body(inputs, base_channels, base_depth, weight_decay=5e-4):

    x1,x2 = inputs
    x1 = Focus()(x1)
    x2 = Focus()(x2)

    x1 = DarknetConv2D_BN_SiLU(base_channels, (3, 3), name = 'backbone.stem.conv11')(x1) 
    x2 = DarknetConv2D_BN_SiLU(base_channels, (3, 3), name = 'backbone.stem.conv12')(x2)

    x1 = resblock_body(x1, base_channels * 2, base_depth, name = 'backbone.dark21')
    x2 = resblock_body(x2, base_channels * 2, base_depth, name = 'backbone.dark22')
  
    x1 = resblock_body(x1, base_channels * 4, base_depth * 3, name = 'backbone.dark31')
    x2 = resblock_body(x2, base_channels * 4, base_depth * 3, name = 'backbone.dark32')
    
    x1_weight = concen(x1,base_channels*4)
    x2_weight = concen(x2,base_channels*4)
    feat1 = Add()([x1_weight,x2_weight])

    Pfeat1_upsample = C3(feat1, int(base_channels * 4), base_depth, shortcut = False, name = 'feat1_for_upsample1_feat1')
    feat1 = Pfeat1_upsample 
    # 80, 80, 256 => 40, 40, 512
    
    x = resblock_body(feat1, base_channels * 8, base_depth * 3, weight_decay=weight_decay, name = 'backbone.dark4')
    feat2 = x
    # 40, 40, 512 => 20, 20, 1024
    x = resblock_body(x, base_channels * 16, base_depth, shortcut=False, last=True, weight_decay=weight_decay, name = 'backbone.dark5')
    feat3 = x
    return feat1,feat2,feat3

