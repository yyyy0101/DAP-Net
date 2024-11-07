import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Activation, Add, Concatenate, Conv1D, Conv2D, Dense,
                          GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda, BatchNormalization,
                          Reshape, multiply)
import math

#改进后的SE
def se_block(input_feature, ratio=16, name=""):
	channel = K.int_shape(input_feature)[-1]

	se_feature1 = GlobalAveragePooling2D()(input_feature)
	se_feature1 = Reshape((1, 1, channel))(se_feature1)

	se_feature1 = Dense(channel // ratio,
					   activation='relu',
					   kernel_initializer='he_normal',
					   use_bias=False,
					   name = "se_block_one_1"+str(name))(se_feature1)
					   
	se_feature1 = Dense(channel,
					   kernel_initializer='he_normal',
					   use_bias=False,
					   name = "se_block_two_1"+str(name))(se_feature1)
	se_feature1 = Activation('sigmoid')(se_feature1)

	se_feature2 = GlobalMaxPooling2D()(input_feature)
	se_feature2 = Reshape((1, 1, channel))(se_feature2)
	se_feature2 = Dense(channel//ratio,
						activation='relu',
					    kernel_initializer='he_normal',
					    use_bias=False,
					    name = "se_block_one_2"+str(name))(se_feature2)
	se_feature2 = Dense(channel,
					   kernel_initializer='he_normal',
					   use_bias=False,
					   name = "se_block_two_2"+str(name))(se_feature2)
	se_feature2 = Activation('sigmoid')(se_feature2)
	se_feature = tf.add(se_feature1, se_feature2)
	se_feature = multiply([input_feature, se_feature])
	return se_feature

def channel_attention(input_feature, ratio=8, name=""):
	channel = K.int_shape(input_feature)[-1]
	
	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=False,
							 bias_initializer='zeros',
							 name = "channel_attention_shared_one_"+str(name))
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=False,
							 bias_initializer='zeros',
							 name = "channel_attention_shared_two_"+str(name))
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	max_pool = GlobalMaxPooling2D()(input_feature)

	avg_pool = Reshape((1,1,channel))(avg_pool)
	max_pool = Reshape((1,1,channel))(max_pool)

	avg_pool = shared_layer_one(avg_pool)
	max_pool = shared_layer_one(max_pool)

	avg_pool = shared_layer_two(avg_pool)
	max_pool = shared_layer_two(max_pool)
	
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)
	
	
	return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature, name=""):
	kernel_size = 7

	cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	concat = Concatenate(axis=3)([avg_pool, max_pool])

	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					kernel_initializer='he_normal',
					use_bias=False,
					name = "spatial_attention_"+str(name))(concat)	
	cbam_feature = Activation('sigmoid')(cbam_feature)
		
	return multiply([input_feature, cbam_feature])

def bam_block(input_feature, ratio=8, name=""):
	channel_feature = channel_attention(input_feature, ratio, name=name)
	spatial_feature = spatial_attention(input_feature, name=name)
	add_feature = tf.add(channel_feature,spatial_feature)
	add_feature = Activation('sigmoid')(add_feature)
	multi_feature = tf.multiply(add_feature,input_feature)
	bam_feature = tf.add(multi_feature,input_feature)
	return bam_feature


def cbam_block(cbam_feature, ratio=8, name=""):
	cbam_feature = channel_attention(cbam_feature, ratio, name=name)
	cbam_feature = spatial_attention(cbam_feature, name=name)
	return cbam_feature




