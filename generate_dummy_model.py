#!/usr/bin/env python3
from tensorflow.keras.layers import Input, DepthwiseConv2D
from tensorflow.keras.models import Model

######## TEST - large - dwconv 5x5 - stride 2  #########################
x0 = Input(shape=(52, 92, 480))
x = DepthwiseConv2D((5, 5), strides=(2, 2), padding='same', depthwise_initializer='LecunNormal', activation='linear', use_bias=False, trainable=False)(x0)

model = Model([x0], [x], name='test')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.save('large_depthwiseConv_5x5_stride_2.h5')
model.save('large_depthwiseConv_5x5_stride_2')

######## TEST - large - dwconv 5x5 - stride 1  #########################
x0 = Input(shape=(52, 92, 480))
x = DepthwiseConv2D((5, 5), padding='same', depthwise_initializer='LecunNormal', activation='linear', use_bias=False, trainable=False)(x0)

model = Model([x0], [x], name='test')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.save('large_depthwiseConv_5x5_stride_1.h5')
model.save('large_depthwiseConv_5x5_stride_1')

######## TEST - medium - dwconv 5x5 - stride 2  #########################
x0 = Input(shape=(32, 32, 384))
x = DepthwiseConv2D((5, 5), strides=(2, 2), padding='same', depthwise_initializer='LecunNormal', activation='linear', use_bias=False, trainable=False)(x0)

model = Model([x0], [x], name='test')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.save('medium_depthwiseConv_5x5_stride_2.h5')
model.save('medium_depthwiseConv_5x5_stride_2')