#!/usr/bin/python3

import tensorflow as tf
from tensorflow import python as tf_python
import numpy as np


def representative_data_l():
    for _ in range(100):
      data = np.random.rand(1, 52, 92, 480)
      yield [data.astype(np.float32)]

def representative_data_m():
    for _ in range(100):
      data = np.random.rand(1, 32, 32, 384)
      yield [data.astype(np.float32)]

######## Conversion - INT8 #########
tf_model = tf.keras.models.load_model('large_depthwiseConv_5x5_stride_2.h5')

# Setting batch size into 1 to prevent this error while inferring the model=> ERROR: Attempting to use a delegate that only supports static-sized tensors
for i, _ in enumerate(tf_model.inputs):
    tf_model.inputs[i].shape._dims[0] = tf_python.framework.tensor_shape.Dimension(1)

converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_l
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]


tflite_model_quantized = converter.convert()
tflite_model_quantized_file = 'int8_large_depthwiseConv_5x5_stride_2.tflite'

with open(tflite_model_quantized_file, 'wb') as f:
    f.write(tflite_model_quantized)

#################
tf_model = tf.keras.models.load_model('large_depthwiseConv_5x5_stride_1.h5')

# Setting batch size into 1 to prevent this error while inferring the model=> ERROR: Attempting to use a delegate that only supports static-sized tensors
for i, _ in enumerate(tf_model.inputs):
    tf_model.inputs[i].shape._dims[0] = tf_python.framework.tensor_shape.Dimension(1)

converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_l
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]


tflite_model_quantized = converter.convert()
tflite_model_quantized_file = 'int8_large_depthwiseConv_5x5_stride_1.tflite'

with open(tflite_model_quantized_file, 'wb') as f:
    f.write(tflite_model_quantized)

#################
tf_model = tf.keras.models.load_model('medium_depthwiseConv_5x5_stride_2.h5')

# Setting batch size into 1 to prevent this error while inferring the model=> ERROR: Attempting to use a delegate that only supports static-sized tensors
for i, _ in enumerate(tf_model.inputs):
    tf_model.inputs[i].shape._dims[0] = tf_python.framework.tensor_shape.Dimension(1)

converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_m
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]


tflite_model_quantized = converter.convert()
tflite_model_quantized_file = 'int8_medium_depthwiseConv_5x5_stride_2.tflite'

with open(tflite_model_quantized_file, 'wb') as f:
    f.write(tflite_model_quantized)

######## Conversion - FP32 #########
tf_model = tf.keras.models.load_model('large_depthwiseConv_5x5_stride_2.h5')

# Setting batch size into 1 to prevent this error while inferring the model=> ERROR: Attempting to use a delegate that only supports static-sized tensors
for i, _ in enumerate(tf_model.inputs):
    tf_model.inputs[i].shape._dims[0] = tf_python.framework.tensor_shape.Dimension(1)

converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

tflite_model_quantized = converter.convert()
tflite_model_quantized_file = 'fp32_large_depthwiseConv_5x5_stride_2.tflite'

with open(tflite_model_quantized_file, 'wb') as f:
    f.write(tflite_model_quantized)

#################
tf_model = tf.keras.models.load_model('large_depthwiseConv_5x5_stride_1.h5')

# Setting batch size into 1 to prevent this error while inferring the model=> ERROR: Attempting to use a delegate that only supports static-sized tensors
for i, _ in enumerate(tf_model.inputs):
    tf_model.inputs[i].shape._dims[0] = tf_python.framework.tensor_shape.Dimension(1)

converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

tflite_model_quantized = converter.convert()
tflite_model_quantized_file = 'fp32_large_depthwiseConv_5x5_stride_1.tflite'

with open(tflite_model_quantized_file, 'wb') as f:
    f.write(tflite_model_quantized)

#################
tf_model = tf.keras.models.load_model('medium_depthwiseConv_5x5_stride_2.h5')

# Setting batch size into 1 to prevent this error while inferring the model=> ERROR: Attempting to use a delegate that only supports static-sized tensors
for i, _ in enumerate(tf_model.inputs):
    tf_model.inputs[i].shape._dims[0] = tf_python.framework.tensor_shape.Dimension(1)

converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

tflite_model_quantized = converter.convert()
tflite_model_quantized_file = 'fp32_medium_depthwiseConv_5x5_stride_2.tflite'

with open(tflite_model_quantized_file, 'wb') as f:
    f.write(tflite_model_quantized)

