# NNAPI delegate issue with DepthwiseConv2D nodes

This repo contains scripts and a tool to reproduce the NNAPI delegate issue with models containing DepthwiseConv2D nodes. Our experiments have revealed that DepthwiseConv2D nodes generate a multitude of problems when using NNAPI delegate. Here is a summary of our findings:

1. Inferring DepthwiseConv2D nodes with XNNPACK delegate is always faster than the NNAPI delegate (tested on Snapdragon 855 and Snapdragon 865 - Android 12).
2. On devices with Snapdragon 888 (tested with Android 12), INT8 models starting with a DepthwiseConv2D node always crash with the NNAPI delegate. 
3. After a specific threshold for the node size, the elapsed time inside DepthwiseConv2D nodes increases significantly so that one single node can even take about 20 ms to run with NNAPI delegate. For example, the following model runs in 2.82 ms with NNAPI delegate on Snapdragon 855 (Android 10):

![int8_5_384](https://user-images.githubusercontent.com/45400368/186978999-c0dd7a75-42c5-4d3a-a2f9-571a565772db.png)

However, the next model which is very similar to the previous one runs in 19.62 ms on the same device with the same delegate:

![int8_5_480](https://user-images.githubusercontent.com/45400368/186979435-70365554-f26f-4adf-b96e-2133dd01e640.png)

4. On devices with Snapdragon 865 (tested with Android 12), quantized DepthwiseConv2D nodes with large kernel size (e.g. 7x7) cause accuracy loss with NNAPI delegate.

5. On devices with Snapdragon 855 (tested with Android 12), quantized DepthwiseConv2D nodes with stride size more than one always result in accuracy loss.

Note: To compute the accuracy loss, we consider the Cosine Similarity (CS) between the results generated by the INT8 tflite version with the NNAPI delegate and the expected results from the FP32 tflite version of the same model with the XNNPACK delegate.

## Building and converting the model
* `model_files` folder contains simple models representing the above-mentioned issues. 
  * You can also use `generate_dummy_model.py` to build the models and use `convert_model.py` to convert them to tflite.

## tflite_inference tool 
We have implemented a small tool to feed an input to our sample INT8 tflite models using the `NNAPI` delegate and compare the results with what we get from the corresponding FP32 tflite versions using the `XNNPACK` delegate.

### PREREQUISITES: ###
* Linux host computer
* Connectivity to the target device via adb
* Android NDK, version 22 or later
* CMake 3.18 or later

### BUILD INSTRUCTIONS ###
* Unzip the `tensorflow_lite_cpp_2_9_1_nightly.zip` file inside the `tflite_inference_tool` folder.
* In a terminal, from `tflite_inference_tool` folder:
```console
$ mkdir build
$ cd build
$ cmake -G "Unix Makefiles"
        -DCMAKE_SYSTEM_NAME=Android 
        -DANDROID_ABI=arm64-v8a 
        -DANDROID_STL=c++_shared 
        -DANDROID_NATIVE_API_LEVEL=27 
        -DCMAKE_VERBOSE_MAKEFILE=ON 
        -DCMAKE_TOOLCHAIN_FILE=<path-to-ndk>/build/cmake/android.toolchain.cmake 
        -DCMAKE_BUILD_TYPE=Release
        -DTensorFlowLite_ROOT=../tensorflow_lite_cpp_2_9_1_nightly ..
$ make
```
* Here, you must replace <path-to-ndk> with the absolute path of the ndk installed on your computer. If you installed NDK through Android studio, it is typically located at:
    `/home/<username>/Android/Sdk/ndk/<version>/` on Linux

* `tensorflow_lite_cpp_2_9_1_nightly` is TensorflowFlow Lite library (nightly version) package.
### Run INSTRUCTIONS ###
WARNING: This step will write to your `/data/local/tmp` folder on device. Please make sure existing files in that folder are backed up as needed.

In a terminal, from `tflite_inference_tool` folder:
```console
$ adb push ./build/model_test /data/local/tmp
$ adb push ./model_files /data/local/tmp
```

To run the tool you can use different parameters. In the following, we have listed the output of tool when running on Snapdragon 855: 

Sample 1:
```console
$ adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. ./model_test --model_a=model_files/int8_large_depthwiseConv_5x5_stride_2.tflite --model_b=model_files/fp32_large_depthwiseConv_5x5_stride_2.tflite --input_shape=52,92,480 --output_shape=26,46,480"

INFO: Created TensorFlow Lite delegate for NNAPI.
INFO: Initialized TensorFlow Lite runtime.
INFO: Replacing 3 node(s) with delegate (TfLiteNnapiDelegate) node, yielding 1 partitions.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
INFO: Replacing 1 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 1 partitions.

The average elapsed time in nnapi delegate: 19.2918ms
The average elapsed time in xnnpack delegate: 5.97154ms
Cosine Similarity: 0.799906 
```

Sample 2:
```console
$ adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. ./model_test --model_a=model_files/int8_large_depthwiseConv_5x5_stride_1.tflite --model_b=model_files/fp32_large_depthwiseConv_5x5_stride_1.tflite --input_shape=52,92,480 --output_shape=52,92,480"

INFO: Created TensorFlow Lite delegate for NNAPI.
INFO: Initialized TensorFlow Lite runtime.
INFO: Replacing 3 node(s) with delegate (TfLiteNnapiDelegate) node, yielding 1 partitions.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
INFO: Replacing 1 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 1 partitions.

The average elapsed time in nnapi delegate: 27.2882ms
The average elapsed time in xnnpack delegate: 9.43883ms
Cosine Similarity: 0.999899 
```

Sample 3:
```console
$ adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. ./model_test --model_a=model_files/int8_medium_depthwiseConv_5x5_stride_2.tflite --model_b=model_files/fp32_medium_depthwiseConv_5x5_stride_2.tflite --input_shape=32,32,384 --output_shape=16,16,384"

INFO: Created TensorFlow Lite delegate for NNAPI.
INFO: Initialized TensorFlow Lite runtime.
INFO: Replacing 3 node(s) with delegate (TfLiteNnapiDelegate) node, yielding 1 partitions.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
INFO: Replacing 1 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 1 partitions.

The average elapsed time in nnapi delegate: 2.58259ms
The average elapsed time in xnnpack delegate: 0.922166ms
Cosine Similarity: 0.839082 
```