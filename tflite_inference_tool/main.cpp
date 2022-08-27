#include "tensorflow/lite/c/c_api.h"    
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"

#include <algorithm>
#include <vector>
#include <random>
#include <iostream>
#include <cassert>
#include "cxxopts.hpp"

double cosine_similarity(std::vector<float> const &A,
                         std::vector<float> const &B) 
{
  double dot = 0.0, denom_a = 0.0, denom_b = 0.0;
  for (unsigned int i = 0u; i < A.size(); ++i) {
    dot += A[i] * B[i];
    denom_a += A[i] * A[i];
    denom_b += B[i] * B[i];
  }
  return dot / (sqrt(denom_a) * sqrt(denom_b));
}

void Benchmark(TfLiteInterpreter* interpreter, int iterations = 10)
{
    std::chrono::duration<double, std::milli> duration_total(0);
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::steady_clock::now();
        TfLiteInterpreterInvoke(interpreter);
        auto end = std::chrono::steady_clock::now();
        auto const diff = std::chrono::duration<double, std::milli>(end - start);
        duration_total += diff;
    }
    auto average = duration_total / iterations;
    std::cout << average.count() << "ms";
}

std::vector<float> nnapi_inference(const char* model_path, std::vector<float> const &randomInput, int outputLength)
{
    tflite::StatefulNnApiDelegate::Options opts;
    opts.accelerator_name = "google-edgetpu";
    opts.accelerator_name = "qti-dsp";

    TfLiteDelegate* nnapiDelegate = new tflite::StatefulNnApiDelegate(opts);

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

    TfLiteInterpreterOptionsAddDelegate(options, nnapiDelegate);

    TfLiteModel* model = TfLiteModelCreateFromFile(model_path);
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

    TfLiteInterpreterAllocateTensors(interpreter);
    auto* inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

    auto status = TfLiteTensorCopyFromBuffer(inputTensor, randomInput.data(), randomInput.size()*sizeof(float));
    assert(status == kTfLiteOk);

    TfLiteInterpreterInvoke(interpreter);

    std::vector<float> output(outputLength);
    auto const* outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    status = TfLiteTensorCopyToBuffer(outputTensor, output.data(), output.size()*sizeof(float));
    assert(status == kTfLiteOk);

    std::cout<<"\nThe average elapsed time in nnapi delegate: ";
    Benchmark(interpreter);

    TfLiteInterpreterDelete(interpreter);
    delete reinterpret_cast<tflite::StatefulNnApiDelegate*>(nnapiDelegate);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    return output;
}

std::vector<float> xnnpack_inference(const char* model_path, std::vector<float> const &randomInput, int outputLength)
{
    TfLiteXNNPackDelegateOptions opts = TfLiteXNNPackDelegateOptionsDefault();
    opts.num_threads = 4;
    TfLiteDelegate* xnnpackDelegate = TfLiteXNNPackDelegateCreate(&opts);

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

    TfLiteInterpreterOptionsAddDelegate(options, xnnpackDelegate);

    TfLiteModel* model = TfLiteModelCreateFromFile(model_path);
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

    TfLiteInterpreterAllocateTensors(interpreter);
    auto* inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

    auto status = TfLiteTensorCopyFromBuffer(inputTensor, randomInput.data(), randomInput.size()*sizeof(float));
    assert(status == kTfLiteOk);

    TfLiteInterpreterInvoke(interpreter);

    std::vector<float> output(outputLength);
    auto const* outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    status = TfLiteTensorCopyToBuffer(outputTensor, output.data(), output.size()*sizeof(float));
    assert(status == kTfLiteOk);

    std::cout<<"\nThe average elapsed time in xnnpack delegate: ";
    Benchmark(interpreter);

    TfLiteInterpreterDelete(interpreter);
    TfLiteXNNPackDelegateDelete(xnnpackDelegate);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    return output;
}

int main(int argc, char** argv) {

    cxxopts::Options options("ModelTest", "Test model on mobile device");

    options.add_options()
        ("a,model_a", "model name (int8 tflite version)", cxxopts::value<std::string>())
        ("b,model_b", "model name (fp32 tflite version)", cxxopts::value<std::string>())
        ("o,output_shape", "height, width and channel of output", cxxopts::value<std::vector<int>>())
        ("i,input_shape", "height, width and channel of input", cxxopts::value<std::vector<int>>());

    auto result = options.parse(argc, argv);

    if (!result.count("model_a") || !result.count("model_b"))
    {
        throw std::runtime_error("You must provide model name for both int8 and fp32 versions.");
    }
    if (!result.count("input_shape") || !result.count("output_shape"))
    {
        throw std::runtime_error("You must provide input and output shapes.");
    }
    const std::vector<int> inputShape = result["input_shape"].as<std::vector<int>>();
    const std::vector<int> outputShape = result["output_shape"].as<std::vector<int>>();

    auto model_a = "./" + result["model_a"].as<std::string>();
    auto model_b = "./" + result["model_b"].as<std::string>();

    std::vector<float> randomInput(inputShape.at(0)*inputShape.at(1)*inputShape.at(2));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.f, 1.f); 
    std::generate(randomInput.begin(), randomInput.end(), [&](){return dis(gen);});

    auto output_nnapi = nnapi_inference(const_cast<char*>(model_a.c_str()), randomInput, outputShape.at(0)*outputShape.at(1)*outputShape.at(2));

    auto output_xnnpack = xnnpack_inference(const_cast<char*>(model_b.c_str()), randomInput, outputShape.at(0)*outputShape.at(1)*outputShape.at(2));

    auto CS = cosine_similarity(output_nnapi, output_xnnpack);
    
    std::cout<<"\nCosine Similarity: "<<CS<<std::endl;

}
