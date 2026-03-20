// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "yoloworld.h"
#ifdef PERFORMANCE_TEST
#include "common/utils/time_stamp.hpp"
#include "common/utils/utils.hpp"
#endif

namespace fastdeploy {
namespace vision {
namespace detection {

YOLOWorld::YOLOWorld(const std::string& model_file, const std::string& params_file,
                     const RuntimeOption& custom_option,
                     const ModelFormat& model_format,
                     std::shared_ptr<YOLOv8Preprocessor> pre,
                     std::shared_ptr<YOLOv8Postprocessor> post) {
  if (model_format == ModelFormat::ONNX) {
    valid_cpu_backends = {Backend::OPENVINO, Backend::ORT};
    valid_gpu_backends = {Backend::ORT, Backend::TRT};
  } else if (model_format == ModelFormat::SOPHGO) {
    valid_sophgonpu_backends = {Backend::SOPHGOTPU};
  } else if (model_format == ModelFormat::ASCEND) {
    valid_ascend_backends = {Backend::ASCEND};
  } else {
    valid_cpu_backends = {Backend::PDINFER, Backend::ORT, Backend::LITE};
    valid_gpu_backends = {Backend::PDINFER, Backend::ORT, Backend::TRT};
  }
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;

  preprocessor_  = pre  ? pre  : std::make_shared<YOLOv8Preprocessor>();
  postprocessor_ = post ? post : std::make_shared<YOLOv8Postprocessor>();
  initialized = Initialize();
}

bool YOLOWorld::Initialize() {
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  return true;
}

bool YOLOWorld::Predict(const cv::Mat& im, DetectionResult* result) {
  std::vector<DetectionResult> results;
  if (!BatchPredict({im}, &results)) {
    return false;
  }
  *result = std::move(results[0]);
  return true;
}

bool YOLOWorld::BatchPredict(const std::vector<cv::Mat>& images,
                             std::vector<DetectionResult>* results) {
  std::vector<std::map<std::string, std::array<float, 2>>> ims_info;
  std::vector<FDMat> fd_images = WrapMat(images);
#ifdef PERFORMANCE_TEST
  auto ts = std::make_shared<TimeStamp>();
  LOG_TS(ts, "start");
#endif
  if (!preprocessor_->Run(&fd_images, &reused_input_tensors_, &ims_info)) {
    FDERROR << "Failed to preprocess the input image." << std::endl;
    return false;
  }
#ifdef PERFORMANCE_TEST
  LOG_TS(ts, "preprocess");
#endif
  reused_input_tensors_[0].name = InputInfoOfRuntime(0).name;
  if (!Infer(reused_input_tensors_, &reused_output_tensors_)) {
    FDERROR << "Failed to inference by runtime." << std::endl;
    return false;
  }
#ifdef PERFORMANCE_TEST
  LOG_TS(ts, "infer");
#endif
  if (!postprocessor_->Run(reused_output_tensors_, results, ims_info)) {
    FDERROR << "Failed to postprocess the inference results by runtime." << std::endl;
    return false;
  }
#ifdef PERFORMANCE_TEST
  LOG_TS(ts, "postprocess");
#endif
  return true;
}

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
