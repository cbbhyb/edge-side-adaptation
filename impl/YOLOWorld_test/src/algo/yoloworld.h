// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.  //NOLINT
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

#pragma once

#include "fastdeploy/fastdeploy_model.h"
#include "preprocessor_yolov8.h"
#include "postprocessor_yolov8.h"

namespace fastdeploy {
namespace vision {
namespace detection {
/*! @brief YOLOWorld model object used when to load a YOLOWorld model exported for Ascend.
 */
class FASTDEPLOY_DECL YOLOWorld : public FastDeployModel {
 public:
  YOLOWorld(const std::string& model_file, const std::string& params_file = "",
            const RuntimeOption& custom_option = RuntimeOption(),
            const ModelFormat& model_format = ModelFormat::ONNX,
            std::shared_ptr<YOLOv8Preprocessor> pre = nullptr,
            std::shared_ptr<YOLOv8Postprocessor> post = nullptr);

  std::string ModelName() const { return "yoloworld"; }

  virtual bool Predict(const cv::Mat& img, DetectionResult* result);

  virtual bool BatchPredict(const std::vector<cv::Mat>& imgs,
                            std::vector<DetectionResult>* results);

  virtual YOLOv8Preprocessor& GetPreprocessor() { return *preprocessor_; }
  virtual YOLOv8Postprocessor& GetPostprocessor() { return *postprocessor_; }

 protected:
  bool Initialize();
  std::shared_ptr<YOLOv8Preprocessor> preprocessor_;
  std::shared_ptr<YOLOv8Postprocessor> postprocessor_;
};

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
