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

#pragma once
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {
namespace vision {

namespace detection {
/*! @brief Postprocessor object for YOLOv8 serials model.
 */
class FASTDEPLOY_DECL YOLOv8Postprocessor {
 public:
  YOLOv8Postprocessor();
  YOLOv8Postprocessor(float conf_threshold, float nms_threshold, bool multi_label, float max_wh);

  virtual bool Run(const std::vector<FDTensor>& tensors,
     std::vector<DetectionResult>* results,
     const std::vector<std::map<std::string, std::array<float, 2>>>& ims_info);

  void SetConfThreshold(const float& conf_threshold) { conf_threshold_ = conf_threshold; }
  float GetConfThreshold() const { return conf_threshold_; }

  void SetNMSThreshold(const float& nms_threshold) { nms_threshold_ = nms_threshold; }
  float GetNMSThreshold() const { return nms_threshold_; }

  void SetMultiLabel(bool multi_label) { multi_label_ = multi_label; }
  bool GetMultiLabel() const { return multi_label_; }

  // 类别数：对应模型输出 channel 数中类别部分，即 24 - 4 = 20
  void SetNumClasses(int num_classes) { num_classes_ = num_classes; }
  int GetNumClasses() const { return num_classes_; }

  // sigmoid 开关：true 表示模型输出是 logits（需要 sigmoid），false 表示已是概率
  void SetClsNeedSigmoid(bool need) { cls_need_sigmoid_ = need; }
  bool GetClsNeedSigmoid() const { return cls_need_sigmoid_; }

  void sigmoid(FDTensor& tensor) {
    float* data = reinterpret_cast<float*>(tensor.MutableData());
    int numel = tensor.Numel();
    for (int i = 0; i < numel; ++i) {
      data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
  }

  void DivideTensorByTwo(FDTensor& tensor) {
    if (tensor.dtype != FP32) {
      std::cerr << "Tensor data type is not FP32." << std::endl;
      return;
    }
    auto shape = tensor.shape;
    size_t num_elements = 1;
    for (size_t dim : shape) { num_elements *= dim; }
    float* data = static_cast<float*>(tensor.Data());
    for (size_t i = 0; i < num_elements; ++i) { data[i] /= 2.0f; }
  }

  // 调试开关：true=打印 shape/原始值/NMS结果，确认格式后改 false
  void SetDebugPrint(bool v) { debug_print_ = v; }
  bool GetDebugPrint() const { return debug_print_; }

 protected:
  float conf_threshold_;
  float nms_threshold_;
  bool  multi_label_;
  float max_wh_;
  int   num_classes_;       // 类别数，从 config 读取，默认 20
  bool  cls_need_sigmoid_;  // 类别通道是否需要 sigmoid，默认 true
  bool  debug_print_;       // 调试打印开关
};

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
