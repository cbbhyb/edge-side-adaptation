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

#include "postprocessor_yolov8.h"
#include "fastdeploy/vision/utils/utils.h"
#include "fastdeploy/function/concat.h"
#include "fastdeploy/function/elementwise.h"
#include "fastdeploy/function/split.h"

namespace fastdeploy {
namespace vision {
namespace detection {

YOLOv8Postprocessor::YOLOv8Postprocessor() {
  conf_threshold_ = 0.25;
  nms_threshold_ = 0.5;
  multi_label_ = true;
  max_wh_ = 7680.0;
  num_classes_ = 20;
  cls_need_sigmoid_ = true;
}

YOLOv8Postprocessor::YOLOv8Postprocessor(float conf_threshold, float nms_threshold,
                                          bool multi_label, float max_wh)
    : conf_threshold_(conf_threshold),
      nms_threshold_(nms_threshold),
      multi_label_(multi_label),
      max_wh_(max_wh),
      num_classes_(20),
      cls_need_sigmoid_(true) {}

bool YOLOv8Postprocessor::Run(
    const std::vector<FDTensor>& tensors, std::vector<DetectionResult>* results,
    const std::vector<std::map<std::string, std::array<float, 2>>>& ims_info) {

  // ----------------------------------------------------------------
  // 适配 YOLOWorld 单张量输出：shape = [1, 24, 8400]
  //   channel 0~3  : bbox 距离预测值 (lt_x, lt_y, rb_x, rb_y) — DFL 已解码
  //   channel 4~23 : 20 个类别的 logits（或概率，由 cls_need_sigmoid_ 控制）
  //
  // 内存布局(CHW)：data[(ch * 8400) + anchor_idx]
  // ----------------------------------------------------------------

  if (tensors.empty()) {
    FDERROR << "YOLOv8Postprocessor: no output tensors." << std::endl;
    return false;
  }

  // ----- Step 1: 生成 anchor_points（与原始逻辑完全一致） -----
  // 固定三个特征图尺寸：80x80, 40x40, 20x20（对应 stride 8,16,32）
  const int feat_hw[3][2] = {{80, 80}, {40, 40}, {20, 20}};
  const float strides[3] = {8.f, 16.f, 32.f};

  // anchor_points_xy[i] = {cx, cy}，共 8400 个
  struct AnchorPt { float x, y, stride; };
  std::vector<AnchorPt> anchor_pts;
  anchor_pts.reserve(8400);
  for (int s = 0; s < 3; ++s) {
    int h = feat_hw[s][0], w = feat_hw[s][1];
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        anchor_pts.push_back({j + 0.5f, i + 0.5f, strides[s]});
      }
    }
  }

  int num_anchors = static_cast<int>(anchor_pts.size()); // 8400

  // ----- Step 2: 读取合并张量 [1, 24, 8400] -----
  const FDTensor& t = tensors[0];
  if (t.dtype != FDDataType::FP32) {
    FDERROR << "Only support FP32 output." << std::endl;
    return false;
  }
  const float* data = reinterpret_cast<const float*>(t.Data());
  int batch = static_cast<int>(t.shape[0]);

  results->resize(batch);

  for (int bs = 0; bs < batch; ++bs) {
    (*results)[bs].Clear();
    (*results)[bs].Reserve(num_anchors);

    // 当前 batch 的数据起始偏移（shape [B, 24, 8400]）
    const float* bd = data + bs * (4 + num_classes_) * num_anchors;

    for (int i = 0; i < num_anchors; ++i) {
      // ----- 读取 bbox：channel 0~3，内存布局 [ch * 8400 + i] -----
      float lt_x = bd[0 * num_anchors + i]; // left distance
      float lt_y = bd[1 * num_anchors + i]; // top distance
      float rb_x = bd[2 * num_anchors + i]; // right distance
      float rb_y = bd[3 * num_anchors + i]; // bottom distance

      // dist2bbox：anchor_point ± distance * stride → xyxy
      float ax = anchor_pts[i].x;
      float ay = anchor_pts[i].y;
      float st = anchor_pts[i].stride;

      float x1 = (ax - lt_x) * st;
      float y1 = (ay - lt_y) * st;
      float x2 = (ax + rb_x) * st;
      float y2 = (ay + rb_y) * st;

      // ----- 读取类别分数：channel 4~23 -----
      float max_score = -1.f;
      int   best_cls  = -1;
      for (int c = 0; c < num_classes_; ++c) {
        float logit = bd[(4 + c) * num_anchors + i];
        float score = cls_need_sigmoid_ ? (1.f / (1.f + expf(-logit))) : logit;
        if (score > max_score) {
          max_score = score;
          best_cls  = c;
        }
      }

      if (max_score <= conf_threshold_) continue;

      // ----- NMS 偏移（multi_label 模式）-----
      float offset = multi_label_ ? best_cls * max_wh_ : 0.f;

      (*results)[bs].boxes.emplace_back(std::array<float, 4>{
          x1 + offset, y1 + offset, x2 + offset, y2 + offset});
      (*results)[bs].label_ids.push_back(best_cls);
      (*results)[bs].scores.push_back(max_score);
    }

    if ((*results)[bs].boxes.empty()) continue;

    utils::NMS(&((*results)[bs]), nms_threshold_);

    // ----- 还原到原始图像坐标（去掉 NMS 偏移 + LetterBox 逆变换） -----
    auto iter_out = ims_info[bs].find("output_shape");
    auto iter_ipt = ims_info[bs].find("input_shape");
    FDASSERT(iter_out != ims_info[bs].end() && iter_ipt != ims_info[bs].end(),
             "Cannot find input_shape or output_shape from im_info.");
    float out_h = iter_out->second[0];
    float out_w = iter_out->second[1];
    float ipt_h = iter_ipt->second[0];
    float ipt_w = iter_ipt->second[1];
    float scale = std::min(out_h / ipt_h, out_w / ipt_w);
    float pad_h = (out_h - ipt_h * scale) / 2.f;
    float pad_w = (out_w - ipt_w * scale) / 2.f;

    for (size_t i = 0; i < (*results)[bs].boxes.size(); ++i) {
      int32_t label_id = (*results)[bs].label_ids[i];
      float offset = multi_label_ ? label_id * max_wh_ : 0.f;

      float& bx1 = (*results)[bs].boxes[i][0];
      float& by1 = (*results)[bs].boxes[i][1];
      float& bx2 = (*results)[bs].boxes[i][2];
      float& by2 = (*results)[bs].boxes[i][3];

      // 去掉 NMS 偏移
      bx1 -= offset; by1 -= offset;
      bx2 -= offset; by2 -= offset;

      // LetterBox 逆变换
      bx1 = std::max((bx1 - pad_w) / scale, 0.f);
      by1 = std::max((by1 - pad_h) / scale, 0.f);
      bx2 = std::max((bx2 - pad_w) / scale, 0.f);
      by2 = std::max((by2 - pad_h) / scale, 0.f);

      // 边界裁剪
      bx1 = std::min(bx1, ipt_w);
      by1 = std::min(by1, ipt_h);
      bx2 = std::min(bx2, ipt_w);
      by2 = std::min(by2, ipt_h);
    }
  }

  return true;
}

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
