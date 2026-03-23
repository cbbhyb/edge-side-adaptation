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

// ============================================================
//  YOLOWorld 后处理
//
//  输出布局：CHW [B, 24, 8400]（已由 npy 分析确认）
//    通道 0~3 : cx, cy, w, h（中心坐标+宽高，640x640空间，需转xyxy）
//    通道 4~23: 20 类别 logits（clsNeedSigmoid 由 yaml 控制）
//
// 
// ============================================================

#include "postprocessor_yolov8.h"
#include "fastdeploy/vision/utils/utils.h"
#include "fastdeploy/function/concat.h"
#include "fastdeploy/function/elementwise.h"
#include "fastdeploy/function/split.h"

#include <cmath>
#include <algorithm>
#include <cstdio>

namespace fastdeploy {
namespace vision {
namespace detection {

YOLOv8Postprocessor::YOLOv8Postprocessor() {
  conf_threshold_   = 0.25f;
  nms_threshold_    = 0.5f;
  multi_label_      = true;
  max_wh_           = 7680.0f;
  num_classes_      = 20;
  cls_need_sigmoid_ = true;
  debug_print_      = true;
}

YOLOv8Postprocessor::YOLOv8Postprocessor(float conf_threshold,
                                          float nms_threshold,
                                          bool  multi_label,
                                          float max_wh)
    : conf_threshold_(conf_threshold),
      nms_threshold_(nms_threshold),
      multi_label_(multi_label),
      max_wh_(max_wh),
      num_classes_(20),
      cls_need_sigmoid_(true),
      debug_print_(true) {}

bool YOLOv8Postprocessor::Run(
    const std::vector<FDTensor>& tensors,
    std::vector<DetectionResult>* results,
    const std::vector<std::map<std::string, std::array<float, 2>>>& ims_info) {

  if (tensors.empty()) {
    fprintf(stderr, "[PostProc] ERROR: no output tensors.\n");
    return false;
  }

  const FDTensor& t = tensors[0];
  if (t.dtype != FDDataType::FP32) {
    fprintf(stderr, "[PostProc] ERROR: only FP32 output supported.\n");
    return false;
  }

  // ---- 解析 shape，固定 CHW [B, 24, 8400] ----
  const int expected_ch   = 4 + num_classes_;  // 24
  const int expected_anch = 8400;

  int batch       = static_cast<int>(t.shape[0]);
  int dim1        = static_cast<int>(t.shape[1]);
  int dim2        = static_cast<int>(t.shape[2]);
  int num_anchors = dim2;
  int total_ch    = dim1;

  if (dim1 != expected_ch || dim2 != expected_anch) {
    fprintf(stderr,
      "[PostProc] WARN: unexpected shape [%d, %d, %d], expected [B, %d, %d].\n",
      batch, dim1, dim2, expected_ch, expected_anch);
  }

  if (debug_print_) {
    fprintf(stderr,
      "[PostProc] Shape=[%d,%d,%d]  Layout=CHW [B,C,N]  num_anchors=%d  total_ch=%d\n"
      "           num_classes=%d  clsNeedSigmoid=%s  conf_thresh=%.3f\n",
      batch, dim1, dim2,
      num_anchors, total_ch,
      num_classes_, cls_need_sigmoid_ ? "true" : "false",
      conf_threshold_);
  }

  const float* data = reinterpret_cast<const float*>(t.Data());
  results->resize(batch);

  for (int bs = 0; bs < batch; ++bs) {
    (*results)[bs].Clear();
    (*results)[bs].Reserve(num_anchors);

    // CHW: data[ch * num_anchors + anchor_idx]
    const float* bd = data + bs * num_anchors * total_ch;

    // ===== DEBUG: 打印前 5 个 anchor 的原始值 =====
    if (debug_print_) {
      fprintf(stderr, "[PostProc] --- batch=%d: first 5 anchors (raw) ---\n", bs);
      int print_n = std::min(5, num_anchors);
      for (int i = 0; i < print_n; ++i) {
        float cx = bd[0 * num_anchors + i];
        float cy = bd[1 * num_anchors + i];
        float w  = bd[2 * num_anchors + i];
        float h  = bd[3 * num_anchors + i];
        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;
        float max_logit = -1e9f;
        int   max_cls   = -1;
        for (int c = 0; c < num_classes_; ++c) {
          float v = bd[(4 + c) * num_anchors + i];
          if (v > max_logit) { max_logit = v; max_cls = c; }
        }
        float max_score = cls_need_sigmoid_
            ? (1.f / (1.f + expf(-max_logit)))
            : max_logit;
        fprintf(stderr,
          "  anchor[%d] cxcywh=(%.1f,%.1f,%.1f,%.1f) xyxy=(%.1f,%.1f,%.1f,%.1f) "
          "max_cls=%d raw_logit=%.4f score=%.4f\n",
          i, cx, cy, w, h, x1, y1, x2, y2,
          max_cls, max_logit, max_score);
      }
    }
    // ===== END DEBUG =====

    // ---- 主循环：候选框生成 ----
    // bbox 通道 0~3 是 cx, cy, w, h，需转换为 xyxy
    for (int i = 0; i < num_anchors; ++i) {
      float cx = bd[0 * num_anchors + i];
      float cy = bd[1 * num_anchors + i];
      float w  = bd[2 * num_anchors + i];
      float h  = bd[3 * num_anchors + i];
      float x1 = cx - w * 0.5f;
      float y1 = cy - h * 0.5f;
      float x2 = cx + w * 0.5f;
      float y2 = cy + h * 0.5f;

      if (multi_label_) {
        for (int c = 0; c < num_classes_; ++c) {
          float logit = bd[(4 + c) * num_anchors + i];
          float score = cls_need_sigmoid_
              ? (1.f / (1.f + expf(-logit))) : logit;
          if (score <= conf_threshold_) continue;

          float offset = c * max_wh_;
          (*results)[bs].boxes.emplace_back(std::array<float, 4>{
              x1 + offset, y1 + offset, x2 + offset, y2 + offset});
          (*results)[bs].label_ids.push_back(c);
          (*results)[bs].scores.push_back(score);
        }
      } else {
        float max_score = -1.f;
        int   best_cls  = -1;
        for (int c = 0; c < num_classes_; ++c) {
          float logit = bd[(4 + c) * num_anchors + i];
          float score = cls_need_sigmoid_
              ? (1.f / (1.f + expf(-logit))) : logit;
          if (score > max_score) { max_score = score; best_cls = c; }
        }
        if (max_score <= conf_threshold_) continue;

        (*results)[bs].boxes.emplace_back(std::array<float, 4>{x1,y1,x2,y2});
        (*results)[bs].label_ids.push_back(best_cls);
        (*results)[bs].scores.push_back(max_score);
      }
    }

    if (debug_print_) {
      fprintf(stderr,
        "[PostProc] batch=%d: pre-NMS candidates=%zu\n",
        bs, (*results)[bs].boxes.size());
    }

    if ((*results)[bs].boxes.empty()) continue;

    utils::NMS(&((*results)[bs]), nms_threshold_);

    // ---- LetterBox 逆变换 ----
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
      float   offset   = multi_label_ ? label_id * max_wh_ : 0.f;

      float& bx1 = (*results)[bs].boxes[i][0];
      float& by1 = (*results)[bs].boxes[i][1];
      float& bx2 = (*results)[bs].boxes[i][2];
      float& by2 = (*results)[bs].boxes[i][3];

      bx1 -= offset; by1 -= offset;
      bx2 -= offset; by2 -= offset;

      bx1 = std::max((bx1 - pad_w) / scale, 0.f);
      by1 = std::max((by1 - pad_h) / scale, 0.f);
      bx2 = std::max((bx2 - pad_w) / scale, 0.f);
      by2 = std::max((by2 - pad_h) / scale, 0.f);

      bx1 = std::min(bx1, ipt_w);
      by1 = std::min(by1, ipt_h);
      bx2 = std::min(bx2, ipt_w);
      by2 = std::min(by2, ipt_h);
    }

    if (debug_print_) {
      fprintf(stderr,
        "[PostProc] batch=%d: post-NMS boxes=%zu  (scale=%.4f pad_h=%.1f pad_w=%.1f)\n",
        bs, (*results)[bs].boxes.size(), scale, pad_h, pad_w);
      int print_n = std::min((int)(*results)[bs].boxes.size(), 8);
      for (int i = 0; i < print_n; ++i) {
        auto& b = (*results)[bs].boxes[i];
        fprintf(stderr,
          "  box[%d] xyxy=(%.1f,%.1f,%.1f,%.1f) cls=%d score=%.4f\n",
          i, b[0], b[1], b[2], b[3],
          (*results)[bs].label_ids[i],
          (*results)[bs].scores[i]);
      }
    }
  }

  return true;
}

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
