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
//  【调试版后处理】YOLOWorld 自适应 layout
//
//  支持两种输出布局，由 shape 自动判断：
//    CHW模式: [B, 24, 8400]  — 原始假设
//    HWC模式: [B, 8400, 24]  — 另一种可能
//
//  每次推理会打印：
//    1. 张量 shape 及检测到的布局
//    2. 前 5 个 anchor 的原始 bbox 值 + 最大类别分数（sigmoid前后）
//    3. NMS 后框的数量及前几个框的坐标
//
//  确认格式正确后，可删除 DEBUG_PRINT 段落（或关掉 debug_print_ 开关）
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
  debug_print_      = true;   // ← 调试开关：确认无误后改 false
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

// ----------------------------------------------------------------
//  辅助：从 data 读取一个值，两种 layout 封装成统一接口
//    CHW: data[ch * num_anchors + idx]
//    HWC: data[idx * total_ch   + ch ]
// ----------------------------------------------------------------
static inline float readVal(const float* data, int ch, int idx,
                             int num_anchors, int total_ch, bool is_chw) {
  return is_chw ? data[ch * num_anchors + idx]
                : data[idx * total_ch   + ch ];
}

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

  // ---- 解析 shape，自动判断布局 ----
  //   期望：[B, C, N] (CHW)  或  [B, N, C] (HWC)
  //   其中 C = 4 + num_classes_,  N = 8400
  const int expected_ch    = 4 + num_classes_;   // e.g. 24
  const int expected_anch  = 8400;

  int batch       = static_cast<int>(t.shape[0]);
  int dim1        = static_cast<int>(t.shape[1]);
  int dim2        = static_cast<int>(t.shape[2]);

  bool is_chw = false;   // CHW: [B, 24, 8400]
  bool is_hwc = false;   // HWC: [B, 8400, 24]
  int  num_anchors = expected_anch;
  int  total_ch    = expected_ch;

  if (dim1 == expected_ch && dim2 == expected_anch) {
    is_chw = true;
    num_anchors = dim2;
    total_ch    = dim1;
  } else if (dim1 == expected_anch && dim2 == expected_ch) {
    is_hwc = true;
    num_anchors = dim1;
    total_ch    = dim2;
  } else {
    // shape 不符合预期，但继续尝试用 dim 推断
    fprintf(stderr,
      "[PostProc] WARN: unexpected shape [%d, %d, %d], "
      "expected [B, %d, %d] or [B, %d, %d]. Will try CHW.\n",
      batch, dim1, dim2,
      expected_ch, expected_anch,
      expected_anch, expected_ch);
    // 强制按 CHW 尝试
    is_chw      = true;
    num_anchors = dim2;
    total_ch    = dim1;
  }

  if (debug_print_) {
    fprintf(stderr,
      "[PostProc] Shape=[%d,%d,%d]  Layout=%s  num_anchors=%d  total_ch=%d\n"
      "           num_classes=%d  clsNeedSigmoid=%s  conf_thresh=%.3f\n",
      batch, dim1, dim2,
      is_chw ? "CHW [B,C,N]" : "HWC [B,N,C]",
      num_anchors, total_ch,
      num_classes_, cls_need_sigmoid_ ? "true" : "false",
      conf_threshold_);
  }

  // ---- 生成 anchor_points（固定 80x80 / 40x40 / 20x20）----
  struct AnchorPt { float x, y, stride; };
  std::vector<AnchorPt> anchor_pts;
  anchor_pts.reserve(8400);
  const int   feat_hw[3][2] = {{80,80},{40,40},{20,20}};
  const float strides[3]    = {8.f, 16.f, 32.f};
  for (int s = 0; s < 3; ++s) {
    int h = feat_hw[s][0], w = feat_hw[s][1];
    for (int i = 0; i < h; ++i)
      for (int j = 0; j < w; ++j)
        anchor_pts.push_back({j + 0.5f, i + 0.5f, strides[s]});
  }

  const float* data = reinterpret_cast<const float*>(t.Data());
  results->resize(batch);

  for (int bs = 0; bs < batch; ++bs) {
    (*results)[bs].Clear();
    (*results)[bs].Reserve(num_anchors);

    // 当前 batch 的起始偏移
    const float* bd = data + bs * num_anchors * total_ch;

    // ===== DEBUG: 打印前 5 个 anchor 的原始值 =====
    if (debug_print_) {
      fprintf(stderr, "[PostProc] --- batch=%d: first 5 anchors (raw) ---\n", bs);
      int print_n = std::min(5, num_anchors);
      for (int i = 0; i < print_n; ++i) {
        float lt_x = readVal(bd, 0, i, num_anchors, total_ch, is_chw);
        float lt_y = readVal(bd, 1, i, num_anchors, total_ch, is_chw);
        float rb_x = readVal(bd, 2, i, num_anchors, total_ch, is_chw);
        float rb_y = readVal(bd, 3, i, num_anchors, total_ch, is_chw);
        // 最大类别 logit
        float max_logit = -1e9f;
        int   max_cls   = -1;
        for (int c = 0; c < num_classes_; ++c) {
          float v = readVal(bd, 4 + c, i, num_anchors, total_ch, is_chw);
          if (v > max_logit) { max_logit = v; max_cls = c; }
        }
        float max_score = cls_need_sigmoid_
            ? (1.f / (1.f + expf(-max_logit)))
            : max_logit;
        // 解码成像素坐标（供对比参考）
        float ax = anchor_pts[i].x, ay = anchor_pts[i].y;
        float st = anchor_pts[i].stride;
        float px1 = (ax - lt_x) * st;
        float py1 = (ay - lt_y) * st;
        float px2 = (ax + rb_x) * st;
        float py2 = (ay + rb_y) * st;
        fprintf(stderr,
          "  anchor[%d] raw_bbox=(%.3f,%.3f,%.3f,%.3f) "
          "decoded_xyxy=(%.1f,%.1f,%.1f,%.1f) "
          "max_cls=%d raw_logit=%.4f score=%.4f\n",
          i, lt_x, lt_y, rb_x, rb_y,
          px1, py1, px2, py2,
          max_cls, max_logit, max_score);
      }

      // 同样用 HWC 视角打印前 5 个，帮助对比哪种更合理
      if (is_chw) {
        fprintf(stderr,
          "[PostProc]   (为对比：如果 layout 其实是 HWC，则相同 5 个 anchor 原始值为:)\n");
        for (int i = 0; i < print_n; ++i) {
          float lt_x = readVal(bd, 0, i, num_anchors, total_ch, false/*HWC*/);
          float lt_y = readVal(bd, 1, i, num_anchors, total_ch, false);
          float rb_x = readVal(bd, 2, i, num_anchors, total_ch, false);
          float rb_y = readVal(bd, 3, i, num_anchors, total_ch, false);
          float max_logit = -1e9f; int max_cls = -1;
          for (int c = 0; c < num_classes_; ++c) {
            float v = readVal(bd, 4+c, i, num_anchors, total_ch, false);
            if (v > max_logit) { max_logit = v; max_cls = c; }
          }
          float max_score = cls_need_sigmoid_
              ? (1.f / (1.f + expf(-max_logit))) : max_logit;
          float ax = anchor_pts[i].x, ay = anchor_pts[i].y;
          float st = anchor_pts[i].stride;
          fprintf(stderr,
            "  anchor[%d] HWC raw_bbox=(%.3f,%.3f,%.3f,%.3f) "
            "decoded_xyxy=(%.1f,%.1f,%.1f,%.1f) "
            "max_cls=%d raw_logit=%.4f score=%.4f\n",
            i, lt_x, lt_y, rb_x, rb_y,
            (ax-lt_x)*st, (ay-lt_y)*st, (ax+rb_x)*st, (ay+rb_y)*st,
            max_cls, max_logit, max_score);
        }
      }
    }
    // ===== END DEBUG =====

    // ---- 主循环：候选框生成 ----
    for (int i = 0; i < num_anchors; ++i) {
      float lt_x = readVal(bd, 0, i, num_anchors, total_ch, is_chw);
      float lt_y = readVal(bd, 1, i, num_anchors, total_ch, is_chw);
      float rb_x = readVal(bd, 2, i, num_anchors, total_ch, is_chw);
      float rb_y = readVal(bd, 3, i, num_anchors, total_ch, is_chw);

      float ax = anchor_pts[i].x;
      float ay = anchor_pts[i].y;
      float st = anchor_pts[i].stride;

      float x1 = (ax - lt_x) * st;
      float y1 = (ay - lt_y) * st;
      float x2 = (ax + rb_x) * st;
      float y2 = (ay + rb_y) * st;

      if (multi_label_) {
        for (int c = 0; c < num_classes_; ++c) {
          float logit = readVal(bd, 4+c, i, num_anchors, total_ch, is_chw);
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
          float logit = readVal(bd, 4+c, i, num_anchors, total_ch, is_chw);
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

    // ===== DEBUG: 打印 NMS 后结果 =====
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
    // ===== END DEBUG =====
  }

  return true;
}

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
