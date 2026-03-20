#include <string>
#include <algorithm>
#include "nlohmann/json.hpp"
#include "common/utils/one_logger.hpp"
#include "common/infer_process/detect/logic_post_process.hpp"
#include "common/utils/JsonBuilder.hpp"
#include "common/utils/safe_queue.hpp"
#include "common/utils/BusinessData.hpp"

using namespace nlohmann;

class LogicPostProcessImpl : public LogicPostProcessBase
{
public:
    LogicPostProcessImpl() : cacheQueue_(SafeQueue<algo::DetResWithMask *>(32)) {}
    bool logicPostProcess(std::vector<std::shared_ptr<LogicInputParam>> &frameResults, LogicRetFromAlgoSP &postResult) override;

private:
    int count_ = 0;
    SafeQueue<algo::DetResWithMask *> cacheQueue_;
};

bool LogicPostProcessImpl::logicPostProcess(std::vector<std::shared_ptr<LogicInputParam>> &frameResults, LogicRetFromAlgoSP &postResult)
{
    for (auto &result : frameResults)
    {
        std::shared_ptr<DetPropertyLogicInputParam> detFrameResult = std::dynamic_pointer_cast<DetPropertyLogicInputParam>(result);
        auto &detResPerFrame = detFrameResult->det_result.DetResList;
        width_ = detFrameResult->det_result.frame_img.cols;
        height_ = detFrameResult->det_result.frame_img.rows;
        const auto &labelMap = getLabelMap(detFrameResult->det_result.algoParams);

        std::vector<algo::DetResultProperty>& filteredResults = filterByRoi(detFrameResult->det_result.DetResList);
        nlohmann::json rps;
        for (auto &ele : filteredResults)
        {
            nlohmann::json rp;
            rp["location"] = {{ele.left, ele.top}, {ele.left + ele.width, ele.top + ele.height}};
            rp["shapeType"] = "RECTANGLE";
            rp["name"] = labelMap.at(std::to_string(ele.classId));
            rp["text"] = labelMap.at(std::to_string(ele.classId)) + ": " + std::to_string(ele.detectionConfidence);
            rp["color"] = {0, 165, 255};  // 橙色 (BGR)
            rp["thickness"] = 3;
            rps.push_back(rp);
        }

        isEvent_ = filteredResults.empty() ? false : true;

        std::shared_ptr<DetLogicRetFromAlgo> detLogicResult = std::dynamic_pointer_cast<DetLogicRetFromAlgo>(postResult);
        if (detLogicResult)
        {
            handleResultData(detLogicResult->det.detResult, "SD_施工物品检测", isEvent_, rps);
        }
    }
    return true;
}

extern "C"
{
    LogicPostProcessBase *create(CV_TASK_TYPE type)
    {
        if (type == CV_TASK_TYPE::DETECTION_PROPERTY)
        {
            spdlog::info("create det task logic processor ");
            return new LogicPostProcessImpl;
        }
        else
        {
            spdlog::error("unsupport task type ");
            return new LogicPostProcessBase;
        }
    }

    void destroy(LogicPostProcessBase *p)
    {
        if (p)
            delete p;
    }
}
