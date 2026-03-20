#include <thread>
#include <vector>
#include <functional>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <pwd.h>
#include "infer_impl.hpp"
#include "algo/yoloworld.h"
#include "common/utils/one_logger.hpp"
#include "common/utils/time_stamp.hpp"
#include "common/utils/utils.hpp"

using namespace std;
using json = nlohmann::json;

namespace Model{

    void InferImpl::worker(promise<bool>& pro){
        fastdeploy::RuntimeOption option;
        fastdeploy::ModelFormat  model_format;
        initModel(option, model_format);

        // 单模型加载：YOLOWorld
        auto model = fastdeploy::vision::detection::YOLOWorld(
            det_model_dir_ + "/" + config_.get<string>("algo.modelName", ""),
            "", option, model_format, nullptr, nullptr);

        model.GetPostprocessor().SetConfThreshold(config_.get<float>("algo.confidence", 0.5));
        model.GetPostprocessor().SetNMSThreshold(config_.get<float>("algo.nms", 0.45));
        model.GetPostprocessor().SetNumClasses(config_.get<int>("algo.numClasses", 20));
        model.GetPostprocessor().SetMultiLabel(config_.get<bool>("algo.multiLabel", false));
        model.GetPostprocessor().SetClsNeedSigmoid(config_.get<bool>("algo.clsNeedSigmoid", true));

        assert(model.Initialized());

        pro.set_value(true);
        vector<Job> fetched_jobs;
        while(running_){
            {
                unique_lock<mutex> l(lock_);
                cv_.wait(l, [&](){
                    return !running_ || !jobs_.empty();
                });

                if(!running_) break;

                for(int i = 0; i < 1 && !jobs_.empty(); ++i){
                    fetched_jobs.emplace_back(std::move(jobs_.front()));
                    jobs_.pop();
                }
            }

            for(int ibatch = 0; ibatch < fetched_jobs.size(); ++ibatch) {
                auto& job = fetched_jobs[ibatch];
                auto& det_image = job.input;
                auto ts = std::make_shared<TimeStamp>();

                fastdeploy::vision::DetectionResult res;

                if (!model.Predict(det_image, &res)){
                    log_warn("Failed to predict.");
                    return;
                }

                log_info("YOLOWorld detected {} boxes.", res.boxes.size());

                DetResultArray box_det_array;
                algo::DetResultProperty box_det;

                for (size_t i = 0; i < res.boxes.size(); ++i) {
                    box_det.left   = res.boxes[i][0];
                    box_det.top    = res.boxes[i][1];
                    box_det.width  = res.boxes[i][2] - box_det.left;
                    box_det.height = res.boxes[i][3] - box_det.top;
                    box_det.classId = res.label_ids[i];
                    box_det.detectionConfidence = res.scores[i];
                    box_det_array.emplace_back(box_det);
                }

                ts->show_summary("Inference");
                job.pro->set_value(box_det_array);
            }
            fetched_jobs.clear();
        }

        unique_lock<mutex> l(lock_);
        while(!jobs_.empty()){
            jobs_.back().pro->set_value({});
            jobs_.pop();
        }
    }

    shared_ptr<Infer> create_infer(const string& file){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(file)){
            instance.reset();
        }
        return instance;
    }
};
