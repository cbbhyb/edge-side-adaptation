//
// Created by CISDI on 2024/9/27.
//
#include <mutex>
#include <condition_variable>
#include <queue>
#include "fastdeploy/vision.h"
#include "common/infer_process/detect/infer.hpp"


#ifndef DEMO_IINFER_IMPL_HPP
#define DEMO_IINFER_IMPL_HPP

using namespace std;

namespace Model{
    class InferImpl : public Infer{
    public:
        void worker(promise<bool>& pro) override;
    private:
        void initModel(fastdeploy::RuntimeOption& option, fastdeploy::ModelFormat& model_format){
            auto platform = config_.get<string>("platform.name", "cpu");
            if (platform == "ascend"){
                option.UseAscendBackend();
                model_format = fastdeploy::ModelFormat::ASCEND;
            } else if (platform == "sophon"){
                option.UseSophgo();
                model_format = fastdeploy::ModelFormat::SOPHGO;
            } else {
                option.UseGpu();
                option.UseTrtBackend();
                option.trt_option.SetShape("input", {1, 3, 640, 640}, {1, 3, 640, 640}, {1, 3, 640, 640});
                option.trt_option.SetSerializeFile(det_model_dir_ + "/" + config_.get<std::string>("algo.modelName", "") + ".trt");
                model_format = fastdeploy::ModelFormat::ONNX;
            }
        }
    };

    shared_ptr<Infer> create_infer(const string& file);
};



#endif //DEMO_IINFER_IMPL_HPP
