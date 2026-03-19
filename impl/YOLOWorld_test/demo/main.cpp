#include <iostream>
#include "common/utils/one_logger.hpp"
#include "common/manage/manager.hpp"

int main(int argc, char **argv){
    log_info("Starting application.");

    // 创建 Manager 实例并启动推理和后处理流程
    log_info("Creating Manager.");
    Manager manager("./demo/default_config.yaml");
    manager.run();

    log_info("Application finished.");

    return 0;
}
