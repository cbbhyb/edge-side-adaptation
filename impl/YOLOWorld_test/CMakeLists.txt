cmake_minimum_required(VERSION 3.10)
# 最好和模型名字保持一致, 这个名字会用来定义推理库的名称
project(algo_YOLOWorld_test)

# 项目根目录
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../)

# 统一编译选项配置
include(${ROOT_DIR}/cmake/compile_options.cmake)

# 算法编译
include(${ROOT_DIR}/cmake/algo.cmake)

# 应用编译
include(${ROOT_DIR}/cmake/app.cmake)

# 所有依赖安装到build/package下
include(${ROOT_DIR}/cmake/install_rule.cmake)
