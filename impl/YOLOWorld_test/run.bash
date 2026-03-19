#!/usr/bin/bash
###############环境变量配置###################
export LD_LIBRARY_PATH=\
$PWD/build/:\
$PWD/build/3rdparty/fastdeploy/third_party/yaml-cpp/:\
$PWD/build/3rdparty/fastdeploy/third_libs/install/onnxruntime/lib/:\
$PWD/build/3rdparty/fastdeploy/third_libs/install/paddle2onnx/lib/:\
$PWD/build/3rdparty/fastdeploy:\
/usr/local/cisdi/opencv-4.3.0/lib:\
/usr/local/cisdi/libhv/:\
/usr/local/cisdi/libhv-1.3.3/lib/:\
$LD_LIBRARY_PATH
###############环境变量配置###################


###############以下脚本不需更改###################
# 获取环境变量
source ./.env

IP=$DEVICE_IP
PORT=$DEVICE_GDB_SERVER_PORT
PROJECT_NAME=$(basename $PROJECT_ROOT)

echo "Hint: bash run.bash gdb | gdbsvr | valgrind."
if [[ "$1" == "gdbsvr" ]];then
  echo "entering gdb server mode, listen port: $IP:$PORT"
  sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH gdbserver $IP:$PORT ./build/demo $MODEL_DIR $INPUT
elif [ "$1" == "valgrind" ]; then
  valgrind --leak-check=full ./build/demo
elif [ "$1" == "gdb" ]; then
  sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH gdb -ex "set substitute-path $PROJECT_ROOT $DEVICE_DEPLOY_DIR/$PROJECT_NAME" ./build/demo
else
  ./build/demo
fi
###############以下脚本不需更改###################
