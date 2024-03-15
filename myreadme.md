
# 容器启动
docker run -dit --name=tts --gpus all --runtime=nvidia -v /home/zhangjl19/:/workspace tts:v1 /bin/bash
docker run -e "http_proxy=http://192.168.200.26:51837" -e "https_proxy=http://192.168.200.26:51837" -dit --name=tts --gpus all --runtime=nvidia --shm-size 24g -v /home/zhangjl19/:/workspace tts:v1 /bin/bash


# 代理设置
export http_proxy=http://127.0.0.1:51837
export https_proxy=http://127.0.0.1:51837

# 验证代理
curl https://www.google.com.hk

# 杀掉进程
ps aux | grep google | awk '{print $2'} | xargs -i kill -9 {}

# 安装工具
apt install iputils

# 运行preprocessing 需要指定环境变量
export MKL_SERVICE_FORCE_INTEL=1
