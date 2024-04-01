
# 容器启动
docker run -dit --name=tts --gpus all --runtime=nvidia -v /home/zhangjl19/:/workspace tts:v1 /bin/bash
docker run -e "http_proxy=http://192.168.200.26:51837" -e "https_proxy=http://192.168.200.26:51837" -dit --name=tts --gpus all --runtime=nvidia --shm-size 24g -v /home/zhangjl19/:/workspace tts:v1 /bin/bash


# 代理设置
export http_proxy=http://127.0.0.1:51837
export https_proxy=http://127.0.0.1:51837

# 取消设置代理
unset https_proxy
unset http_proxy


# 验证代理
curl https://www.google.com.hk

# 杀掉进程
ps aux | grep google | awk '{print $2'} | xargs -i kill -9 {}

# 安装工具
apt install iputils
apt install sox

# wget ssl connection error
update-ca-certificates --fresh

# 运行preprocessing 需要指定环境变量
export MKL_SERVICE_FORCE_INTEL=1

# 文件转换
ffmpeg -i 1_B.mp3 -acodec pcm_s16le -ar 16000 -ac 2 output.wav

# 参考资料
> https://zhuanlan.zhihu.com/p/680339733

# VIST
> https://zhuanlan.zhihu.com/p/571040094
> https://www.zywvvd.com/notes/study/deep-learning/tts/vits/vits/
> https://www.bilibili.com/video/BV1Nk4y1A7SL/?vd_source=64f63f34985a708ab738d22e9d0dd177 视频介绍

## GAN(Generative Adversial Networks)
> https://zhuanlan.zhihu.com/p/408766083

## VAE(Variational Auto-encoder)
1. 后验分布即目标音频
> https://zhuanlan.zhihu.com/p/620113235
> https://www.zhangzhenhu.com/aigc/%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8.html(推导)
> https://www.shenxiaohai.me/pytorch-tutorial-advanced-02/

## NICE
> https://kexue.fm/archives/5776

## Alignment 
> https://www.cnblogs.com/Edison-zzc/p/17589837.html

## FLOW
变量变换定理
预测声音时长，增强真实感。
> https://zhuanlan.zhihu.com/p/142567194
> https://spaces.ac.cn/archives/5776

## 时长估计
> https://zhuanlan.zhihu.com/p/571040094 公式推导

# VIST2
Duration Prediction：VITS中使用的是FLOW++一个很复杂的结构，耗时长，为了平衡耗时和性能，VITS2将其改变为一个基于GAN的结构。
Alignment Search：VITS中使用动态规划算法来强制对齐，这样得到的结构太固定化了。VITS2中在MAS中引入noise，增加多样性，为模型提供了额外的机会来搜索其他对齐。
Speaker-conditioned：在text-encoding中加入speaker-embedding作为condition。
Normalizing Flow with the transformer block：在Flow的结构中加入transformer（原本只有CNN），使得模型能获得长距离的依赖，交流不同位置的特征信息。

# 去噪声
https://github.com/Anjok07/ultimatevocalremovergui

# 其他仓库
https://github.com/v3ucn/Bert-VITS2_V210 支持 中英混合
https://github.com/v3ucn/Bert-vits2-V2.3  https://v3u.cn/a_id_341

# whisper
https://github.com/shuaijiang/Whisper-Finetune whisper 数据微调 


# tts镜像版本管理
1. 2.0 迁移版本
