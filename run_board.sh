ps aux | grep tensorboard | grep -v grep | awk '{print $2}' | xargs -i kill -9 {}
unset http_proxy
unset https_proxy
tensorboard --logdir=./data/huahua/models/