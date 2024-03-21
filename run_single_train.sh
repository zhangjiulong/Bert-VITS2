ps aux | grep train_ms | grep -v grep | awk '{print $2}' | xargs -i kill -9 {}
ps aux | grep webui | grep -v grep | awk '{print $2}' | xargs -i kill -9 {}
unset http_proxy
unset https_proxy
nohup torchrun --nproc_per_node=1 train_ms.py &