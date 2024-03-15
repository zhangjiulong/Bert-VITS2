ps aux | grep train_ms | grep -v ps | xargs -i kill -9 {}
torchrun --nproc_per_node=1 train_ms.py