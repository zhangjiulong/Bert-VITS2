ps aux | grep tensorboard | grep -v grep | awk '{print $2}' | xargs -i kill -9 {}
tensorboard --logdir=./models/