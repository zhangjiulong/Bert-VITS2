ps aux | grep webui.py | grep -v grep | awk '{print $2}' | xargs -i kill -9 {}
nohup python webui.py &