ps aux | grep webui.py | grep -v grep | awk '{print $2}' | xargs -i kill -9 {}
# unset http_proxy
# unset https_proxy
nohup python webui.py &