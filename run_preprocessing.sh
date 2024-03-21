export MKL_SERVICE_FORCE_INTEL=1
ps aux | grep webui.py | grep -v grep | awk '{print $2}' | xargs -i kill -9 {}
unset http_proxy
unset https_proxy
python webui_preprocess.py