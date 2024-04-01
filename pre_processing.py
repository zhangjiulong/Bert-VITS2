import os
import json
import subprocess
import shutil
import argparse


def get_path(data_dir):
    start_path = os.path.join("./data", data_dir)
    lbl_path = os.path.join(start_path, "esd.list")
    train_path = os.path.join(start_path, "train.list")
    val_path = os.path.join(start_path, "val.list")
    config_path = os.path.join(start_path, "configs", "config.json")
    return start_path, lbl_path, train_path, val_path, config_path

def generate_config(data_dir, batch_size):
    assert data_dir != "", "数据集名称不能为空"
    start_path, _, train_path, val_path, config_path = get_path(data_dir)
    if os.path.isfile(config_path):
        config = json.load(open(config_path, "r", encoding="utf-8"))
    else:
        config = json.load(open("configs/config.json", "r", encoding="utf-8"))
    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["train"]["batch_size"] = batch_size
    out_path = os.path.join(start_path, "configs")
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    model_path = os.path.join(start_path, "models")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    if not os.path.exists("config.yml"):
        shutil.copy(src="default_config.yml", dst="config.yml")
    return "配置文件生成完成"

def resample(data_dir):
    assert data_dir != "", "数据集名称不能为空"
    start_path, _, _, _, config_path = get_path(data_dir)
    in_dir = os.path.join(start_path, "raw")
    out_dir = os.path.join(start_path, "wavs")
    subprocess.run(
        f"python resample_legacy.py "
        f"--sr 44100 "
        f"--in_dir {in_dir} "
        f"--out_dir {out_dir} ",
        shell=True,
    )
    return "音频文件预处理完成"

def preprocess_text(data_dir):
    assert data_dir != "", "数据集名称不能为空"
    start_path, lbl_path, train_path, val_path, config_path = get_path(data_dir)
    lines = open(lbl_path, "r", encoding="utf-8").readlines()
    with open(lbl_path, "w", encoding="utf-8") as f:
        for line in lines:
            path, spk, language, text = line.strip().split("|")
            path = os.path.join(start_path, "wavs", os.path.basename(path)).replace(
                "\\", "/"
            )
            f.writelines(f"{path}|{spk}|{language}|{text}\n")
    subprocess.run(
        f"python preprocess_text.py "
        f"--transcription-path {lbl_path} "
        f"--train-path {train_path} "
        f"--val-path {val_path} "
        f"--config-path {config_path}",
        shell=True,
    )
    return "标签文件预处理完成"

def bert_gen(data_dir):
    assert data_dir != "", "数据集名称不能为空"
    _, _, _, _, config_path = get_path(data_dir)
    subprocess.run(
        f"python bert_gen.py " f"--config {config_path}",
        shell=True,
    )
    return "BERT 特征文件生成完成"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="processing corpus for tts")
    parser.add_argument("--data", type=str, help="请指定 --data ")
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
    args = parser.parse_args()
    
    
    data_dir = args.data
    batch_size = args.batch_size

    print(f'generating config infos data dir {data_dir} batch size {batch_size}')
    generate_config(data_dir, batch_size)

    print(f'resampling datas {data_dir}')
    resample(data_dir)

    print(f'processing txt {data_dir}')
    preprocess_text(data_dir)

    print(f'generating bert features {data_dir}')
    bert_gen(data_dir)