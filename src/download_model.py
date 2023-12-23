import os
import argparse
import time
import json
import traceback

from huggingface_hub import hf_hub_download, snapshot_download
from utils import get_dir_size, add_size_str

from dotenv import load_dotenv
load_dotenv(".env")


model_root = os.getenv('HF_MODEL_ROOT')
if model_root is None: raise ValueError("HF_MODEL_ROOT is not set")

model_mirror = os.getenv('HF_MIRROR')
if model_mirror is None: raise ValueError("HF_MIRROR is not set")

model_config_template_path = './config/model_config_template.json'
model_todownload_config_path = './config/model_todownload.json'
model_downloaded_log_path = './log/model_downloaded.json'

## set config template dict
config_template = {}
with open(model_config_template_path, 'r', encoding='utf-8') as f: config_template = json.load(f)

    
def check_config(config: dict):
    if config.get('model_name', "") == "":
        raise ValueError("model_name is required in the config fields")
    if config.get('save_dir', "") == "":
        raise ValueError("save_dir is required in the config fields")

def downloaded_log(config, elapsed_time):
    # additional info to config
    config['download_size'] = get_dir_size(config['save_dir'], format=True)
    config['download_link'] = os.path.join(model_mirror, config['model_name'])
    config['download_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    config['download_elapsed_time'] = time.strftime('%H h: %M m: %S s', time.gmtime(elapsed_time))
    
    if os.path.exists(model_downloaded_log_path):
        with open(model_downloaded_log_path, 'r', encoding='utf-8') as f:
            downloaded_logs = json.load(f)
    else: downloaded_logs = {}
    
    if downloaded_logs == {}: # fist downloaded model, so init the empty downloaded logs
        downloaded_logs = {'model_cnt': 0, 'total_size': "0GB 0MB 0KB", 'models': []}
    
    downloaded_logs['model_cnt'] += 1
    downloaded_logs['total_size'] = add_size_str(downloaded_logs['total_size'], config['download_size'])
    downloaded_logs['models'] = [config] + downloaded_logs['models']
    
    with open(model_downloaded_log_path, 'w', encoding='utf-8') as f:
        json.dump(downloaded_logs, f)

def download(config_dict: dict, idx=-1):
    config = {**config_template, **config_dict}
    check_config(config)
    
    orderstr_dict = {-1: " ", 0: " 1st ", 1: " 2nd ", 2: " 3rd "}
    print("="*25, f" Downloading the{orderstr_dict.get(idx, ' '+str(idx)+'th ')}model: {config['model_name']} ", "="*25)
    
    # print(config); return # FIXME: debug
    
    start_time = time.time()
    try:
        config['save_dir'] = os.path.join(model_root, config['save_dir'], config['model_name'])
        if not os.path.exists(config['save_dir']): os.makedirs(config['save_dir'])
        
        # print(config); return # FIXME: debug

        if config['file_name'] != "": # to download specific files from the model repo
            hf_hub_download(
                repo_type='model',
                repo_id=config['model_name'], 
                filename=config['file_name'], 
                revision=config['version'] if config['version'] != "" else None,
                local_dir=config['save_dir'], 
                local_dir_use_symlinks=False # NOTE: directly download the files instead of symlinks to the cache for perminent storage
            )
        else: # to download the whole model repo
            snapshot_download(
                repo_type='model',
                repo_id=config['model_name'],
                revision=config['version'] if config['version'] not in ["", "latest"] else None,
                allow_patterns=config['allow_patterns'] if config['allow_patterns'] != [] else None,
                ignore_patterns=config['ignore_patterns'] if config['ignore_patterns'] != [] else None,
                local_dir=config['save_dir'], 
                local_dir_use_symlinks=False # NOTE: directly download the files instead of symlinks to the cache for perminent storage
            )
    except Exception as e:
        print("="*25, 
              f" Failed to Download: {config['model_name']} due to the thrown error\n: {e}\n with the trackback below: \n{traceback.format_exc()}\n", 
              "="*25)
        return False
    
    elapsed_time = time.time() - start_time
    downloaded_log(config, elapsed_time)
    print("="*25, f" Successfully Downloaded: {config['model_name']} ", "="*25)
    return True

def main(args):
    start_time = time.time()
    todownload_cnt, downloaded_cnt = 1, 0
    
    if args.from_config: # download a bunch of models from the config file
        with open(model_todownload_config_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        if not isinstance(configs, list) or not isinstance(configs[0], dict): raise TypeError("the json-like config file should be a list of dict")

        unsucceeded_configs, todownload_cnt = [], len(configs)
        for idx, config in enumerate(configs):
            succeeded = download(config, idx=idx)
            if not succeeded: unsucceeded_configs.append(config)
            
        downloaded_cnt = todownload_cnt - len(unsucceeded_configs)
        with open(model_todownload_config_path, 'w', encoding='utf-8') as f:
            json.dump(unsucceeded_configs, f)
    else: # download the single model configed by the arguments
        succeeded = download(vars(args))
        if succeeded: downloaded_cnt = 1
        
    elapsed_time = time.time() - start_time
    print("="*25, f" The model(s) has been downloaded: {downloaded_cnt} / {todownload_cnt}, within {time.strftime('%H h: %M m: %S s', time.gmtime(elapsed_time))} seconds ", "="*25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The script to download a model from HuggingFace or its mirror sites.")
    parser.add_argument("--model_name", type=str, required=True, help="The model's name shown on the huggingface model page, like 'meta-llama/Llama-2-7b-chat-hf'")
    parser.add_argument("--file_name", type=str, default="", help="The specific file name in the model repo, like 'config.json', default is empty to download the whole model repo")
    parser.add_argument("--version", type=str, default="", help="The specific version of the model, like 'v1.0', default is empty for the latest version")
    parser.add_argument("--save_dir", type=str, required=True, help=f"The directory to save the downloaded model repo or files, under the root dir: {model_root}")
    parser.add_argument("--auth_token", type=str, default="", help="The auth token to download the model repo, default is empty to use the default auth token")
    parser.add_argument("--allow_patterns", nargs='*', default="", help="The wildcards-style patterns to allow downloading, default is empty list to allow everything, only used when downloading the entire repo, i.e. file_name=''")
    parser.add_argument("--ignore_patterns", nargs='*', default="", help="The wildcards-style patterns to ignore when downloading, default is empty list to allow everything, only used when downloading the entire repo, i.e. file_name=''")
    parser.add_argument("--from_config", action="store_true", help=f"setting this True to ignore the args above, and directly use the json-like config file: {model_todownload_config_path} to download the models, "+ 
                        "containing a list of dict-like configs, each of which gives the arguments for a model to download "+ 
                        f"and would be cleaned and logged into {model_downloaded_log_path} if successfully downloaded"
                        )

    args = parser.parse_args()
    main(args)
    