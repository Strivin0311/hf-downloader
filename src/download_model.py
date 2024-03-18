import os
import argparse
import time
import json
import traceback

import transformers
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download, snapshot_download


from utils import get_dir_size, add_size_str, update_config_dict, get_proxies_dict, get_order_str

from dotenv import load_dotenv
load_dotenv(".env")

## load some environment variables
model_root = os.getenv('HF_MODEL_ROOT')
if model_root is None: raise ValueError("HF_MODEL_ROOT is not set")

hf_mirror = os.getenv('HF_ENDPOINT') # could be None
if hf_mirror is None: hf_mirror = "https://huggingface.co" # default huggingface official cite

hf_proxies = get_proxies_dict(os.getenv('HF_PROXIES')) # could be None
hf_auth_token = os.getenv('HF_AUTH_TOKEN') # could be None

model_config_template_path = './config/model_config_template.json'
model_todownload_config_path = './config/model_todownload.json'
model_downloaded_log_path = './log/model_downloaded.json'

## set config template dict
config_template = {}
with open(model_config_template_path, 'r', encoding='utf-8') as f: config_template = json.load(f)

    
def check_config(config: dict):
    if config.get('model_name', "") == "" and config.get(''):
        raise ValueError("model_name is required in the config fields")
    if config.get('save_dir', "") == "":
        raise ValueError("save_dir is required in the config fields")

def downloaded_log(config, elapsed_time):
    # additional info to config
    config['download_size'] = get_dir_size(config['save_dir'], format=True)
    config['download_link'] = os.path.join(hf_mirror, config['model_name'])
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
        print(f"model_downloaded_log_path: {model_downloaded_log_path}")
        json.dump(downloaded_logs, f)

def download(config_dict: dict, idx=-1, max_retry=0):
    config = update_config_dict(config_template, config_dict)
    check_config(config)
    
    config['save_dir'] = os.path.join(model_root, config['save_dir'], config['model_name'])
    if not os.path.exists(config['save_dir']): os.makedirs(config['save_dir'])

    def download_single_time(config):
        try:
            # print(config); return # FIXME: debug
            if config['download_mode'] == "necessary": # to download necessary files from the model repo to load tokenizer and pretrained model
                tokenizer = AutoTokenizer.from_pretrained(
                    config['model_name'],
                    revision=config['version'] if config['version'] not in ["", "latest"] else 'main',
                    proxies=hf_proxies,
                    token=hf_auth_token,
                    resume_download=True,
                    trust_remote_code=True,
                )
                tokenizer.save_pretrained(config['save_dir'])

                model = AutoModel.from_pretrained(
                    config['model_name'], 
                    revision=config['version'] if config['version'] not in ["", "latest"] else 'main',
                    proxies=hf_proxies,
                    token=hf_auth_token,
                    resume_download=True,
                    trust_remote_code=True,
                )
                model.save_pretrained(config['save_dir'])
            elif config['download_mode'] == "specific": # to download specific files from the model repo
                hf_hub_download(
                    repo_type='model',
                    repo_id=config['model_name'], 
                    filename=config['file_name'], 
                    revision=config['version'] if config['version'] != "" else 'main',
                    proxies=hf_proxies,
                    token=hf_auth_token,
                    resume_download=True,
                    local_dir=config['save_dir'], 
                    local_dir_use_symlinks=False # NOTE: directly download the files instead of symlinks to the cache for perminent storage
                )
            elif config['download_mode'] == "all": # to download the whole model repo
                snapshot_download(
                    repo_type='model',
                    repo_id=config['model_name'],
                    revision=config['version'] if config['version'] not in ["", "latest"] else 'main',
                    allow_patterns=config['allow_patterns'] if config['allow_patterns'] != [] else None,
                    ignore_patterns=config['ignore_patterns'] if config['ignore_patterns'] != [] else None,
                    proxies=hf_proxies,
                    token=hf_auth_token,
                    resume_download=True,
                    local_dir=config['save_dir'], 
                    local_dir_use_symlinks=False # NOTE: directly download the files instead of symlinks to the cache for perminent storage
                )
            else: raise ValueError("download_mode should be one of ['necessary', 'specific', 'all']")
        except Exception as e:
            print("="*25, f" Download paused due to the thrown error\n: {e}\n with the trackback below: \n{traceback.format_exc()}\n", "="*25)
            return False
        return True
    
    start_time = time.time()
    for try_idx in range(max_retry+1):
        print("="*25, f" Downloading the{get_order_str(idx)}model: {config['model_name']} for the{get_order_str(try_idx)}time", "="*25)
        # print(config); return # FIXME: debug
        succeed = download_single_time(config)
        if succeed: break
    else:
        print("="*25, f" Failed to download: {config['model_name']} out of {try_idx+1} maximum tries", "="*25)
        return False
    
    elapsed_time = time.time() - start_time
    downloaded_log(config, elapsed_time)
    print("="*25, f" Successfully downloaded: {config['model_name']} with {try_idx+1} tries", "="*25)
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
            succeeded = download(config, idx=idx, max_retry=args.max_retry)
            if not succeeded: unsucceeded_configs.append(config)
            
        downloaded_cnt = todownload_cnt - len(unsucceeded_configs)
        with open(model_todownload_config_path, 'w', encoding='utf-8') as f:
            json.dump(unsucceeded_configs, f)
    else: # download the single model configed by the arguments
        succeeded = download(vars(args), max_retry=args.max_retry)
        if succeeded: downloaded_cnt = 1
        
    elapsed_time = time.time() - start_time
    print("="*25, f" The model(s) has been downloaded: {downloaded_cnt} / {todownload_cnt}, within {time.strftime('%H h: %M m: %S s', time.gmtime(elapsed_time))} seconds ", "="*25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The script to download a model from HuggingFace or its mirror sites.")
    parser.add_argument("--from_config", action="store_true", help=f"setting this True to ignore the args above, and directly use the json-like config file: {model_todownload_config_path} to download the models, "+ 
                        "containing a list of dict-like configs, each of which gives the arguments for a model to download "+ 
                        f"and would be cleaned and logged into {model_downloaded_log_path} if successfully downloaded"
                        )
    
    parser.add_argument("--model_name", type=str, default="", help="The model's name shown on the huggingface model page, like 'meta-llama/Llama-2-7b-chat-hf', keep it empty only when using config file")
    parser.add_argument("--save_dir", type=str, default="", help=f"The directory to save the downloaded model repo or files, under the root dir: {model_root}, keep it empty only when using config file")
    parser.add_argument("--download_mode", type=str, default='necessary', choices=['necessary', 'specific', 'all'], 
                        help="The mode to download the model, default is 'necessary' to download what it needs to load the model weights and tokenizer, " + 
                        "while 'specific' is to download the specific files from the model repo " + 
                        "'all' is to download the whole model repo with some allow_patterns/ignore_patterns")

    parser.add_argument("--file_name", type=str, default="", help="In 'specific' download mode, the specific file name in the model repo, like 'config.json', default is empty to download the whole model repo")
    parser.add_argument("--version", type=str, default="", help="In 'specific' download mode, the specific version of the model, like 'v1.0', default is empty for the latest version")
    parser.add_argument("--allow_patterns", nargs='*', default=[], help="In 'all' download mode, The wildcards-style patterns to allow downloading, default is empty list to allow everything, only used when downloading the entire repo, i.e. file_name=''")
    parser.add_argument("--ignore_patterns", nargs='*', default=[], help="In 'all' download mode, The wildcards-style patterns to ignore when downloading, default is empty list to allow everything, only used when downloading the entire repo, i.e. file_name=''")

    parser.add_argument("--max_retry", type=int, default=0, help="The maximum retry times to download the specific model in case of network failure, default is 0 to NOT retry")
    

    args = parser.parse_args()
    main(args)
    