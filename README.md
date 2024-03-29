# hf-downloader
A repo to manage the downloaded resources from huggingface including model shards and dataset splits, with some scripts to add new resources, get meta info, view the resource structure, etc

(*assuming your work-dir is `hf-downloader/` right now*)

#### Preparation:

* Above all, make sure your local machine can access to huggingface by simply pinging huggingface below (*If not, get a VPN and try again*):
  ```sh
  ping huggingface.co/
  ```

* Creat the `.env` file to set some specific environment variables, which needs the python package `dotenv` installed by pip below:
  ```sh
  pip install -U python-dotenv
  ```
  
* Set the root directory to put all the models and datasets in `.env` like:
  ```sh
  HF_MODEL_ROOT="/model/"
  HF_DATASET_ROOT="/dataset/"
  ```

* Set the mirror address for huggingface model/dataset hub in `.env` like:
  ```sh
  HF_MIRROR="https://huggingface.co/"
  ```

* The official tool [huggingface_hub](https://huggingface.co/docs/huggingface_hub/guides/download) is adopted to download the raw files from huggingface, which can be installed by pip as below:
  ```sh
  pip install -U huggingface_hub
  ```
* For faster download, there's another Rust-based tool [hf_transfer](https://huggingface.co/docs/huggingface_hub/v0.19.3/package_reference/environment_variables#hfhubenablehftransfer) to maximize the bandwidth used by dividing large files into smaller parts and transferring them simultaneously using multiple threads, but it lacks several user-friendly features such as resumable downloads and proxies, and sometimes it 's not as stable as the native `from_pretrained` single-thread way especially when your access to huggingface is not directly allowed. If you are running on a machine with high bandwidth to directly access to huggingface, you can increase your download speed with it by simply installing with pip and set the environment variable in the `.env` as below:
  ```sh
  pip install -U hf-transfer
  HF_HUB_ENABLE_HF_TRANSFER=1
  ```
* To perminently store the resources, we've already kept the argument for the huggingface_hub functions `local_dir_use_symlinks=False` in the source code to disable the symlinks from the cache system

#### Download model(s):

* To download a model, you only have to:
  * step1: search the [huggingface model hub](https://huggingface.co/models/) for the model name `source/to/model_name` you want
  * step2: pick the directory `path/to/save_dir` you want to save it relative to the root directory `HF_MODEL_ROOT` set in the `.env` as an enviromental variable:
  * step3: then simply run the command below, and you can wait until the model is successfully downloaded to the directory `HF_MODEL_ROOT/path/to/save_dir/source/to/model_name` and logged into `./log/model_downloaded.json`:
    ```sh
    python src/download_model.py --model_name `source/to/model_name` --save_dir `path/to/save_dir`
    ```
  * you can also refer to the more arguments usage in the demo scripts like `script/demo_download_xxx.sh`
  * and the model config template can be found in `config/model_config_template.json`

* To download a bunch of models sequentially in one go, you only have to:
  * step1: encode the arguments you learned above into a dict-like config and append to the list in `./config/model_todownload.json` for each model you want to download
  * step2: then simply run the command below, and you can patiently wait until the models are all successfully downloaded to the corresponding directories and logged into `./log/model_downloaded.json`: 
    ```sh
    python src/download_model.py --from_config
    ```

* (Easiest) Or you can use the most stable pipeline script `./script/model_download_pipe.sh`, and the few variables you need to set are (taking `chatglm3-6b-128k` as an example), then you can just run the script to download all of the necessary files of the model you want from the hf hub:
  ```sh
  MODEL_NAME=THUDM/chatglm3-6b-128k
  SAVE_DIR=chatglm
  NUM_MODEL_SHARDS=7
  MODEL_FILE_FORMAT=bin
  ```

#### Download dataset(s):

* To download a dataset:
  ```sh
  
  ```


#### Convert any model to Safetensors and open a PR

[here](https://huggingface.co/spaces/safetensors/convert)