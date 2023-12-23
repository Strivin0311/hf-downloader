# hf-downloader
A repo to manage the downloaded resources from huggingface including model shards and dataset splits, with some scripts to add new resources, get meta info, view the resource structure, etc


Note: 
* we use the official tool [huggingface_hub](https://huggingface.co/docs/huggingface_hub/guides/download) to download the raw files from huggingface, which can be installed by pip as below:
  ```sh
  pip install -U huggingface_hub
  ```
* 