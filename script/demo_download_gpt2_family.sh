# Uncomment the following dict-like config and copy it to overwrite ./config/model_todownload.json
# [
#     {'model_name': 'gpt2', 'save_dir': 'gpt2'},
#     {'model_name': 'gpt2-medium', 'save_dir': 'gpt2'},
#     {'model_name': 'gpt2-large', 'save_dir': 'gpt2'},
#     {'model_name': 'gpt2-xl', 'save_dir': 'gpt2'},
# ]

python src/download_model.py \
--from_config