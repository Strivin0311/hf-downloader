# Uncomment the following json-like config and copy it to overwrite ./config/model_todownload.json
# [
#     {
#         "download_mode": "all",
#         "model_name": "mistralai/Mistral-7B-v0.1",
#         "file_name": "",
#         "version": "",
#         "save_dir": "mistral",
#         "allow_patterns": ["*.json", "*.model", "*.safetensors"],
#         "ignore_patterns": []
#     },
#     {
#         "download_mode": "all",
#         "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
#         "file_name": "",
#         "version": "",
#         "save_dir": "mistral",
#         "allow_patterns": ["*.json", "*.model", "*.safetensors"],
#         "ignore_patterns": []
#     },
#     {
#         "download_mode": "all",
#         "model_name": "mistralai/Mixtral-8x7B-v0.1",
#         "file_name": "",
#         "version": "",
#         "save_dir": "mistral",
#         "allow_patterns": ["*.json", "*.model", "*.safetensors"],
#         "ignore_patterns": []
#     },
#     {
#         "download_mode": "all",
#         "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
#         "file_name": "",
#         "version": "",
#         "save_dir": "mistral",
#         "allow_patterns": ["*.json", "*.model", "*.safetensors"],
#         "ignore_patterns": []
#     }
# ]

python src/download_model.py --from_config --max_retry 30