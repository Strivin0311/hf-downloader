# Uncomment the following json-like config and copy it to overwrite ./config/model_todownload.json
# [
#     {
#         "download_mode": "all",
#         "model_name": "THUDM/chatglm3-6b-base",
#         "file_name": "",
#         "version": "",
#         "save_dir": "chatglm",
#         "allow_patterns": ["*.json", "*.py", "*.model", "*.bin"],
#         "ignore_patterns": []
#     },
#     {
#         "download_mode": "all",
#         "model_name": "THUDM/chatglm3-6b",
#         "file_name": "",
#         "version": "",
#         "save_dir": "chatglm",
#         "allow_patterns": ["*.json", "*.py", "*.model", "*.bin"],
#         "ignore_patterns": []
#     },
#     {
#         "download_mode": "all",
#         "model_name": "THUDM/chatglm3-6b-32k",
#         "file_name": "",
#         "version": "",
#         "save_dir": "chatglm",
#         "allow_patterns": ["*.json", "*.py", "*.model", "*.bin"],
#         "ignore_patterns": []
#     }
# ]

python src/download_model.py --from_config --max_retry 30