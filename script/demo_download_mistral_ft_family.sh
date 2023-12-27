# Uncomment the following json-like config and copy it to overwrite ./config/model_todownload.json
# [
#     {
#         "download_mode": "all",
#         "model_name": "teknium/OpenHermes-2.5-Mistral-7B",
#         "file_name": "",
#         "version": "",
#         "save_dir": "mistral/openhermes",
#         "allow_patterns": ["*.json", "*.model", "*.safetensors", "*.py"],
#         "ignore_patterns": []
#     },
#     {
#         "download_mode": "all",
#         "model_name": "Weyaxi/OpenHermes-2.5-neural-chat-7b-v3-1-7B",
#         "file_name": "",
#         "version": "",
#         "save_dir": "mistral/openhermes",
#         "allow_patterns": ["*.json", "*.model", "*.safetensors"],
#         "ignore_patterns": []
#     },
#     {
#         "download_mode": "all",
#         "model_name": "Intel/neural-chat-7b-v3-1",
#         "file_name": "",
#         "version": "",
#         "save_dir": "mistral/neuralchat",
#         "allow_patterns": ["*.json", "*.model", "*.safetensors"],
#         "ignore_patterns": []
#     },
#     {
#         "download_mode": "all",
#         "model_name": "HuggingFaceH4/zephyr-7b-beta",
#         "file_name": "",
#         "version": "",
#         "save_dir": "mistral/zephyr",
#         "allow_patterns": ["*.json", "*.model", "*.safetensors". "training_args.bin"],
#         "ignore_patterns": []
#     }
# ]

python src/download_model.py --from_config --max_retry 30