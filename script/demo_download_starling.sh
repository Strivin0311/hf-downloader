python src/download_model.py --model_name berkeley-nest/Starling-RM-7B-alpha \
--save_dir llama2/starling/ \
--download_mode all \
--allow_patterns *.json *.py *.bin *.pth \
--max_retry 8 \