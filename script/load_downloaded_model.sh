MODEL_PATH=llama3/unsloth/Llama3-8b
PEFT_PATH=""
DEVICES="2"

python src/load_downloaded_model.py \
--model_path $MODEL_PATH \
--model_type clm \
--task text_generation \
--devices $DEVICES