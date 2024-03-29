#!/bin/sh

# the specific information for the model you want to download from huggingface hub
MODEL_NAME=mistralai/Mistral-7B-v0.1
SAVE_DIR=mistral

NON_MODEL_FILE_PATTERNS="*.md *.json *.py *.model"
NUM_MODEL_SHARDS=2
MODEL_FILE_FORMAT=safetensors

# set prefix for model weights according to the model file format
if [ "$MODEL_FILE_FORMAT" = "safetensors" ]; then
    WEIGHT_PREFIX="model"
elif [ "$MODEL_FILE_FORMAT" = "bin" ]; then
    WEIGHT_PREFIX="pytorch_model"
else
    echo "Unknown model file format: $MODEL_FILE_FORMAT"
    exit 1
fi

# the default setting that you don't need to change for the most cases
MODEL_TYPE=clm
TASK="text_generation"
DEVICES="0"
PEFT_PATH=""

if [ $NUM_MODEL_SHARDS -eq 1 ]; then
    ALLOW_PATTERNS="$WEIGHT_PREFIX.${MODEL_FILE_FORMAT}"
else
    ALLOW_PATTERNS="$WEIGHT_PREFIX-0000${i}-of-0000${NUM_MODEL_SHARDS}.${MODEL_FILE_FORMAT}"
fi

## 1. download the non model files with only one process and only 10 times for retrying (enough)
python src/download_model.py \
--model_name $MODEL_NAME \
--save_dir $SAVE_DIR \
--download_mode all \
--allow_patterns  $NON_MODEL_FILE_PATTERNS \
--max_retry 10 \

echo "All non-model files are downloaded, including ${NON_MODEL_FILE_PATTERNS}."

## 2. download the model files with multiple processes and 30 times for retrying (sometimes may not enough)
for i in $(seq 1 $NUM_MODEL_SHARDS); do
    python src/download_model.py \
    --model_name $MODEL_NAME \
    --save_dir $SAVE_DIR \
    --download_mode all \
    --allow_patterns "$WEIGHT_PREFIX-0000${i}-of-0000${NUM_MODEL_SHARDS}.${MODEL_FILE_FORMAT}" \
    --max_retry 50 \
    &
done

# Wait for all model shard downloading processes to finish
wait

echo "All model shards from idx 1 to ${NUM_MODEL_SHARDS} are downloads."


## 3. loaded the downloaded model

echo "Loading the downloaded model automatically..."

python src/load_downloaded_model.py \
--model_path "${SAVE_DIR}/${MODEL_NAME}" \
--model_type $MODEL_TYPE \
--task $TASK \
--devices $DEVICES \
