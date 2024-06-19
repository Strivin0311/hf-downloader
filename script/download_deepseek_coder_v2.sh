MODEL_NAME=deepseek-ai/DeepSeek-Coder-V2-Instruct
SAVE_DIR=/data1/model/llama2/deepseek/$MODEL_NAME/
# mkdir -p $SAVE_DIR

# 1~15
START_INDEX=1
NUM_MODEL_SHARDS=15
for i in $(seq $START_INDEX $NUM_MODEL_SHARDS); do
    FORMATED_IDX=$(printf "%05d\n" $i)
    wget https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct/resolve/main/model-${FORMATED_IDX}-of-000055.safetensors \
    &
done

# 16~31
START_INDEX=16
NUM_MODEL_SHARDS=31
for i in $(seq $START_INDEX $NUM_MODEL_SHARDS); do
    FORMATED_IDX=$(printf "%05d\n" $i)
    wget https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct/resolve/main/model-${FORMATED_IDX}-of-000055.safetensors \
    &
done

# 32~43
START_INDEX=32
NUM_MODEL_SHARDS=43
for i in $(seq $START_INDEX $NUM_MODEL_SHARDS); do
    FORMATED_IDX=$(printf "%05d\n" $i)
    wget https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct/resolve/main/model-${FORMATED_IDX}-of-000055.safetensors \
    &
done

# 44~55
START_INDEX=44
NUM_MODEL_SHARDS=55
for i in $(seq $START_INDEX $NUM_MODEL_SHARDS); do
    FORMATED_IDX=$(printf "%05d\n" $i)
    wget https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct/resolve/main/model-${FORMATED_IDX}-of-000055.safetensors \
    &
done

wait

echo "All ${NUM_MODEL_SHARDS} model shards are downloads."