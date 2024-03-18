MODEL_NAME=xai-org/grok-1
SAVE_DIR=/data1/model/grok/$MODEL_NAME/
# mkdir -p $SAVE_DIR

# 0~9
START_INDEX=0
NUM_MODEL_SHARDS=9
for i in $(seq $START_INDEX $NUM_MODEL_SHARDS); do
    wget https://huggingface.co/xai-org/grok-1/resolve/main/ckpt/tensor0000${i}_000 -O $SAVE_DIR/tensor0000${i}_000 \
    &
done

# 10~99
START_INDEX=10
NUM_MODEL_SHARDS=99
for i in $(seq $START_INDEX $NUM_MODEL_SHARDS); do
    wget https://huggingface.co/xai-org/grok-1/resolve/main/ckpt/tensor000${i}_000 -O $SAVE_DIR/tensor000${i}_000 \
    &
done

# 100~299
START_INDEX=100
NUM_MODEL_SHARDS=299
for i in $(seq $START_INDEX $NUM_MODEL_SHARDS); do
    wget https://huggingface.co/xai-org/grok-1/resolve/main/ckpt/tensor00${i}_000 -O $SAVE_DIR/tensor00${i}_000 \
    &
done

wait

echo "All model shards from idx ${START_INDEX} to ${NUM_MODEL_SHARDS} are downloads."

# 300~499
START_INDEX=300
NUM_MODEL_SHARDS=499
for i in $(seq $START_INDEX $NUM_MODEL_SHARDS); do
    wget https://huggingface.co/xai-org/grok-1/resolve/main/ckpt/tensor00${i}_000 -O $SAVE_DIR/tensor00${i}_000 \
    &
done

wait

echo "All model shards from idx ${START_INDEX} to ${NUM_MODEL_SHARDS} are downloads."

# 500~769
START_INDEX=500
NUM_MODEL_SHARDS=769
for i in $(seq $START_INDEX $NUM_MODEL_SHARDS); do
    wget https://huggingface.co/xai-org/grok-1/resolve/main/ckpt/tensor00${i}_000 -O $SAVE_DIR/tensor00${i}_000 \
    &
done

wait

echo "All model shards from idx ${START_INDEX} to ${NUM_MODEL_SHARDS} are downloads."