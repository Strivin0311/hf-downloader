#!/bin/bash

DATA_ROOT="/data/dataset"
DATA_DIR=$DATA_ROOT/openwebtext

mkdir -p $DATA_DIR
cd $DATA_DIR

README_LINK="https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/README.md"
SCRIPT_LINK="https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/openwebtext.py"

wget $README_LINK
wget $SCRIPT_LINK

SUBSET_DIR=$DATA_DIR/subsets
mkdir -p $SUBSET_DIR
cd $SUBSET_DIR

SUBSET_LINK_PREFIX="https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/subsets/urlsf_subset"
MAX_SUBSET_IDX=20

for i in $(seq 0 $MAX_SUBSET_IDX); do
    if [ $i -lt 10 ]; then
        wget $SUBSET_LINK_PREFIX"0"$i.tar
    else
        wget $SUBSET_LINK_PREFIX$i.tar
    fi
done

echo "All subsets are downloaded into $SUBSET_DIR"
