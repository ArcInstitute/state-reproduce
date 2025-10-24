#!/bin/bash

MODEL_NAME=$1
DATASET_NAME=$2
FOLD_ID=$3
if [ $# -eq 4 ]; then
    CKPT=$4
else
    CKPT=""
fi

# Define output directory
OUTPUT_DIR_BASE="/large_storage/goodarzilab/userspace/mohsen/VCI-PertBench/state_revisions/debug/"

# Define test tasks for each fold
if [ $DATASET_NAME = "replogle" ]; then
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/${MODEL_NAME}_replogle/fold${FOLD_ID}/"
    if [ -z "$CKPT" ]; then
        CKPT="final.ckpt"
    fi
elif [ $DATASET_NAME = "tahoe" ]; then
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/${MODEL_NAME}_tahoe/tahoe_generalization/"
    if [ -z "$CKPT" ]; then
        CKPT="last.ckpt"
    fi
    if [ $MODEL_NAME = "lrlm" ]; then
        CKPT="final.ckpt"
    fi
elif [ $DATASET_NAME = "parse" ]; then
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/${MODEL_NAME}_parse/${FOLD_ID}/"
    if [ -z "$CKPT" ]; then
        CKPT="last.ckpt"
    fi
elif [ $DATASET_NAME = "xaira" ]; then
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/${MODEL_NAME}_xaira/${FOLD_ID}/"
    if [ -z "$CKPT" ]; then
        CKPT="final.ckpt"
    fi
fi

echo "Generating Predictions for $MODEL_NAME on $DATASET_NAME (fold: $FOLD_ID)"
echo "Output directory: $OUTPUT_DIR"

uv run python -m state_sets_reproduce.train.get_predictions \
    --output_dir ${OUTPUT_DIR} \
    --checkpoint ${CKPT} 
