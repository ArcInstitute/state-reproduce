#!/bin/bash

MODEL_NAME=$1
DATASET_NAME=$2
FOLD_ID=$3

# Define output directory
OUTPUT_DIR_BASE="/large_storage/goodarzilab/userspace/mohsen/VCI-PertBench/state_sets_reproduce"

# Define test tasks for each fold
if [ $DATASET_NAME == "replogle" ]; then
    if [ $FOLD_ID -eq 1 ]; then
        DATA_TOML_PATH="/large_storage/ctc/ML/state_sets/replogle_filtered/hepg2.toml"
    elif [ $FOLD_ID -eq 2 ]; then
        DATA_TOML_PATH="/large_storage/ctc/ML/state_sets/replogle_filtered/jurkat.toml"
    elif [ $FOLD_ID -eq 3 ]; then
        DATA_TOML_PATH="/large_storage/ctc/ML/state_sets/replogle_filtered/k562.toml"
    elif [ $FOLD_ID -eq 4 ]; then
        DATA_TOML_PATH="/large_storage/ctc/ML/state_sets/replogle_filtered/rpe1.toml"
    fi

    OUTPUT_DIR="${OUTPUT_DIR_BASE}/${MODEL_NAME}_replogle/fold${FOLD_ID}/"
    WANDB_TAGS="[${MODEL_NAME},replogle,fold${FOLD_ID}]"
    TRAINING_NAME=${MODEL_NAME}

    if [ $MODEL_NAME == "scgpt" ]; then
        MODEL_NAME="scgpt-genetic"
        TRAINING_NAME="scgpt"
    fi

    BATCH_COL="gem_group"
    PERT_COL="gene"
    CELL_TYPE_KEY="cell_type"
    CONTROL_PERT="non-targeting"
elif [ $DATASET_NAME == "tahoe" ]; then
    DATA_TOML_PATH="/large_storage/ctc/userspace/aadduri/datasets/tahoe_5_holdout/generalization.toml"
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/${MODEL_NAME}_tahoe/tahoe_generalization/"
    WANDB_TAGS="[${MODEL_NAME},tahoe,generalization]"
    if [ $MODEL_NAME == "scgpt" ]; then
        MODEL_NAME="scgpt-chemical"
        TRAINING_NAME="scgpt"
    fi

    BATCH_COL="sample"
    PERT_COL="drugname_drugconc"
    CELL_TYPE_KEY="cell_name"
    CONTROL_PERT="DMSO_TF"
fi

echo "Training $MODEL_NAME on $DATASET_NAME fold $FOLD_ID"
echo "Output directory: $OUTPUT_DIR"
echo "Wandb tags: $WANDB_TAGS"
echo "Model name: $MODEL_NAME"
echo "Data toml path: $DATA_TOML_PATH"

uv run python -m state_sets_reproduce.train \
    data.kwargs.toml_config_path=$DATA_TOML_PATH \
    data.kwargs.embed_key=X_hvg \
    data.kwargs.basal_mapping_strategy=random \
    data.kwargs.output_space=gene \
    data.kwargs.num_workers=24 \
    data.kwargs.batch_col=${BATCH_COL} \
    data.kwargs.pert_col=${PERT_COL} \
    data.kwargs.cell_type_key=${CELL_TYPE_KEY} \
    data.kwargs.control_pert=${CONTROL_PERT} \
    training.max_steps=50000 \
    training.val_freq=1000 \
    training.test_freq=2000 \
    training.batch_size=128 \
    wandb=mohsen \
    wandb.tags="${WANDB_TAGS}" \
    model=${MODEL_NAME} \
    training=${TRAINING_NAME} \
    output_dir="${OUTPUT_DIR}" \
    name="fold${FOLD_ID}"