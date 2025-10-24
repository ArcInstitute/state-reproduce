#!/bin/bash

MODEL_NAME=$1
DATASET_NAME=$2
FOLD_ID=$3

# Define output directory
OUTPUT_DIR_BASE="/large_storage/goodarzilab/userspace/mohsen/VCI-PertBench/state_revisions/debug_v3/"

# Define test tasks for each fold
if [ "$DATASET_NAME" = "replogle" ]; then
    if [ "$FOLD_ID" = "1" ]; then
        DATA_TOML_PATH="/large_storage/ctc/userspace/aadduri/revisions/replogle_nogwps_v2/hepg2.toml"
    elif [ "$FOLD_ID" = "2" ]; then
        DATA_TOML_PATH="/large_storage/ctc/userspace/aadduri/revisions/replogle_nogwps_v2/jurkat.toml"
    elif [ "$FOLD_ID" = "3" ]; then
        DATA_TOML_PATH="/large_storage/ctc/userspace/aadduri/revisions/replogle_nogwps_v2/k562.toml"
    elif [ "$FOLD_ID" = "4" ]; then
        DATA_TOML_PATH="/large_storage/ctc/userspace/aadduri/revisions/replogle_nogwps_v2/rpe1.toml"
    fi

    OUTPUT_DIR="${OUTPUT_DIR_BASE}/${MODEL_NAME}_replogle_v2/"
    WANDB_TAGS="[${MODEL_NAME},replogle,fold${FOLD_ID}]"
    TRAINING_NAME=${MODEL_NAME}

    if [ "$MODEL_NAME" = "scgpt" ]; then
        MODEL_NAME="scgpt-genetic"
        TRAINING_NAME="scgpt"
    fi

    BATCH_COL="gem_group"
    PERT_COL="gene"
    CELL_TYPE_KEY="cell_line"
    CONTROL_PERT="non-targeting"
    FOLD_NAME="fold${FOLD_ID}"
elif [ "$DATASET_NAME" = "tahoe" ]; then
    DATA_TOML_PATH="/large_storage/ctc/ML/transcriptomics_filtered/tahoe_se/generalization.toml"
    OUTPUT_DIR="${OUTPUT_DIR_BASE}/${MODEL_NAME}_tahoe/"
    WANDB_TAGS="[${MODEL_NAME},tahoe,generalization]"
    
    TRAINING_NAME=${MODEL_NAME}
    if [ "$MODEL_NAME" = "scgpt" ]; then
        MODEL_NAME="scgpt-chemical"
        TRAINING_NAME="scgpt"
    fi

    if [ "$MODEL_NAME" = "gears" ]; then
        MODEL_NAME="gears-chemical"
        TRAINING_NAME="gears"
    fi
    
    BATCH_COL="sample"
    PERT_COL="drugname_drugconc"
    CELL_TYPE_KEY="cell_name"
    CONTROL_PERT="DMSO_TF"
    FOLD_NAME="tahoe_generalization"
elif [ "$DATASET_NAME" = "parse" ]; then
    if [ "$FOLD_ID" = "donor" ]; then
        DATA_TOML_PATH="/large_storage/ctc/ML/state_sets/parse_se/donor.toml"
        BATCH_COL="cell_type"
        CELL_TYPE_KEY="donor_ct_clean"
        FOLD_NAME=${FOLD_ID}
    elif [ "$FOLD_ID" = "cell_type" ]; then
        DATA_TOML_PATH="/large_storage/ctc/ML/state_sets/parse_se/celltype.toml"
        BATCH_COL="donor"
        CELL_TYPE_KEY="cell_type"
        FOLD_NAME=${FOLD_ID}
    elif [ "$FOLD_ID" = "1" ]; then
        DATA_TOML_PATH="/large_storage/goodarzilab/userspace/mohsen/VCI/datasets/parse/split_1.toml"
        BATCH_COL="donor"
        CELL_TYPE_KEY="cell_type_clean"
        FOLD_NAME="split_1"
    elif [ "$FOLD_ID" = "2" ]; then
        DATA_TOML_PATH="/large_storage/goodarzilab/userspace/mohsen/VCI/datasets/parse/split_2.toml"
        BATCH_COL="donor"
        CELL_TYPE_KEY="cell_type_clean"
        FOLD_NAME="split_2"
    elif [ "$FOLD_ID" = "3" ]; then
        DATA_TOML_PATH="/large_storage/goodarzilab/userspace/mohsen/VCI/datasets/parse/split_3.toml"
        BATCH_COL="donor"
        CELL_TYPE_KEY="cell_type_clean"
        FOLD_NAME="split_3"
    elif [ "$FOLD_ID" = "4" ]; then
        DATA_TOML_PATH="/large_storage/goodarzilab/userspace/mohsen/VCI/datasets/parse/split_4.toml"
        BATCH_COL="donor"
        CELL_TYPE_KEY="cell_type_clean"
        FOLD_NAME="split_4"
    elif [ "$FOLD_ID" = "5" ]; then
        DATA_TOML_PATH="/large_storage/goodarzilab/userspace/mohsen/VCI/datasets/parse/split_5.toml"
        BATCH_COL="donor"
        CELL_TYPE_KEY="cell_type_clean"
        FOLD_NAME="split_5"
    fi

    OUTPUT_DIR="${OUTPUT_DIR_BASE}/${MODEL_NAME}_parse/"
    WANDB_TAGS="[${MODEL_NAME},parse,${FOLD_NAME}]"

    TRAINING_NAME=${MODEL_NAME}

    if [ "$MODEL_NAME" = "scgpt" ]; then
        MODEL_NAME="scgpt-chemical"
        TRAINING_NAME="scgpt"
    fi
    if [ "$MODEL_NAME" = "gears" ]; then
        MODEL_NAME="gears-chemical"
        TRAINING_NAME="gears"
    fi

    PERT_COL="cytokine"
    CONTROL_PERT="PBS"
    
elif [ "$DATASET_NAME" = "xaira" ]; then
    if [ "$FOLD_ID" = "HEK293T" ]; then
        DATA_TOML_PATH="/large_storage/ctc/ML/state_sets/xaira/HEK293T.toml"
    elif [ "$FOLD_ID" = "HCT116" ]; then
        DATA_TOML_PATH="/large_storage/ctc/ML/state_sets/xaira/HCT116.toml"
    fi

    OUTPUT_DIR="${OUTPUT_DIR_BASE}/${MODEL_NAME}_xaira/"
    WANDB_TAGS="[${MODEL_NAME},xaira,${FOLD_ID}]"
    
    TRAINING_NAME=${MODEL_NAME}
    if [ "$MODEL_NAME" = "scgpt" ]; then
        MODEL_NAME="scgpt-genetic"
        TRAINING_NAME="scgpt"
    fi

    BATCH_COL="sample"
    PERT_COL="gene_target"
    CELL_TYPE_KEY="cell_type"
    CONTROL_PERT="Non-Targeting"
    FOLD_NAME=${FOLD_ID}
fi

echo "Training $MODEL_NAME on $DATASET_NAME fold $FOLD_ID"
echo "Output directory: $OUTPUT_DIR"
echo "Wandb tags: $WANDB_TAGS"
echo "Model name: $MODEL_NAME"
echo "Data toml path: $DATA_TOML_PATH"

BATCH_SIZE=128
if [ "$MODEL_NAME" = "lrlm" ]; then
    GENE_EMB="scgpt"

    if [ $# -eq 4 ]; then
        PERT_EMB=$4
    else
        if [ "$DATASET_NAME" = "replogle" ]; then
            PERT_EMB="scgpt"
            PROCESS_PERT_NAMES=""
        elif [ "$DATASET_NAME" = "tahoe" ]; then
            PERT_EMB="tahoe_rdkit"
            PROCESS_PERT_NAMES="tahoe_style"
        elif [ "$DATASET_NAME" = "parse" ]; then
            PERT_EMB="identity"
            PROCESS_PERT_NAMES=""
        elif [ "$DATASET_NAME" = "xaira" ]; then
            PERT_EMB="gears_norman"
            PROCESS_PERT_NAMES=""
        fi
    fi

    BATCH_SIZE=8192

    echo "using gene embedding: $GENE_EMB"
    echo "using perturbation embedding: $PERT_EMB"

    echo "Running the following command:"
    python -m state_sets_reproduce.train \
        data.kwargs.toml_config_path=$DATA_TOML_PATH \
        data.kwargs.embed_key=X_hvg \
        data.kwargs.basal_mapping_strategy=random \
        data.kwargs.output_space=gene \
        data.kwargs.num_workers=24 \
        data.kwargs.batch_col=${BATCH_COL} \
        data.kwargs.pert_col=${PERT_COL} \
        data.kwargs.cell_type_key=${CELL_TYPE_KEY} \
        data.kwargs.control_pert=${CONTROL_PERT} \
        model.kwargs.process_pert_names=${PROCESS_PERT_NAMES} \
        model.kwargs.gene_emb=${GENE_EMB} \
        model.kwargs.pert_emb=${PERT_EMB} \
        training.max_steps=250000 \
        training.val_freq=5000 \
        training.test_freq=9000 \
        training.batch_size=${BATCH_SIZE} \
        wandb=mohsen \
        wandb.tags="${WANDB_TAGS}" \
        model=${MODEL_NAME} \
        training=${TRAINING_NAME} \
        output_dir="${OUTPUT_DIR}" \
        name="${FOLD_NAME}"
else
    echo "Running the following command:"
    python -m state_sets_reproduce.train \
        data.kwargs.toml_config_path=$DATA_TOML_PATH \
        data.kwargs.embed_key=X_hvg \
        data.kwargs.basal_mapping_strategy=random \
        data.kwargs.output_space=gene \
        data.kwargs.num_workers=24 \
        data.kwargs.batch_col=${BATCH_COL} \
        data.kwargs.pert_col=${PERT_COL} \
        data.kwargs.cell_type_key=${CELL_TYPE_KEY} \
        data.kwargs.control_pert=${CONTROL_PERT} \
        training.max_steps=250000 \
        training.val_freq=5000 \
        training.test_freq=9000 \
        training.batch_size=128 \
        wandb=mohsen \
        wandb.tags="${WANDB_TAGS}" \
        model=${MODEL_NAME} \
        training=${TRAINING_NAME} \
        output_dir="${OUTPUT_DIR}" \
        name="${FOLD_NAME}"
fi