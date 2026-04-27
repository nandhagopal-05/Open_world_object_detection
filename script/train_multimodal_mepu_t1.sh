#!/bin/bash
# Training script for Multi-Modal MEPU with Uncertainty-Aware Pseudo-Labeling
# S-OWOD Benchmark - Task 1
# 
# IMPORTANT: Activate the conda environment before running this script:
#   conda activate best
#   bash script/train_multimodal_mepu_t1.sh

echo "========================================="
echo "Multi-Modal MEPU Training - Task 1"
echo "========================================="

# Configuration
NUM_GPUS=1
CONFIG_FILE="config/MEPU-SOWOD/t1/train_multimodal.yaml"
OUTPUT_DIR="training_dir/multimodal-mepu/sowod-t1"

# Step 1: Generate initial pseudo labels using FreeSOLO
echo ""
echo "Step 1: Generating pseudo labels with FreeSOLO..."
python tools/gen_pseudo_label_new.py \
    --proposal_path proposals/proposals_freesolo.json \
    --data_path datasets/sowod \
    --save_path datasets/sowod/Annotations/pseudo_label_fs_initial.json \
    --keep_type num \
    --num_keep 5 \
    --known_cls_num 19 \
    --data_split t1_train

# Step 2: Update Weibull models of REW using known object labels
echo ""
echo "Step 2: Updating REW Weibull models..."
python train_net.py \
    --dist-url auto \
    --num-gpus ${NUM_GPUS} \
    --config config/REW/rew_t1_sowod.yaml \
    OUTPUT_DIR training_dir/rew/sowod_t1 \
    MODEL.WEIGHTS training_dir/rew/model_final.pth

# Step 3: Extract multi-modal features and compute REW scores
echo ""
echo "Step 3: Computing multi-modal REW scores..."
python train_net.py \
    --eval-only \
    --inference-rew \
    --resume \
    --dist-url auto \
    --num-gpus ${NUM_GPUS} \
    --config config/REW/rew_t1_sowod.yaml \
    OUTPUT_DIR ${OUTPUT_DIR}-rew \
    DATASETS.TEST '("sowod_train_t1_fs",)' \
    OPENSET.OUTPUT_PATH_REW datasets/sowod/Annotations/pseudo_label_fs_initial.json \
    MULTIMODAL.ENABLED True

# Step 4: Estimate uncertainty using MC-Dropout and/or Ensemble
echo ""
echo "Step 4: Estimating uncertainty..."
python tools/estimate_uncertainty.py \
    --config config/REW/rew_t1_sowod.yaml \
    --model_weights training_dir/rew/sowod_t1/model_final.pth \
    --proposal_path datasets/sowod/Annotations/pseudo_label_fs_initial.json \
    --output_path datasets/sowod/Annotations/uncertainty_scores_t1.npy \
    --method mc_dropout \
    --n_samples 10

# Step 5: Generate high-quality pseudo labels with uncertainty filtering
echo ""
echo "Step 5: Generating uncertainty-aware pseudo labels..."
python tools/gen_pseudo_label_uncertainty.py \
    --proposal_path datasets/sowod/Annotations/pseudo_label_fs_initial.json \
    --rew_scores_path ${OUTPUT_DIR}-rew/rew_scores.npy \
    --uncertainty_scores_path datasets/sowod/Annotations/uncertainty_scores_t1.npy \
    --save_path datasets/sowod/Annotations/pseudo_label_fs_filtered.json \
    --known_cls_num 19 \
    --uncertainty_threshold 0.3 \
    --quality_threshold 0.5 \
    --keep_type quality \
    --use_active_learning \
    --active_learning_budget 1000

# Step 6: Train detector with known + filtered unknown pseudo labels
echo ""
echo "Step 6: Training multi-modal detector..."
python train_net.py \
    --resume \
    --dist-url auto \
    --num-gpus ${NUM_GPUS} \
    --config ${CONFIG_FILE} \
    DATASETS.TRAIN '("sowod_train_t1_multimodal",)' \
    OUTPUT_DIR ${OUTPUT_DIR}-train \
    OPENSET.REW.GAMMA 4.0 \
    MULTIMODAL.ENABLED True \
    UNCERTAINTY.ENABLED True

# Step 7: Self-training - inference with OLN for refinement
echo ""
echo "Step 7: Self-training inference with OLN..."
python train_net.py \
    --resume \
    --eval-only \
    --dist-url auto \
    --num-gpus ${NUM_GPUS} \
    --config config/MEPU-SOWOD/t1/self-train.yaml \
    DATASETS.TEST '("sowod_train_t1",)' \
    OUTPUT_DIR ${OUTPUT_DIR}-self-train \
    OPENSET.OLN_INFERENCE True \
    OPENSET.INFERENCE_SELT_TRAIN True \
    MODEL.WEIGHTS ${OUTPUT_DIR}-train/model_final.pth \
    MULTIMODAL.ENABLED True

# Step 8: Generate refined pseudo labels using OLN proposals
echo ""
echo "Step 8: Generating refined pseudo labels..."
python tools/gen_pseudo_label_new.py \
    --proposal_path ${OUTPUT_DIR}-self-train/inference/inference_results.json \
    --data_path datasets/sowod \
    --save_path datasets/sowod/Annotations/pseudo_label_st_initial.json \
    --keep_type percent \
    --percent_keep 0.3 \
    --known_cls_num 19 \
    --data_split t1_train

# Step 9: Assign REW scores to refined pseudo labels
echo ""
echo "Step 9: Computing REW scores for refined proposals..."
python train_net.py \
    --eval-only \
    --inference-rew \
    --resume \
    --dist-url auto \
    --num-gpus ${NUM_GPUS} \
    --config config/REW/rew_t1_sowod.yaml \
    OUTPUT_DIR training_dir/rew/sowod_t1 \
    DATASETS.TEST '("sowod_train_t1_st",)' \
    OPENSET.OUTPUT_PATH_REW datasets/sowod/Annotations/pseudo_label_st_initial.json \
    MULTIMODAL.ENABLED True

# Step 10: Re-compute uncertainty for refined proposals
echo ""
echo "Step 10: Re-estimating uncertainty for refined proposals..."
python tools/estimate_uncertainty.py \
    --config config/REW/rew_t1_sowod.yaml \
    --model_weights training_dir/rew/sowod_t1/model_final.pth \
    --proposal_path datasets/sowod/Annotations/pseudo_label_st_initial.json \
    --output_path datasets/sowod/Annotations/uncertainty_scores_t1_refined.npy \
    --method mc_dropout \
    --n_samples 10

# Step 11: Filter refined pseudo labels with uncertainty
echo ""
echo "Step 11: Filtering refined pseudo labels..."
python tools/gen_pseudo_label_uncertainty.py \
    --proposal_path datasets/sowod/Annotations/pseudo_label_st_initial.json \
    --rew_scores_path datasets/sowod/Annotations/pseudo_label_st_initial.json \
    --uncertainty_scores_path datasets/sowod/Annotations/uncertainty_scores_t1_refined.npy \
    --save_path datasets/sowod/Annotations/pseudo_label_st_filtered.json \
    --known_cls_num 19 \
    --uncertainty_threshold 0.3 \
    --quality_threshold 0.5 \
    --keep_type quality \
    --use_active_learning \
    --active_learning_budget 1000

# Step 12: Final training with refined pseudo labels
echo ""
echo "Step 12: Final training with refined pseudo labels..."
python train_net.py \
    --resume \
    --dist-url auto \
    --num-gpus ${NUM_GPUS} \
    --config config/MEPU-SOWOD/t1/self-train.yaml \
    DATASETS.TRAIN "(\"sowod_train_t1_multimodal_refined\",)" \
    OUTPUT_DIR ${OUTPUT_DIR}-final \
    OPENSET.REW.GAMMA 4.0 \
    MODEL.WEIGHTS ${OUTPUT_DIR}-train/model_final.pth \
    MULTIMODAL.ENABLED True \
    UNCERTAINTY.ENABLED True

echo ""
echo "========================================="
echo "Task 1 Training Complete!"
echo "Final model saved to: ${OUTPUT_DIR}-final/model_final.pth"
echo "========================================="
