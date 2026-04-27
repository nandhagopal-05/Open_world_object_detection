# Open-World Object Detection with Multi-Modal MEPU

This repository provides the enhanced Open-World Object Detection (OWOD) codebase built upon the Multi-modal Evidence Per-Unknown (MEPU) framework. Our core contribution tackles the challenging task of discriminating unknown objects from both the background and known categories by leveraging visual-linguistic multi-modal features and establishing mathematically grounded uncertainty estimation models.

## Key Features

1. **Multi-Modal Feature Fusion**: Integrates powerful vision-language model (CLIP) semantic features alongside robust visual features, addressing representation biases towards known classes.
2. **Weibull-Based Uncertainty Estimation**: Implements theoretically justified Weibull mixture modeling for accurate pseudo-labeling, estimating the epistemic and aleatoric uncertainty involved in discovering "unknown" object candidates.
3. **Advanced Pseudo-Labeling Pipeline**: Re-engineered iterative pseudo-labeling logic filtering out noise with strict uncertainty thresholds and an active-learning budget.

### Core File Structure
- `mepu/model/rew/multimodal_rew.py` & `fusion.py`: Our multi-modal fusion architecture.
- `mepu/model/uncertainty_estimator.py`: Formal framework for Weibull-based uncertainty modeling.
- `tools/gen_pseudo_label_uncertainty.py`: Uncertainty-aware pseudo-label generator.
- `tools/estimate_uncertainty.py`: Modular uncertainty estimator using MC-Dropout strategies.
- `script/train_multimodal_mepu_t1.sh`: Standardized multi-step end-to-end training and inference script for the Multi-Modal MEPU model.

---

## Installation & Prerequisites

1. **Environment Setup**:
   ```bash
   conda create -n mepu-multimodal python=3.8
   conda activate mepu-multimodal
   conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch
   ```
2. **Install Detectron2**:
   ```bash
   git clone https://github.com/facebookresearch/detectron2.git
   cd detectron2
   pip install -e .
   cd ..
   ```
3. **Install Multi-Modal Tools & Dependencies**:
   ```bash
   pip install ftfy regex
   pip install git+https://github.com/openai/CLIP.git
   pip install -r requirements.txt
   ```

## Dataset Preparation

This project operates on the Microsoft COCO benchmark adapted for S-OWOD (Open World Object Detection). Please structure your MS COCO download according to the following layout:

```text
mepu-owod/
└── datasets/
    └── coco/
        ├── annotations/
        ├── train2017/
        └── val2017/
```

Then generate the S-OWOD subsets:
```bash
python prepare_dataset.py
```

## Quick Start & Usage

We provide a comprehensive multi-step script that takes you systematically through initial pseudo-label generation, Weibull probability updates, multi-modal feature fusion, uncertainty filtering, and final detector training.

### Train Multi-Modal MEPU on Task 1

The principal entry point for replicating our evaluation loop on Task 1 is:

```bash
bash script/train_multimodal_mepu_t1.sh
```

**What this script does:**
1. Generates initial pseudo-labels for unknown proposals via FreeSOLO.
2. Updates and distills the Reconstruction Error-based Weibull (REW) models utilizing baseline known instances.
3. Automatically computes comprehensive MULTI-MODAL REW scores (merging CLIP textual semantics and ResNet visual feature maps).
4. Estimates epistemic prediction uncertainty by running MC-Dropout sampling.
5. Performs rigorous intersection/quality/uncertainty checking to generate filtered, high-quality pseudo-labels for training.
6. Self-trains the detection pipeline on standard OWOD evaluation tasks using the verified high-confidence proposals.

### Evaluation

To evaluate an established Multimodal MEPU model state on standard Validation bounds (e.g., Task 1 metrics):

```bash
python train_net.py \
    --eval-only \
    --config config/MEPU-SOWOD/t1/train_multimodal.yaml \
    --num-gpus 1 \
    MODEL.WEIGHTS training_dir/multimodal-mepu/sowod-t1-final/model_final.pth \
    DATASETS.TEST '("sowod_val_t1",)'
```

*Note: Visualized inference outputs and theoretical proofs corresponding to this pipeline are detailed thoroughly within the method's paper documentation and the `methodology.tex` source file.*
