# Installation and Setup Guide for Multi-Modal MEPU

## Prerequisites
- Python 3.8+
- CUDA 10.2 or higher
- PyTorch 1.12.0

## Installation Steps

### 1. Create Conda Environment
```bash
conda create -n mepu-multimodal python=3.8
conda activate mepu-multimodal
```

### 2. Install PyTorch
```bash
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch
```

### 3. Install Detectron2
```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..
```

### 4. Install CLIP
```bash
pip install git+https://github.com/openai/CLIP.git
```

### 5. Install Other Dependencies
```bash
pip install -r requirements.txt
```

## Dataset Preparation

### 1. Download MS COCO Dataset
Download and organize as:
```
mepu-owod/
└── datasets/
    └── coco/
        ├── annotations/
        ├── train2017/
        └── val2017/
```

### 2. Prepare S-OWOD Dataset
```bash
python prepare_dataset.py
```

## Download Pre-trained Models

Download from [Google Drive](https://drive.google.com/drive/folders/1AhFY-aH-ewwukEFlA3QsE5tjGR3lw1j4):
1. SoCo backbone → `models/soco_backbone.pth`
2. Pre-trained REW → `training_dir/rew/model_final.pth`
3. FreeSOLO proposals → `proposals/proposals_freesolo.json`

## Quick Start

### Train Multi-Modal MEPU on Task 1
```bash
bash script/train_multimodal_mepu_t1.sh
```

### Evaluate
```bash
python train_net.py \
    --eval-only \
    --config config/MEPU-SOWOD/t1/train_multimodal.yaml \
    --num-gpus 1 \
    MODEL.WEIGHTS training_dir/multimodal-mepu/sowod-t1-final/model_final.pth \
    DATASETS.TEST '("sowod_val_t1",)'
```

## Key Configuration Parameters

### Multi-Modal Settings
- `MULTIMODAL.CLIP_MODEL`: CLIP variant (ViT-B/32, ViT-L/14, RN50)
- `MULTIMODAL.FUSION_TYPE`: Fusion strategy (attention, gating, concat, adaptive)
- `MULTIMODAL.VISUAL_WEIGHT`: Weight for visual REW (default: 0.6)
- `MULTIMODAL.SEMANTIC_WEIGHT`: Weight for semantic REW (default: 0.4)

### Uncertainty Settings
- `UNCERTAINTY.MC_SAMPLES`: Number of MC-Dropout samples (default: 10)
- `UNCERTAINTY.THRESHOLD`: Uncertainty threshold for filtering (default: 0.3)
- `UNCERTAINTY.CALIBRATION`: Calibration method (temperature, platt, none)

### Pseudo-Label Settings
- `PSEUDO_LABEL.UNCERTAINTY_THRESHOLD`: Max uncertainty to accept (default: 0.3)
- `PSEUDO_LABEL.QUALITY_THRESHOLD`: Min quality score (default: 0.5)
- `PSEUDO_LABEL.ACTIVE_LEARNING_BUDGET`: Samples for active learning (default: 1000)

## Troubleshooting

### CLIP Import Error
```bash
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
```

### CUDA Out of Memory
Reduce batch size in config:
```yaml
SOLVER:
  IMS_PER_BATCH: 2  # Reduce from 4
```

### Slow Training
- Reduce `UNCERTAINTY.MC_SAMPLES` to 5
- Use smaller CLIP model (RN50 instead of ViT-L/14)
- Cache CLIP features offline

## Next Steps

1. Train on Task 1 using the provided script
2. Evaluate and analyze results
3. Run ablation studies (see implementation_plan.md)
4. Extend to Tasks 2-4
5. Compare against baseline MEPU
