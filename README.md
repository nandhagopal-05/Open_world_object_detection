# Open-World Object Detection with Multi-Modal MEPU

> **MEPU-OWOD** is a research framework for **Open-World Object Detection (OWOD)** that can detect both *known* and *unknown* objects — even objects the model has never seen during training.

---

## What Is This Project?

Traditional object detectors only recognize categories they were trained on. When they see an unfamiliar object, they either miss it or wrongly classify it as background. **Open-World Object Detection** solves this by teaching the model to also flag objects it doesn't recognize.

This project extends the **MEPU** (Multi-modal Evidence Per-Unknown) framework by adding:

| Enhancement | Description |
|---|---|
| **Multi-Modal Fusion** | Combines visual features (ResNet) with language features (CLIP) for richer object representations |
| **Weibull Uncertainty Modeling** | Uses statistical Weibull distributions to estimate how confident the model is about unknown object candidates |
| **Uncertainty-Aware Pseudo-Labeling** | Automatically generates training labels for unknown objects, filtered by quality and confidence |

---

## Key Files at a Glance

```
mepu-owod/
│
├── mepu/model/
│   ├── rew/
│   │   ├── multimodal_rew.py       # Multi-modal fusion model (CLIP + ResNet)
│   │   └── fusion.py               # Feature fusion strategies (attention, gating, concat)
│   └── uncertainty_estimator.py    # Weibull-based uncertainty estimation
│
├── tools/
│   ├── gen_pseudo_label_new.py         # Step 1: Generate initial pseudo-labels (FreeSOLO)
│   ├── gen_pseudo_label_uncertainty.py # Step 5: Filter labels using uncertainty scores
│   └── estimate_uncertainty.py         # Step 4: MC-Dropout uncertainty estimation
│
├── script/
│   └── train_multimodal_mepu_t1.sh     # Main end-to-end training script (Task 1)
│
├── config/                             # YAML config files for training & evaluation
├── datasets/                           # Dataset files (COCO / S-OWOD)
├── models/                             # Pre-trained model weights
├── proposals/                          # FreeSOLO region proposals
└── train_net.py                        # Main training/evaluation entry point
```

---

## Installation

Follow these steps in order. This project requires a Linux/Mac environment or WSL on Windows.

### Step 1 — Create a Conda Environment

```bash
conda create -n mepu-multimodal python=3.8
conda activate mepu-multimodal
```

### Step 2 — Install PyTorch (with CUDA support)

```bash
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch
```

> **Requirements:** CUDA 10.2 or higher, GPU with at least 8 GB VRAM recommended.

### Step 3 — Install Detectron2

Detectron2 is Facebook's object detection library that this project is built on top of.

```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..
```

### Step 4 — Install CLIP

CLIP (Contrastive Language-Image Pretraining) provides the semantic language features used in multi-modal fusion.

```bash
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
```

### Step 5 — Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

This project uses **MS COCO** data reorganized into the **S-OWOD** (Superclass Open World Object Detection) benchmark format.

### 1. Download MS COCO

Download the COCO 2017 dataset from [cocodataset.org](https://cocodataset.org/#download) and organize it as shown below:

```
mepu-owod/
└── datasets/
    └── coco/
        ├── annotations/    ← JSON annotation files
        ├── train2017/      ← Training images
        └── val2017/        ← Validation images
```

### 2. Generate S-OWOD Subsets

Run the preparation script to convert COCO into the task-split format used for open-world evaluation:

```bash
python prepare_dataset.py
```

This will create the S-OWOD annotation files under `datasets/sowod/Annotations/`.

---

## Download Pre-trained Models

Download the following files from [Google Drive](https://drive.google.com/drive/folders/1AhFY-aH-ewwukEFlA3QsE5tjGR3lw1j4) and place them in the correct locations:

| File | Save Location | Purpose |
|------|--------------|---------|
| `soco_backbone.pth` | `models/soco_backbone.pth` | SoCo pre-trained backbone |
| `model_final.pth` | `training_dir/rew/model_final.pth` | Pre-trained REW model |
| `proposals_freesolo.json` | `proposals/proposals_freesolo.json` | FreeSOLO region proposals |

---

## Training (Task 1)

Once setup is complete, run the full multi-stage training pipeline with a single command:

```bash
conda activate mepu-multimodal
bash script/train_multimodal_mepu_t1.sh
```

### What Happens Inside the Script

The script runs **12 sequential steps** automatically:

| Step | What It Does |
|------|-------------|
| **1** | Generate initial unknown-object proposals using **FreeSOLO** |
| **2** | Update **Weibull (REW)** models using known-category training images |
| **3** | Extract **multi-modal REW scores** (CLIP language + ResNet visual) |
| **4** | Estimate prediction **uncertainty** via MC-Dropout sampling |
| **5** | Filter proposals and generate **high-quality pseudo-labels** |
| **6** | Train the detector using known objects + filtered pseudo-labels |
| **7** | **Self-training**: Run OLN inference on training set to refine proposals |
| **8** | Re-generate pseudo-labels from the self-trained model |
| **9** | Re-compute REW scores for the refined proposals |
| **10** | Re-estimate uncertainty for the refined proposals |
| **11** | Filter refined proposals with uncertainty thresholds |
| **12** | Final training with the refined pseudo-labels |

> The final model is saved to: `training_dir/multimodal-mepu/sowod-t1-final/model_final.pth`

---

## Evaluation

To evaluate a trained model on the Task 1 validation set:

```bash
python train_net.py \
    --eval-only \
    --config config/MEPU-SOWOD/t1/train_multimodal.yaml \
    --num-gpus 1 \
    MODEL.WEIGHTS training_dir/multimodal-mepu/sowod-t1-final/model_final.pth \
    DATASETS.TEST '("sowod_val_t1",)'
```

The evaluation reports standard OWOD metrics including **Known AP**, **Unknown Recall (U-Recall)**, and **Wilderness Impact (WI)**.

---

## Configuration Reference

Key configuration parameters (set in YAML files under `config/`):

### Multi-Modal Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `MULTIMODAL.CLIP_MODEL` | `ViT-B/32` | CLIP backbone (`ViT-B/32`, `ViT-L/14`, `RN50`) |
| `MULTIMODAL.FUSION_TYPE` | `attention` | How visual and semantic features are merged |
| `MULTIMODAL.VISUAL_WEIGHT` | `0.6` | Weight given to visual (ResNet) REW scores |
| `MULTIMODAL.SEMANTIC_WEIGHT` | `0.4` | Weight given to semantic (CLIP) REW scores |

### Uncertainty Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `UNCERTAINTY.MC_SAMPLES` | `10` | Number of MC-Dropout forward passes |
| `UNCERTAINTY.THRESHOLD` | `0.3` | Maximum uncertainty to accept a proposal |
| `UNCERTAINTY.CALIBRATION` | `temperature` | Calibration method (`temperature`, `platt`, `none`) |

### Pseudo-Label Settings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `PSEUDO_LABEL.UNCERTAINTY_THRESHOLD` | `0.3` | Reject proposals with uncertainty above this |
| `PSEUDO_LABEL.QUALITY_THRESHOLD` | `0.5` | Reject proposals with quality score below this |
| `PSEUDO_LABEL.ACTIVE_LEARNING_BUDGET` | `1000` | Number of uncertain samples to query per round |

---

## Troubleshooting

### CLIP import fails

```bash
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
```

### CUDA Out of Memory

Reduce the batch size in your config YAML:

```yaml
SOLVER:
  IMS_PER_BATCH: 2   # Reduce from 4 to 2
```

### Training is too slow

Try these speed-ups in your config:

- Lower `UNCERTAINTY.MC_SAMPLES` from `10` → `5`
- Use a lighter CLIP model: `RN50` instead of `ViT-L/14`
- Pre-cache CLIP features offline to avoid recomputing them each epoch

---

## Results

### FreeSOLO 

<img src="output_T1\FreeSOLO\Figure_9.png" alt="FreeSOLO" width="50%">

### Refined-pseudo-label 

<img src="output_T1\Refined-pseudo-labels\Figure_5.png" alt="Refined-pseudo-label" width="50%">

### Weibull 

<img src="output_T1\Weibull models of REW\T-1-Figure_5.png" alt="Weibull" width="50%">

## Extending to Tasks 2–4

The S-OWOD benchmark has 4 incremental tasks. To train on Task 2, 3, or 4:

1. Copy `script/train_multimodal_mepu_t1.sh` → e.g., `train_multimodal_mepu_t2.sh`
2. Update the config path: `config/MEPU-SOWOD/t2/train_multimodal.yaml`
3. Update dataset splits: `--data_split t2_train`, `DATASETS.TEST '("sowod_val_t2",)'`
4. Update the output directory: `training_dir/multimodal-mepu/sowod-t2`

---

## License

This project is built upon [MEPU](https://github.com/fredzzhang/mepu) and [Detectron2](https://github.com/facebookresearch/detectron2). Please refer to their respective licenses.
