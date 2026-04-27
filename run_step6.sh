source /root/miniconda3/etc/profile.d/conda.sh
conda activate best
cd '/mnt/e/New folder/mepu-owod/'
python train_net.py \
    --resume \
    --dist-url auto \
    --num-gpus 1 \
    --config config/MEPU-SOWOD/t1/train_multimodal.yaml \
    DATASETS.TRAIN '("sowod_train_t1_multimodal",)' \
    OUTPUT_DIR training_dir/multimodal-mepu/sowod-t1-train \
    OPENSET.REW.GAMMA 4.0 \
    MULTIMODAL.ENABLED True \
    UNCERTAINTY.ENABLED True
