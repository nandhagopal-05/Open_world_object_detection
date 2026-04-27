from detectron2.config import CfgNode as CN

def add_config(cfg):
    _C = cfg
    _C.OPENSET = CN()
    _C.OPENSET.ENABLE_REW = False
    
    _C.OPENSET.REW = CN()
    _C.OPENSET.REW.ENABLED = False  # Enable REW scoring
    _C.OPENSET.REW.MULTIMODAL = False  # Use multi-modal REW
    _C.OPENSET.REW.AE_INTER = [64, 32, 16, 8, 4]
    _C.OPENSET.REW.FITER_PERCENT = 0.2
    _C.OPENSET.REW.HARD_THR = False
    _C.OPENSET.REW.ENABLE_SL = True
    _C.OPENSET.REW.GAMMA = 2.0
    _C.OPENSET.REW.ALPHA = 1.0
    _C.OPENSET.REW.SAMPLING_ITERS = 2500
    _C.OPENSET.REW.NUM_SAMPLES = 160000
    _C.OPENSET.REW.UPDATE_WEIBULL = False


    _C.OPENSET.ENABLE_OLN = False
    _C.OPENSET.OLN_INFERENCE = False
    _C.OPENSET.INFERENCE_SELT_TRAIN = False
    _C.OPENSET.OLN = CN()
    _C.OPENSET.OLN.IOU_LABELS = [-1, 1]
    _C.OPENSET.OLN.IOU_THRESHOLDS = [0.3]
    _C.OPENSET.OLN.NMS_THRESH = 0.7
    _C.OPENSET.OLN.POSITIVE_FRACTION = 1.0
    _C.OPENSET.OLN.POST_NMS_TOPK_TEST = 10000
    _C.OPENSET.OLN.POST_NMS_TOPK_TRAIN = 10000
    _C.OPENSET.OLN.PRE_NMS_TOPK_TEST = 2000
    _C.OPENSET.OLN.PRE_NMS_TOPK_TRAIN = 2000
    _C.OPENSET.OLN.BATCH_SIZE_PER_IMAGE = 256
    
    _C.OPENSET.NUM_KNOWN_CLASSES = 20
    _C.OPENSET.NUM_PREV_KNOWN_CLASSES = 0
    _C.OPENSET.EVAL_UNKNOWN = False
    _C.OPENSET.OUTPUT_PATH_REW = "./"
    _C.OPENSET.CALIBRATE = True
    _C.OPENSET.CALIBRATE_WEIGHT = 5.0
    _C.OPENSET.FILTER_THRESH = 0.5
    
    # Pseudo-Label Configuration
    _C.OPENSET.PSEUDO_LABEL = CN()
    _C.OPENSET.PSEUDO_LABEL.UNCERTAINTY_FILTERING = False  # Filter by uncertainty
    _C.OPENSET.PSEUDO_LABEL.QUALITY_WEIGHTING = False  # Weight samples by quality score
    _C.OPENSET.PSEUDO_LABEL.ACTIVE_LEARNING = False  # Use active learning selection
    _C.OPENSET.PSEUDO_LABEL.ACTIVE_LEARNING_BUDGET = 1000  # Number of samples for active learning
    _C.OPENSET.PSEUDO_LABEL.UNCERTAINTY_THRESHOLD = 0.3  # Max uncertainty to accept
    _C.OPENSET.PSEUDO_LABEL.CONFIDENCE_THRESHOLD = 0.5  # Min confidence to accept
    _C.OPENSET.PSEUDO_LABEL.REW_THRESHOLD = 0.6  # Min REW score to accept
    _C.OPENSET.PSEUDO_LABEL.QUALITY_THRESHOLD = 0.5  # Min overall quality score
    
    # Multi-Modal Configuration
    _C.MULTIMODAL = CN()
    _C.MULTIMODAL.ENABLED = False
    _C.MULTIMODAL.CLIP_MODEL = "ViT-B/32"  # CLIP model variant
    _C.MULTIMODAL.FUSION_TYPE = "attention"  # Fusion strategy: concat, attention, gating, adaptive
    _C.MULTIMODAL.OUTPUT_DIM = 1024  # Dimension of fused features
    _C.MULTIMODAL.VISUAL_WEIGHT = 0.6  # Weight for visual REW score
    _C.MULTIMODAL.SEMANTIC_WEIGHT = 0.4  # Weight for semantic REW score
    
    # Uncertainty Estimation Configuration
    _C.UNCERTAINTY = CN()
    _C.UNCERTAINTY.ENABLED = False
    _C.UNCERTAINTY.MC_DROPOUT = True  # Enable Monte Carlo Dropout
    _C.UNCERTAINTY.MC_SAMPLES = 10  # Number of MC-Dropout samples
    _C.UNCERTAINTY.ENSEMBLE_SIZE = 1  # Number of models in ensemble
    _C.UNCERTAINTY.CALIBRATION = "temperature"  # Calibration method: temperature, platt, or none
    _C.UNCERTAINTY.THRESHOLD = 0.3  # Uncertainty threshold for filtering
    _C.UNCERTAINTY.DYNAMIC_THRESHOLD = False  # Use adaptive thresholding
    _C.UNCERTAINTY.THRESHOLD_METHOD = "percentile"  # Method for dynamic threshold
    _C.UNCERTAINTY.THRESHOLD_PERCENTILE = 0.7  # Percentile for threshold

    return _C
    
   
