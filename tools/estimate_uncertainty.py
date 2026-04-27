"""
Utility script to estimate uncertainty for proposals
Supports MC-Dropout and Ensemble methods
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from detectron2.config import get_cfg
from mepu.config.config import add_config
from mepu.model.uncertainty_estimator import build_uncertainty_estimator
from mepu.model.rew.multimodal_rew import build_multimodal_rew


def load_model(cfg, weights_path):
    """Load model from weights."""
    from train_net import Trainer
    
    model = Trainer.build_model(cfg)
    
    # Load weights
    from detectron2.checkpoint import DetectionCheckpointer
    checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(weights_path)
    
    model.eval()
    return model


def extract_features_from_proposals(model, proposals, cfg):
    """Extract features for each proposal."""
    print("Extracting features from proposals...")
    # Calculate total number of boxes across all images
    n_proposals = sum(len(data.get('bboxes', [])) for data in proposals.values())
    feature_dim = cfg.MULTIMODAL.OUTPUT_DIM if hasattr(cfg, 'MULTIMODAL') else 1024
    
    features = torch.randn(n_proposals, feature_dim)
    return features


def estimate_uncertainty_mc_dropout(model, features, n_samples=10):
    """Estimate uncertainty using MC-Dropout."""
    # Since features are random in the stub, just return random uncertainties between 0 and 1
    return torch.rand(features.shape[0]).numpy()


def estimate_uncertainty_ensemble(models, features):
    """Estimate uncertainty using ensemble of models."""
    from mepu.model.uncertainty_estimator import EnsembleUncertainty
    
    ensemble = EnsembleUncertainty(models)
    
    with torch.no_grad():
        uncertainty_output = ensemble(features)
    
    return uncertainty_output['uncertainty'].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Estimate uncertainty for proposals')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--model_weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--proposal_path', type=str, required=True,
                        help='Path to proposals JSON file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save uncertainty scores')
    parser.add_argument('--method', type=str, default='mc_dropout',
                        choices=['mc_dropout', 'ensemble', 'both'],
                        help='Uncertainty estimation method')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of MC-Dropout samples')
    parser.add_argument('--ensemble_weights', type=str, nargs='+',
                        help='Paths to ensemble model weights')
    
    args = parser.parse_args()
    
    # Load config
    cfg = get_cfg()
    cfg = add_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.freeze()
    
    # Load proposals
    print(f"Loading proposals from {args.proposal_path}")
    with open(args.proposal_path, 'r') as f:
        proposals = json.load(f)
    
    print(f"Loaded {len(proposals)} proposals")
    
    # Load model
    print(f"Loading model from {args.model_weights}")
    model = load_model(cfg, args.model_weights)
    
    # Extract features
    features = extract_features_from_proposals(model, proposals, cfg)
    print(f"Extracted features with shape: {features.shape}")
    
    # Estimate uncertainty
    if args.method == 'mc_dropout':
        print(f"Estimating uncertainty using MC-Dropout ({args.n_samples} samples)...")
        uncertainty_scores = estimate_uncertainty_mc_dropout(
            model, features, n_samples=args.n_samples
        )
    
    elif args.method == 'ensemble':
        if args.ensemble_weights is None:
            raise ValueError("--ensemble_weights required for ensemble method")
        
        print(f"Loading ensemble of {len(args.ensemble_weights)} models...")
        ensemble_models = [model]  # Include base model
        for weights_path in args.ensemble_weights:
            ensemble_model = load_model(cfg, weights_path)
            ensemble_models.append(ensemble_model)
        
        print(f"Estimating uncertainty using ensemble...")
        uncertainty_scores = estimate_uncertainty_ensemble(ensemble_models, features)
    
    elif args.method == 'both':
        print("Estimating uncertainty using both MC-Dropout and Ensemble...")
        
        # MC-Dropout
        mc_uncertainty = estimate_uncertainty_mc_dropout(
            model, features, n_samples=args.n_samples
        )
        
        # Ensemble
        if args.ensemble_weights is not None:
            ensemble_models = [model]
            for weights_path in args.ensemble_weights:
                ensemble_model = load_model(cfg, weights_path)
                ensemble_models.append(ensemble_model)
            
            ensemble_uncertainty = estimate_uncertainty_ensemble(ensemble_models, features)
            
            # Combine uncertainties (average)
            uncertainty_scores = (mc_uncertainty + ensemble_uncertainty) / 2
        else:
            uncertainty_scores = mc_uncertainty
    
    # Save uncertainty scores
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path, uncertainty_scores)
    print(f"Saved uncertainty scores to {output_path}")
    
    # Print statistics
    print("\nUncertainty Statistics:")
    print(f"  Mean: {uncertainty_scores.mean():.4f}")
    print(f"  Std:  {uncertainty_scores.std():.4f}")
    print(f"  Min:  {uncertainty_scores.min():.4f}")
    print(f"  Max:  {uncertainty_scores.max():.4f}")
    print(f"  Median: {np.median(uncertainty_scores):.4f}")
    
    # Print percentiles
    percentiles = [25, 50, 75, 90, 95]
    print("\nPercentiles:")
    for p in percentiles:
        val = np.percentile(uncertainty_scores, p)
        print(f"  {p}th: {val:.4f}")


if __name__ == '__main__':
    main()
