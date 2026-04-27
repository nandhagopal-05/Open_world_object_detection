"""
Multi-Modal REW (Reconstruction Error-based Weibull Modeling)
Extends the original REW to leverage both visual and semantic information from CLIP
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import scipy.stats

from .clip_wrapper import CLIPFeatureExtractor
from .fusion import MultiModalFusion, CrossModalConsistency
from .distribution_fitter import get_distribution


class MultiModalREW(nn.Module):
    """
    Multi-Modal Reconstruction Error-based Weibull Modeling.
    Combines visual reconstruction error (SoCo) with semantic features (CLIP)
    for improved unknown object detection.
    """
    
    def __init__(
        self,
        soco_backbone: nn.Module,
        clip_model_name: str = 'ViT-B/32',
        fusion_type: str = 'attention',
        output_dim: int = 1024,
        visual_weight: float = 0.6,
        semantic_weight: float = 0.4,
        device: str = 'cuda'
    ):
        """
        Args:
            soco_backbone: Pre-trained SoCo backbone for reconstruction
            clip_model_name: CLIP model variant
            fusion_type: Feature fusion strategy
            output_dim: Dimension of fused features
            visual_weight: Weight for visual REW score
            semantic_weight: Weight for semantic REW score
            device: Device to run on
        """
        super().__init__()
        
        self.soco_backbone = soco_backbone
        self.device = device
        self.visual_weight = visual_weight
        self.semantic_weight = semantic_weight
        
        # Initialize CLIP extractor
        self.clip_extractor = CLIPFeatureExtractor(
            model_name=clip_model_name,
            device=device,
            freeze=True
        )
        
        # Initialize fusion module
        self.fusion_module = MultiModalFusion(
            soco_dim=2048,  # SoCo output dimension
            clip_visual_dim=self.clip_extractor.visual_dim,
            clip_text_dim=self.clip_extractor.text_dim,
            fusion_type=fusion_type,
            output_dim=output_dim
        )
        
        # Cross-modal consistency module
        self.consistency_module = CrossModalConsistency(
            feature_dim=output_dim
        )
        
        # Weibull distribution parameters (fitted during training)
        self.visual_weibull_params = None  # For visual reconstruction error
        self.semantic_weibull_params = None  # For semantic features
        self.fused_weibull_params = None  # For fused features
        
        # Background Weibull parameters
        self.visual_bg_weibull_params = None
        self.semantic_bg_weibull_params = None
        self.fused_bg_weibull_params = None
        
        # Class text features (cached)
        self.class_text_features = None
        self.class_names = None
        
    def set_class_names(self, class_names: List[str]):
        """
        Set known class names and compute their text features.
        
        Args:
            class_names: List of known class names
        """
        self.class_names = class_names
        self.class_text_features = self.clip_extractor.get_class_text_features(
            class_names
        )
        
    def extract_features(
        self,
        images: torch.Tensor,
        roi_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract multi-modal features from images.
        
        Args:
            images: Input images, shape (B, 3, H, W)
            roi_features: Optional ROI features from SoCo backbone
            
        Returns:
            Dictionary containing:
                - soco_features: Visual features from SoCo
                - clip_visual_features: Visual features from CLIP
                - clip_text_features: Text features for known classes
                - fused_features: Fused multi-modal features
                - reconstruction_error: Reconstruction error from SoCo
        """
        batch_size = images.shape[0]
        
        # Extract SoCo features and reconstruction error
        if roi_features is None:
            # Full image features
            soco_output = self.soco_backbone(images)
            soco_features = soco_output['features']
            reconstruction_error = soco_output.get('reconstruction_error', None)
        else:
            # Use provided ROI features
            soco_features = roi_features
            reconstruction_error = None
        
        # Extract CLIP visual features
        # Resize images to CLIP input size (224x224)
        clip_images = torch.nn.functional.interpolate(
            images, size=(224, 224), mode='bilinear', align_corners=False
        )
        clip_visual_features = self.clip_extractor.extract_visual_features(clip_images)
        
        # Get text features for known classes
        clip_text_features = None
        if self.class_text_features is not None:
            # Expand to batch size
            clip_text_features = self.class_text_features.unsqueeze(0).expand(
                batch_size, -1, -1
            ).mean(dim=1)  # Average over classes for now
        
        # Fuse features
        fusion_output = self.fusion_module(
            soco_features=soco_features,
            clip_visual_features=clip_visual_features,
            clip_text_features=clip_text_features
        )
        
        return {
            'soco_features': soco_features,
            'clip_visual_features': clip_visual_features,
            'clip_text_features': clip_text_features,
            'fused_features': fusion_output['fused'],
            'fusion_weights': fusion_output['weights'],
            'reconstruction_error': reconstruction_error
        }
    
    def fit_weibull_distributions(
        self,
        known_features: Dict[str, torch.Tensor],
        background_features: Dict[str, torch.Tensor]
    ):
        """
        Fit Weibull distributions for known objects and background.
        
        Args:
            known_features: Dictionary with features from known objects
                - 'visual': Visual reconstruction errors
                - 'semantic': Semantic features
                - 'fused': Fused features
            background_features: Dictionary with features from background regions
        """
        # Convert to numpy
        known_visual = known_features['visual'].cpu().numpy()
        known_semantic = known_features['semantic'].cpu().numpy()
        known_fused = known_features['fused'].cpu().numpy()
        
        bg_visual = background_features['visual'].cpu().numpy()
        bg_semantic = background_features['semantic'].cpu().numpy()
        bg_fused = background_features['fused'].cpu().numpy()
        
        # Fit Weibull distributions for visual (reconstruction error)
        self.visual_weibull_params = scipy.stats.exponweib.fit(known_visual)
        self.visual_bg_weibull_params = scipy.stats.exponweib.fit(bg_visual)
        
        # Fit Weibull distributions for semantic features
        # Use L2 norm as the scalar value for Weibull fitting
        semantic_norms_known = np.linalg.norm(known_semantic, axis=1)
        semantic_norms_bg = np.linalg.norm(bg_semantic, axis=1)
        
        self.semantic_weibull_params = scipy.stats.exponweib.fit(semantic_norms_known)
        self.semantic_bg_weibull_params = scipy.stats.exponweib.fit(semantic_norms_bg)
        
        # Fit Weibull distributions for fused features
        fused_norms_known = np.linalg.norm(known_fused, axis=1)
        fused_norms_bg = np.linalg.norm(bg_fused, axis=1)
        
        self.fused_weibull_params = scipy.stats.exponweib.fit(fused_norms_known)
        self.fused_bg_weibull_params = scipy.stats.exponweib.fit(fused_norms_bg)
        
        print("Fitted Weibull distributions for multi-modal features")
        
    def compute_rew_scores(
        self,
        features: Dict[str, torch.Tensor],
        use_fused: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute REW uncertainty scores for given features.
        
        Args:
            features: Dictionary with multi-modal features
            use_fused: Whether to use fused features or combine scores
            
        Returns:
            Dictionary with REW scores:
                - visual_score: Score from visual REW
                - semantic_score: Score from semantic REW
                - fused_score: Score from fused features
                - combined_score: Weighted combination
        """
        if self.visual_weibull_params is None:
            raise ValueError("Weibull distributions not fitted. Call fit_weibull_distributions first.")
        
        batch_size = features['soco_features'].shape[0]
        
        # Visual REW score (reconstruction error based)
        if features['reconstruction_error'] is not None:
            visual_errors = features['reconstruction_error'].cpu().numpy()
        else:
            # Fallback: use feature norm as proxy
            visual_errors = torch.norm(features['soco_features'], dim=-1).cpu().numpy()
        
        visual_scores = self._compute_weibull_score(
            visual_errors,
            self.visual_weibull_params,
            self.visual_bg_weibull_params
        )
        
        # Semantic REW score
        semantic_features = features['clip_visual_features'].cpu().numpy()
        semantic_norms = np.linalg.norm(semantic_features, axis=1)
        
        semantic_scores = self._compute_weibull_score(
            semantic_norms,
            self.semantic_weibull_params,
            self.semantic_bg_weibull_params
        )
        
        # Fused REW score
        fused_features = features['fused_features'].cpu().numpy()
        fused_norms = np.linalg.norm(fused_features, axis=1)
        
        fused_scores = self._compute_weibull_score(
            fused_norms,
            self.fused_weibull_params,
            self.fused_bg_weibull_params
        )
        
        # Combined score (weighted)
        if use_fused:
            combined_scores = fused_scores
        else:
            combined_scores = (
                self.visual_weight * visual_scores +
                self.semantic_weight * semantic_scores
            )
        
        return {
            'visual_score': torch.from_numpy(visual_scores).to(self.device),
            'semantic_score': torch.from_numpy(semantic_scores).to(self.device),
            'fused_score': torch.from_numpy(fused_scores).to(self.device),
            'combined_score': torch.from_numpy(combined_scores).to(self.device)
        }
    
    def _compute_weibull_score(
        self,
        values: np.ndarray,
        known_params: Tuple,
        bg_params: Tuple
    ) -> np.ndarray:
        """
        Compute Weibull-based uncertainty score.
        Higher score = more likely to be unknown.
        
        Args:
            values: Input values to score
            known_params: Weibull parameters for known objects
            bg_params: Weibull parameters for background
            
        Returns:
            Uncertainty scores (0-1 range, higher = more uncertain/unknown)
        """
        # Compute PDF values
        known_pdf = scipy.stats.exponweib.pdf(values, *known_params)
        bg_pdf = scipy.stats.exponweib.pdf(values, *bg_params)
        
        # Uncertainty score: ratio of background to known likelihood
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        scores = bg_pdf / (known_pdf + epsilon)
        
        # Normalize to [0, 1] range using sigmoid
        scores = 1 / (1 + np.exp(-scores))
        
        return scores
    
    def forward(
        self,
        images: torch.Tensor,
        roi_features: Optional[torch.Tensor] = None,
        compute_scores: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-modal REW.
        
        Args:
            images: Input images
            roi_features: Optional ROI features
            compute_scores: Whether to compute REW scores
            
        Returns:
            Dictionary with features and optionally REW scores
        """
        # Extract features
        features = self.extract_features(images, roi_features)
        
        # Compute REW scores if requested and Weibull is fitted
        if compute_scores and self.visual_weibull_params is not None:
            scores = self.compute_rew_scores(features)
            features.update(scores)
        
        return features


def build_multimodal_rew(cfg, soco_backbone) -> MultiModalREW:
    """
    Build Multi-Modal REW from config.
    
    Args:
        cfg: Configuration object
        soco_backbone: Pre-trained SoCo backbone
        
    Returns:
        MultiModalREW instance
    """
    return MultiModalREW(
        soco_backbone=soco_backbone,
        clip_model_name=cfg.MULTIMODAL.CLIP_MODEL,
        fusion_type=cfg.MULTIMODAL.FUSION_TYPE,
        output_dim=cfg.MULTIMODAL.OUTPUT_DIM,
        visual_weight=cfg.MULTIMODAL.VISUAL_WEIGHT,
        semantic_weight=cfg.MULTIMODAL.SEMANTIC_WEIGHT,
        device=cfg.MODEL.DEVICE
    )
