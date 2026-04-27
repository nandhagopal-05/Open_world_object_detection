"""
Multi-Modal Feature Fusion Module
Combines visual features from SoCo and CLIP using various fusion strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Literal


class MultiModalFusion(nn.Module):
    """
    Fuses visual features from multiple sources (SoCo, CLIP visual, CLIP text).
    Supports multiple fusion strategies: concatenation, attention, gating.
    """
    
    def __init__(
        self,
        soco_dim: int = 2048,
        clip_visual_dim: int = 512,
        clip_text_dim: int = 512,
        fusion_type: Literal['concat', 'attention', 'gating', 'adaptive'] = 'attention',
        output_dim: int = 1024,
        dropout: float = 0.1
    ):
        """
        Args:
            soco_dim: Dimension of SoCo features
            clip_visual_dim: Dimension of CLIP visual features
            clip_text_dim: Dimension of CLIP text features
            fusion_type: Type of fusion strategy
            output_dim: Output feature dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.soco_dim = soco_dim
        self.clip_visual_dim = clip_visual_dim
        self.clip_text_dim = clip_text_dim
        self.fusion_type = fusion_type
        self.output_dim = output_dim
        
        # Projection layers to align dimensions
        self.soco_proj = nn.Linear(soco_dim, output_dim)
        self.clip_visual_proj = nn.Linear(clip_visual_dim, output_dim)
        self.clip_text_proj = nn.Linear(clip_text_dim, output_dim)
        
        # Fusion-specific modules
        if fusion_type == 'concat':
            # Simple concatenation + projection
            self.fusion_proj = nn.Sequential(
                nn.Linear(output_dim * 3, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        elif fusion_type == 'attention':
            # Multi-head attention fusion
            self.attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.norm = nn.LayerNorm(output_dim)
            
        elif fusion_type == 'gating':
            # Gated fusion with learnable weights
            self.gate = nn.Sequential(
                nn.Linear(output_dim * 3, 3),
                nn.Softmax(dim=-1)
            )
            
        elif fusion_type == 'adaptive':
            # Adaptive fusion with context-dependent weights
            self.context_encoder = nn.Sequential(
                nn.Linear(output_dim * 3, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, 3),
                nn.Softmax(dim=-1)
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        soco_features: torch.Tensor,
        clip_visual_features: torch.Tensor,
        clip_text_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse multi-modal features.
        
        Args:
            soco_features: Shape (B, soco_dim)
            clip_visual_features: Shape (B, clip_visual_dim)
            clip_text_features: Shape (B, clip_text_dim) or None
            
        Returns:
            Dictionary with 'fused' features and fusion weights
        """
        batch_size = soco_features.shape[0]
        
        # Project all features to same dimension
        soco_proj = self.soco_proj(soco_features)  # (B, output_dim)
        clip_visual_proj = self.clip_visual_proj(clip_visual_features)  # (B, output_dim)
        
        # Handle optional text features
        if clip_text_features is not None:
            clip_text_proj = self.clip_text_proj(clip_text_features)  # (B, output_dim)
        else:
            # Use zero features if text not provided
            clip_text_proj = torch.zeros_like(soco_proj)
        
        # Apply fusion strategy
        if self.fusion_type == 'concat':
            # Concatenate and project
            concat_features = torch.cat([soco_proj, clip_visual_proj, clip_text_proj], dim=-1)
            fused = self.fusion_proj(concat_features)
            weights = None
            
        elif self.fusion_type == 'attention':
            # Stack features for attention
            # Query: SoCo, Key/Value: CLIP visual and text
            query = soco_proj.unsqueeze(1)  # (B, 1, output_dim)
            key_value = torch.stack([clip_visual_proj, clip_text_proj], dim=1)  # (B, 2, output_dim)
            
            # Apply attention
            attn_output, attn_weights = self.attention(query, key_value, key_value)
            fused = self.norm(soco_proj + attn_output.squeeze(1))
            weights = attn_weights.squeeze(1)  # (B, 2)
            
        elif self.fusion_type == 'gating':
            # Compute gating weights
            concat_features = torch.cat([soco_proj, clip_visual_proj, clip_text_proj], dim=-1)
            gate_weights = self.gate(concat_features)  # (B, 3)
            
            # Weighted combination
            fused = (
                gate_weights[:, 0:1] * soco_proj +
                gate_weights[:, 1:2] * clip_visual_proj +
                gate_weights[:, 2:3] * clip_text_proj
            )
            weights = gate_weights
            
        elif self.fusion_type == 'adaptive':
            # Context-dependent adaptive fusion
            concat_features = torch.cat([soco_proj, clip_visual_proj, clip_text_proj], dim=-1)
            adaptive_weights = self.context_encoder(concat_features)  # (B, 3)
            
            # Weighted combination
            fused = (
                adaptive_weights[:, 0:1] * soco_proj +
                adaptive_weights[:, 1:2] * clip_visual_proj +
                adaptive_weights[:, 2:3] * clip_text_proj
            )
            weights = adaptive_weights
        
        # Apply dropout
        fused = self.dropout(fused)
        
        return {
            'fused': fused,
            'weights': weights,
            'soco_proj': soco_proj,
            'clip_visual_proj': clip_visual_proj,
            'clip_text_proj': clip_text_proj
        }


class CrossModalConsistency(nn.Module):
    """
    Enforces consistency between visual and semantic representations.
    Uses contrastive learning to align features from different modalities.
    """
    
    def __init__(
        self,
        feature_dim: int = 1024,
        temperature: float = 0.07
    ):
        """
        Args:
            feature_dim: Dimension of features
            temperature: Temperature for contrastive loss
        """
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        visual_features: torch.Tensor,
        semantic_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cross-modal consistency loss.
        
        Args:
            visual_features: Shape (B, feature_dim)
            semantic_features: Shape (B, feature_dim)
            
        Returns:
            Dictionary with consistency loss and similarity matrix
        """
        # Normalize features
        visual_features = F.normalize(visual_features, dim=-1)
        semantic_features = F.normalize(semantic_features, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(visual_features, semantic_features.T) / self.temperature
        
        # Contrastive loss (InfoNCE)
        batch_size = visual_features.shape[0]
        labels = torch.arange(batch_size, device=visual_features.device)
        
        loss_v2s = F.cross_entropy(similarity, labels)
        loss_s2v = F.cross_entropy(similarity.T, labels)
        
        consistency_loss = (loss_v2s + loss_s2v) / 2
        
        return {
            'consistency_loss': consistency_loss,
            'similarity': similarity,
            'visual_to_semantic_acc': (similarity.argmax(dim=1) == labels).float().mean(),
            'semantic_to_visual_acc': (similarity.T.argmax(dim=1) == labels).float().mean()
        }


def build_fusion_module(cfg) -> MultiModalFusion:
    """
    Build multi-modal fusion module from config.
    
    Args:
        cfg: Configuration object
        
    Returns:
        MultiModalFusion instance
    """
    fusion_type = cfg.MULTIMODAL.FUSION_TYPE if hasattr(cfg.MULTIMODAL, 'FUSION_TYPE') else 'attention'
    output_dim = cfg.MULTIMODAL.OUTPUT_DIM if hasattr(cfg.MULTIMODAL, 'OUTPUT_DIM') else 1024
    
    return MultiModalFusion(
        soco_dim=2048,  # Default SoCo dimension
        clip_visual_dim=512,  # Default CLIP ViT-B/32 dimension
        clip_text_dim=512,
        fusion_type=fusion_type,
        output_dim=output_dim
    )
