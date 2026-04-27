"""
CLIP Feature Extractor for Multi-Modal REW
Wraps OpenAI's CLIP model for extracting visual and text features
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    import clip
except ImportError:
    print("CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
    clip = None


class CLIPFeatureExtractor(nn.Module):
    """
    Wrapper for CLIP model to extract visual and text features.
    Supports multiple CLIP architectures (ViT-B/32, ViT-L/14, RN50, etc.)
    """
    
    def __init__(
        self, 
        model_name: str = 'ViT-B/32',
        device: str = 'cuda',
        freeze: bool = True
    ):
        """
        Args:
            model_name: CLIP model variant ('ViT-B/32', 'ViT-L/14', 'RN50', etc.)
            device: Device to load model on
            freeze: Whether to freeze CLIP weights (recommended for efficiency)
        """
        super().__init__()
        
        if clip is None:
            raise ImportError(
                "CLIP is not installed. Install with:\n"
                "pip install git+https://github.com/openai/CLIP.git"
            )
        
        self.model_name = model_name
        self.device = device
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=device)
        
        # Freeze weights if specified
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        
        # Get feature dimensions
        self.visual_dim = self.model.visual.output_dim
        self.text_dim = self.model.transformer.width
        
        print(f"Loaded CLIP model: {model_name}")
        print(f"Visual feature dim: {self.visual_dim}, Text feature dim: {self.text_dim}")
    
    def extract_visual_features(
        self, 
        images: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Extract visual features from images using CLIP visual encoder.
        
        Args:
            images: Tensor of shape (B, C, H, W), already preprocessed
            normalize: Whether to L2-normalize features
            
        Returns:
            Visual features of shape (B, visual_dim)
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            features = self.model.encode_image(images)
            
            if normalize:
                features = features / features.norm(dim=-1, keepdim=True)
                
        return features
    
    def extract_text_features(
        self, 
        texts: List[str],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Extract text features from text descriptions using CLIP text encoder.
        
        Args:
            texts: List of text strings
            normalize: Whether to L2-normalize features
            
        Returns:
            Text features of shape (len(texts), text_dim)
        """
        # Tokenize text
        text_tokens = clip.tokenize(texts).to(self.device)
        
        with torch.no_grad() if not self.training else torch.enable_grad():
            features = self.model.encode_text(text_tokens)
            
            if normalize:
                features = features / features.norm(dim=-1, keepdim=True)
                
        return features
    
    def get_class_text_features(
        self,
        class_names: List[str],
        templates: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Get text features for object classes using prompt templates.
        
        Args:
            class_names: List of class names (e.g., ['person', 'car', 'dog'])
            templates: List of prompt templates. If None, uses default templates.
            
        Returns:
            Text features of shape (num_classes, text_dim)
        """
        if templates is None:
            # Default CLIP prompt templates
            templates = [
                'a photo of a {}.',
                'a photo of the {}.',
                'a photo of one {}.',
                'an image of a {}.',
                'an image of the {}.',
            ]
        
        # Generate prompts for each class
        all_features = []
        for class_name in class_names:
            prompts = [template.format(class_name) for template in templates]
            features = self.extract_text_features(prompts, normalize=True)
            # Average over templates
            class_feature = features.mean(dim=0)
            all_features.append(class_feature)
        
        return torch.stack(all_features)
    
    def compute_similarity(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute cosine similarity between visual and text features.
        
        Args:
            visual_features: Shape (B, visual_dim)
            text_features: Shape (N, text_dim)
            temperature: Temperature scaling factor
            
        Returns:
            Similarity matrix of shape (B, N)
        """
        # Normalize features
        visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = (visual_features @ text_features.T) / temperature
        
        return similarity
    
    def forward(
        self,
        images: torch.Tensor,
        texts: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to extract both visual and text features.
        
        Args:
            images: Tensor of shape (B, C, H, W)
            texts: Optional list of text strings
            
        Returns:
            Dictionary with 'visual' and optionally 'text' features
        """
        output = {}
        
        # Extract visual features
        output['visual'] = self.extract_visual_features(images)
        
        # Extract text features if provided
        if texts is not None:
            output['text'] = self.extract_text_features(texts)
            output['similarity'] = self.compute_similarity(
                output['visual'], 
                output['text']
            )
        
        return output


def build_clip_extractor(cfg) -> CLIPFeatureExtractor:
    """
    Build CLIP feature extractor from config.
    
    Args:
        cfg: Configuration object
        
    Returns:
        CLIPFeatureExtractor instance
    """
    model_name = cfg.MULTIMODAL.CLIP_MODEL if hasattr(cfg, 'MULTIMODAL') else 'ViT-B/32'
    device = cfg.MODEL.DEVICE
    
    return CLIPFeatureExtractor(
        model_name=model_name,
        device=device,
        freeze=True
    )
