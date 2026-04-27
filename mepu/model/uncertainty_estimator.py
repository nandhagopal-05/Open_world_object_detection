"""
Uncertainty Estimation Module for Multi-Modal REW
Provides multiple uncertainty quantification methods:
1. Monte Carlo Dropout
2. Ensemble-based uncertainty
3. Confidence calibration (Temperature Scaling, Platt Scaling)
4. Dynamic threshold adjustment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Callable
import numpy as np
from scipy.optimize import minimize


class MCDropoutUncertainty(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.
    Performs multiple forward passes with dropout enabled to estimate uncertainty.
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 10,
        dropout_rate: float = 0.1
    ):
        """
        Args:
            model: Model to apply MC-Dropout on
            n_samples: Number of forward passes
            dropout_rate: Dropout probability
        """
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
        
        # Add dropout layers if not present
        self._add_dropout_layers()
    
    def _add_dropout_layers(self):
        """Add dropout layers to the model if not already present."""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                # Check if dropout already follows this layer
                # For simplicity, we'll enable dropout in training mode
                pass
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_samples: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Perform MC-Dropout inference.
        
        Args:
            x: Input tensor
            return_all_samples: Whether to return all samples or just statistics
            
        Returns:
            Dictionary with mean prediction, variance (uncertainty), and optionally all samples
        """
        # Enable dropout
        self.model.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(x)
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions)  # (n_samples, batch_size, ...)
        
        # Compute statistics
        mean_pred = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        std = predictions.std(dim=0)
        
        # Uncertainty score (normalized variance)
        uncertainty = variance.mean(dim=-1) if variance.dim() > 1 else variance
        
        output = {
            'mean': mean_pred,
            'variance': variance,
            'std': std,
            'uncertainty': uncertainty
        }
        
        if return_all_samples:
            output['all_samples'] = predictions
        
        # Restore model to eval mode
        self.model.eval()
        
        return output


class EnsembleUncertainty(nn.Module):
    """
    Ensemble-based uncertainty estimation.
    Uses multiple independently trained models to estimate uncertainty.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        aggregation: str = 'mean'
    ):
        """
        Args:
            models: List of models in the ensemble
            aggregation: How to aggregate predictions ('mean', 'median', 'vote')
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.aggregation = aggregation
        self.n_models = len(models)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_predictions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Perform ensemble inference.
        
        Args:
            x: Input tensor
            return_all_predictions: Whether to return predictions from all models
            
        Returns:
            Dictionary with aggregated prediction and uncertainty (disagreement)
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                pred = model(x)
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions)  # (n_models, batch_size, ...)
        
        # Aggregate predictions
        if self.aggregation == 'mean':
            aggregated = predictions.mean(dim=0)
        elif self.aggregation == 'median':
            aggregated = predictions.median(dim=0)[0]
        else:
            aggregated = predictions.mean(dim=0)  # Default to mean
        
        # Compute disagreement as uncertainty
        # Use variance or pairwise disagreement
        variance = predictions.var(dim=0)
        disagreement = variance.mean(dim=-1) if variance.dim() > 1 else variance
        
        # Alternative: pairwise disagreement
        pairwise_diff = []
        for i in range(self.n_models):
            for j in range(i + 1, self.n_models):
                diff = (predictions[i] - predictions[j]).abs().mean(dim=-1)
                pairwise_diff.append(diff)
        
        if pairwise_diff:
            pairwise_disagreement = torch.stack(pairwise_diff).mean(dim=0)
        else:
            pairwise_disagreement = disagreement
        
        output = {
            'prediction': aggregated,
            'variance': variance,
            'disagreement': disagreement,
            'pairwise_disagreement': pairwise_disagreement,
            'uncertainty': pairwise_disagreement  # Use pairwise as main uncertainty
        }
        
        if return_all_predictions:
            output['all_predictions'] = predictions
        
        return output


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for confidence calibration.
    Learns a single temperature parameter to calibrate model confidence.
    """
    
    def __init__(self, initial_temperature: float = 1.0):
        """
        Args:
            initial_temperature: Initial temperature value
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Model logits
            
        Returns:
            Temperature-scaled logits
        """
        return logits / self.temperature
    
    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ):
        """
        Calibrate temperature on validation set.
        
        Args:
            logits: Validation logits
            labels: Validation labels
            lr: Learning rate
            max_iter: Maximum iterations
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        print(f"Calibrated temperature: {self.temperature.item():.4f}")


class PlattScaling(nn.Module):
    """
    Platt Scaling for binary confidence calibration.
    Fits a logistic regression model to calibrate probabilities.
    """
    
    def __init__(self):
        """Initialize Platt scaling parameters."""
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))
    
    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply Platt scaling to scores.
        
        Args:
            scores: Raw scores
            
        Returns:
            Calibrated probabilities
        """
        return torch.sigmoid(self.a * scores + self.b)
    
    def calibrate(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100
    ):
        """
        Calibrate Platt scaling parameters.
        
        Args:
            scores: Validation scores
            labels: Binary labels (0 or 1)
            lr: Learning rate
            max_iter: Maximum iterations
        """
        optimizer = torch.optim.Adam([self.a, self.b], lr=lr)
        
        for _ in range(max_iter):
            optimizer.zero_grad()
            calibrated_probs = self.forward(scores)
            loss = F.binary_cross_entropy(calibrated_probs, labels.float())
            loss.backward()
            optimizer.step()
        
        print(f"Calibrated Platt parameters: a={self.a.item():.4f}, b={self.b.item():.4f}")


class DynamicThreshold(nn.Module):
    """
    Dynamic threshold adjustment based on uncertainty distribution.
    Adapts thresholds per-task or per-class.
    """
    
    def __init__(
        self,
        initial_threshold: float = 0.5,
        adaptation_rate: float = 0.1
    ):
        """
        Args:
            initial_threshold: Initial threshold value
            adaptation_rate: Rate of threshold adaptation
        """
        super().__init__()
        self.threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.uncertainty_history = []
    
    def update_threshold(
        self,
        uncertainties: torch.Tensor,
        percentile: float = 0.7
    ):
        """
        Update threshold based on uncertainty distribution.
        
        Args:
            uncertainties: Uncertainty scores
            percentile: Percentile to use as threshold (0-1)
        """
        # Compute percentile-based threshold
        new_threshold = torch.quantile(uncertainties, percentile).item()
        
        # Smooth update
        self.threshold = (
            (1 - self.adaptation_rate) * self.threshold +
            self.adaptation_rate * new_threshold
        )
        
        # Store history
        self.uncertainty_history.append(uncertainties.mean().item())
        
        return self.threshold
    
    def get_adaptive_threshold(
        self,
        uncertainties: torch.Tensor,
        method: str = 'percentile',
        **kwargs
    ) -> float:
        """
        Get adaptive threshold based on current uncertainty distribution.
        
        Args:
            uncertainties: Current uncertainty scores
            method: Thresholding method ('percentile', 'otsu', 'mean_std')
            **kwargs: Additional arguments for specific methods
            
        Returns:
            Adaptive threshold value
        """
        if method == 'percentile':
            percentile = kwargs.get('percentile', 0.7)
            threshold = torch.quantile(uncertainties, percentile).item()
        
        elif method == 'otsu':
            # Otsu's method for automatic thresholding
            threshold = self._otsu_threshold(uncertainties)
        
        elif method == 'mean_std':
            # Mean + k * std
            k = kwargs.get('k', 1.0)
            threshold = uncertainties.mean().item() + k * uncertainties.std().item()
        
        else:
            threshold = self.threshold
        
        return threshold
    
    def _otsu_threshold(self, uncertainties: torch.Tensor) -> float:
        """
        Compute Otsu's threshold for binary classification.
        
        Args:
            uncertainties: Uncertainty scores
            
        Returns:
            Optimal threshold
        """
        # Convert to numpy for easier computation
        values = uncertainties.cpu().numpy()
        
        # Compute histogram
        hist, bin_edges = np.histogram(values, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Normalize histogram
        hist = hist.astype(float) / hist.sum()
        
        # Compute cumulative sums
        weight1 = np.cumsum(hist)
        weight2 = 1 - weight1
        
        # Compute cumulative means
        mean1 = np.cumsum(hist * bin_centers) / (weight1 + 1e-10)
        mean2 = (np.sum(hist * bin_centers) - np.cumsum(hist * bin_centers)) / (weight2 + 1e-10)
        
        # Compute between-class variance
        variance = weight1 * weight2 * (mean1 - mean2) ** 2
        
        # Find threshold that maximizes variance
        idx = np.argmax(variance)
        threshold = bin_centers[idx]
        
        return float(threshold)


class UncertaintyEstimator(nn.Module):
    """
    Unified uncertainty estimation module combining multiple methods.
    """
    
    def __init__(
        self,
        model: nn.Module,
        methods: List[str] = ['mc_dropout', 'ensemble'],
        mc_samples: int = 10,
        ensemble_models: Optional[List[nn.Module]] = None,
        calibration: str = 'temperature'
    ):
        """
        Args:
            model: Base model
            methods: List of uncertainty methods to use
            mc_samples: Number of MC-Dropout samples
            ensemble_models: List of ensemble models
            calibration: Calibration method ('temperature', 'platt', None)
        """
        super().__init__()
        
        self.methods = methods
        self.model = model
        
        # Initialize uncertainty estimators
        if 'mc_dropout' in methods:
            self.mc_dropout = MCDropoutUncertainty(model, n_samples=mc_samples)
        
        if 'ensemble' in methods and ensemble_models is not None:
            self.ensemble = EnsembleUncertainty(ensemble_models)
        
        # Initialize calibration
        if calibration == 'temperature':
            self.calibrator = TemperatureScaling()
        elif calibration == 'platt':
            self.calibrator = PlattScaling()
        else:
            self.calibrator = None
        
        # Dynamic threshold
        self.dynamic_threshold = DynamicThreshold()
    
    def estimate_uncertainty(
        self,
        x: torch.Tensor,
        combine_methods: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate uncertainty using configured methods.
        
        Args:
            x: Input tensor
            combine_methods: Whether to combine uncertainties from multiple methods
            
        Returns:
            Dictionary with uncertainty estimates
        """
        uncertainties = {}
        
        # MC-Dropout uncertainty
        if 'mc_dropout' in self.methods:
            mc_output = self.mc_dropout(x)
            uncertainties['mc_dropout'] = mc_output['uncertainty']
        
        # Ensemble uncertainty
        if 'ensemble' in self.methods and hasattr(self, 'ensemble'):
            ensemble_output = self.ensemble(x)
            uncertainties['ensemble'] = ensemble_output['uncertainty']
        
        # Combine uncertainties
        if combine_methods and len(uncertainties) > 1:
            # Average uncertainties
            combined = torch.stack(list(uncertainties.values())).mean(dim=0)
            uncertainties['combined'] = combined
        elif len(uncertainties) == 1:
            uncertainties['combined'] = list(uncertainties.values())[0]
        
        return uncertainties
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with uncertainty estimation."""
        return self.estimate_uncertainty(x)


def build_uncertainty_estimator(cfg, model, ensemble_models=None) -> UncertaintyEstimator:
    """
    Build uncertainty estimator from config.
    
    Args:
        cfg: Configuration object
        model: Base model
        ensemble_models: Optional ensemble models
        
    Returns:
        UncertaintyEstimator instance
    """
    methods = []
    if cfg.UNCERTAINTY.MC_DROPOUT:
        methods.append('mc_dropout')
    if cfg.UNCERTAINTY.ENSEMBLE_SIZE > 1 and ensemble_models is not None:
        methods.append('ensemble')
    
    return UncertaintyEstimator(
        model=model,
        methods=methods,
        mc_samples=cfg.UNCERTAINTY.MC_SAMPLES,
        ensemble_models=ensemble_models,
        calibration=cfg.UNCERTAINTY.CALIBRATION
    )
