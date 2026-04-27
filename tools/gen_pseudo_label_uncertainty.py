"""
Enhanced Pseudo-Label Generation with Uncertainty-Aware Filtering
Improves pseudo-label quality using multi-modal REW scores and uncertainty estimates
"""

import json
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import argparse


class UncertaintyAwarePseudoLabeler:
    """
    Generates high-quality pseudo-labels using:
    1. Multi-modal REW scores
    2. Uncertainty estimates
    3. Active learning sample selection
    4. Quality scoring and filtering
    """
    
    def __init__(
        self,
        uncertainty_threshold: float = 0.3,
        confidence_threshold: float = 0.5,
        rew_threshold: float = 0.6,
        quality_threshold: float = 0.5,
        use_active_learning: bool = True,
        active_learning_budget: int = 1000
    ):
        """
        Args:
            uncertainty_threshold: Maximum uncertainty to accept (lower = more certain)
            confidence_threshold: Minimum confidence score to accept
            rew_threshold: Minimum REW score to accept
            quality_threshold: Minimum overall quality score
            use_active_learning: Whether to use active learning selection
            active_learning_budget: Number of samples to select via active learning
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.confidence_threshold = confidence_threshold
        self.rew_threshold = rew_threshold
        self.quality_threshold = quality_threshold
        self.use_active_learning = use_active_learning
        self.active_learning_budget = active_learning_budget
    
    def filter_by_uncertainty(
        self,
        proposals: List[Dict],
        uncertainty_scores: np.ndarray
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Filter proposals by uncertainty threshold.
        
        Args:
            proposals: List of proposal dictionaries
            uncertainty_scores: Uncertainty score for each proposal
            
        Returns:
            Filtered proposals and their uncertainty scores
        """
        # Keep only low-uncertainty proposals
        mask = uncertainty_scores < self.uncertainty_threshold
        
        filtered_proposals = [p for i, p in enumerate(proposals) if mask[i]]
        filtered_uncertainties = uncertainty_scores[mask]
        
        print(f"Uncertainty filtering: {len(proposals)} -> {len(filtered_proposals)} proposals")
        
        return filtered_proposals, filtered_uncertainties
    
    def compute_quality_scores(
        self,
        proposals: List[Dict],
        rew_scores: np.ndarray,
        uncertainty_scores: np.ndarray,
        proposal_scores: Optional[np.ndarray] = None,
        weights: Dict[str, float] = None
    ) -> np.ndarray:
        """
        Compute overall quality score for each proposal.
        
        Args:
            proposals: List of proposals
            rew_scores: REW scores (higher = more likely unknown)
            uncertainty_scores: Uncertainty scores (lower = more certain)
            proposal_scores: Original proposal confidence scores
            weights: Weights for different components
            
        Returns:
            Quality scores for each proposal
        """
        if weights is None:
            weights = {
                'rew': 0.4,
                'uncertainty': 0.4,
                'proposal': 0.2
            }
        
        n_proposals = len(proposals)
        
        # Normalize scores to [0, 1]
        rew_norm = self._normalize(rew_scores)
        uncertainty_norm = 1.0 - self._normalize(uncertainty_scores)  # Invert: low uncertainty = high quality
        
        if proposal_scores is not None:
            proposal_norm = self._normalize(proposal_scores)
        else:
            proposal_norm = np.ones(n_proposals)
        
        # Compute weighted quality score
        quality = (
            weights['rew'] * rew_norm +
            weights['uncertainty'] * uncertainty_norm +
            weights['proposal'] * proposal_norm
        )
        
        return quality
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        min_val = scores.min()
        max_val = scores.max()
        
        if max_val - min_val < 1e-10:
            return np.ones_like(scores) * 0.5
        
        return (scores - min_val) / (max_val - min_val)
    
    def select_informative_samples(
        self,
        proposals: List[Dict],
        uncertainty_scores: np.ndarray,
        quality_scores: np.ndarray,
        budget: Optional[int] = None
    ) -> List[int]:
        """
        Select most informative samples using active learning strategy.
        
        Args:
            proposals: List of proposals
            uncertainty_scores: Uncertainty for each proposal
            quality_scores: Quality score for each proposal
            budget: Number of samples to select
            
        Returns:
            Indices of selected samples
        """
        if budget is None:
            budget = self.active_learning_budget
        
        n_proposals = len(proposals)
        budget = min(budget, n_proposals)
        
        # Informativeness: balance between uncertainty and quality
        # We want samples that are:
        # 1. Moderately uncertain (not too confident, not too noisy)
        # 2. High quality overall
        
        # Compute informativeness score
        # Prefer moderate uncertainty (peak around 0.3-0.5)
        optimal_uncertainty = 0.4
        uncertainty_penalty = np.abs(uncertainty_scores - optimal_uncertainty)
        uncertainty_informativeness = 1.0 - self._normalize(uncertainty_penalty)
        
        # Combine with quality
        informativeness = 0.6 * quality_scores + 0.4 * uncertainty_informativeness
        
        # Select top-k most informative
        selected_indices = np.argsort(informativeness)[-budget:]
        
        print(f"Active learning: selected {len(selected_indices)} informative samples")
        
        return selected_indices.tolist()
    
    def generate_pseudo_labels(
        self,
        proposals: List[Dict],
        rew_scores: Dict[str, np.ndarray],
        uncertainty_scores: np.ndarray,
        known_cls_num: int,
        keep_type: str = 'quality',
        num_keep: Optional[int] = None,
        percent_keep: Optional[float] = None
    ) -> List[Dict]:
        """
        Generate high-quality pseudo-labels with uncertainty-aware filtering.
        
        Args:
            proposals: List of proposal dictionaries
            rew_scores: Dictionary with different REW scores
            uncertainty_scores: Uncertainty estimates
            known_cls_num: Number of known classes
            keep_type: How to select proposals ('quality', 'num', 'percent')
            num_keep: Number of proposals to keep (if keep_type='num')
            percent_keep: Percentage of proposals to keep (if keep_type='percent')
            
        Returns:
            List of filtered pseudo-labels
        """
        # Extract proposal scores if available
        proposal_scores = np.array([p.get('score', 0.5) for p in proposals])
        
        # Use combined REW score (or fused score if available)
        if 'combined_score' in rew_scores:
            rew_score_values = rew_scores['combined_score']
        elif 'fused_score' in rew_scores:
            rew_score_values = rew_scores['fused_score']
        else:
            rew_score_values = rew_scores.get('visual_score', np.zeros(len(proposals)))
        
        # Step 1: Filter by uncertainty
        filtered_proposals, filtered_uncertainties = self.filter_by_uncertainty(
            proposals, uncertainty_scores
        )
        
        if len(filtered_proposals) == 0:
            print("Warning: No proposals passed uncertainty filtering!")
            return []
        
        # Update scores for filtered proposals
        filtered_indices = [i for i, u in enumerate(uncertainty_scores) if u < self.uncertainty_threshold]
        filtered_rew = rew_score_values[filtered_indices]
        filtered_proposal_scores = proposal_scores[filtered_indices]
        
        # Step 2: Compute quality scores
        quality_scores = self.compute_quality_scores(
            filtered_proposals,
            filtered_rew,
            filtered_uncertainties,
            filtered_proposal_scores
        )
        
        # Step 3: Filter by quality threshold
        quality_mask = quality_scores >= self.quality_threshold
        high_quality_proposals = [p for i, p in enumerate(filtered_proposals) if quality_mask[i]]
        high_quality_scores = quality_scores[quality_mask]
        high_quality_uncertainties = filtered_uncertainties[quality_mask]
        
        print(f"Quality filtering: {len(filtered_proposals)} -> {len(high_quality_proposals)} proposals")
        
        if len(high_quality_proposals) == 0:
            print("Warning: No proposals passed quality filtering!")
            return []
        
        # Step 4: Select final proposals based on keep_type
        if keep_type == 'quality':
            # Use active learning to select most informative
            if self.use_active_learning:
                selected_indices = self.select_informative_samples(
                    high_quality_proposals,
                    high_quality_uncertainties,
                    high_quality_scores
                )
                final_proposals = [high_quality_proposals[i] for i in selected_indices]
            else:
                # Keep all high-quality proposals
                final_proposals = high_quality_proposals
        
        elif keep_type == 'num':
            # Keep top-N by quality score
            if num_keep is None:
                num_keep = len(high_quality_proposals)
            num_keep = min(num_keep, len(high_quality_proposals))
            top_indices = np.argsort(high_quality_scores)[-num_keep:]
            final_proposals = [high_quality_proposals[i] for i in top_indices]
        
        elif keep_type == 'percent':
            # Keep top percentage by quality score
            if percent_keep is None:
                percent_keep = 0.5
            num_keep = int(len(high_quality_proposals) * percent_keep)
            num_keep = max(1, num_keep)
            top_indices = np.argsort(high_quality_scores)[-num_keep:]
            final_proposals = [high_quality_proposals[i] for i in top_indices]
        
        else:
            final_proposals = high_quality_proposals
        
        # Step 5: Assign unknown class label and quality weights
        for i, proposal in enumerate(final_proposals):
            # Assign unknown class (class ID = known_cls_num)
            proposal['category_id'] = known_cls_num
            
            # Add quality score as weight for training
            idx = high_quality_proposals.index(proposal)
            proposal['quality_score'] = float(high_quality_scores[idx])
            proposal['uncertainty'] = float(high_quality_uncertainties[idx])
        
        print(f"Final pseudo-labels: {len(final_proposals)} selected")
        
        return final_proposals
    
    def save_pseudo_labels(
        self,
        pseudo_labels: List[Dict],
        output_path: str
    ):
        """
        Save pseudo-labels to JSON file.
        
        Args:
            pseudo_labels: List of pseudo-label dictionaries
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(pseudo_labels, f, indent=2)
        
        print(f"Saved {len(pseudo_labels)} pseudo-labels to {output_path}")


def main():
    """
    Main function for generating uncertainty-aware pseudo-labels.
    """
    parser = argparse.ArgumentParser(description='Generate uncertainty-aware pseudo-labels')
    
    parser.add_argument('--proposal_path', type=str, required=True,
                        help='Path to proposal JSON file')
    parser.add_argument('--rew_scores_path', type=str, required=False,
                        help='Path to REW scores file (ignored, uses scores from json instead)')
    parser.add_argument('--uncertainty_scores_path', type=str, required=True,
                        help='Path to uncertainty scores file')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to save pseudo-labels')
    parser.add_argument('--known_cls_num', type=int, required=True,
                        help='Number of known classes')
    
    # Filtering parameters
    parser.add_argument('--uncertainty_threshold', type=float, default=0.3, help='Maximum uncertainty threshold')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Minimum confidence threshold')
    parser.add_argument('--rew_threshold', type=float, default=0.6, help='Minimum REW score threshold')
    parser.add_argument('--quality_threshold', type=float, default=0.5, help='Minimum quality score threshold')
    
    # Selection parameters
    parser.add_argument('--keep_type', type=str, default='quality', choices=['quality', 'num', 'percent'], help='How to select final proposals')
    parser.add_argument('--num_keep', type=int, default=None, help='Number of proposals to keep (for keep_type=num)')
    parser.add_argument('--percent_keep', type=float, default=None, help='Percentage of proposals to keep (for keep_type=percent)')
    
    # Active learning
    parser.add_argument('--use_active_learning', action='store_true', help='Use active learning for sample selection')
    parser.add_argument('--active_learning_budget', type=int, default=1000, help='Budget for active learning selection')
    
    args = parser.parse_args()
    
    print(f"Loading proposals from {args.proposal_path}")
    with open(args.proposal_path, 'r') as f:
        proposals_dict = json.load(f)

    flat_proposals = []
    flat_rew_scores = []
    
    for img_id, data in proposals_dict.items():
        bboxes = data.get("bboxes", [])
        scores = data.get("scores", [])
        img_id_val = data.get("image_id", img_id)
        
        for i, box in enumerate(bboxes):
            flat_proposals.append({
                "bbox": box,
                "image_id": img_id_val,
                "score": 1.0  # Or extract from a conf_score field if it exists
            })
            flat_rew_scores.append(scores[i] if i < len(scores) else 0.5)

    rew_scores = {'combined_score': np.array(flat_rew_scores)}
    proposals = flat_proposals
    
    print(f"Loading uncertainty scores from {args.uncertainty_scores_path}")
    uncertainty_scores = np.load(args.uncertainty_scores_path)
    
    labeler = UncertaintyAwarePseudoLabeler(
        uncertainty_threshold=args.uncertainty_threshold,
        confidence_threshold=args.confidence_threshold,
        rew_threshold=args.rew_threshold,
        quality_threshold=args.quality_threshold,
        use_active_learning=args.use_active_learning,
        active_learning_budget=args.active_learning_budget
    )
    
    pseudo_labels = labeler.generate_pseudo_labels(
        proposals=proposals,
        rew_scores=rew_scores,
        uncertainty_scores=uncertainty_scores,
        known_cls_num=args.known_cls_num,
        keep_type=args.keep_type,
        num_keep=args.num_keep,
        percent_keep=args.percent_keep
    )
    
    # Repack pseudo-labels into the image_id keyed dictionary format
    out_dict = {}
    for p in pseudo_labels:
        img_id = p["image_id"]
        if img_id not in out_dict:
            out_dict[img_id] = {"bboxes": [], "scores": [], "image_id": img_id}
        out_dict[img_id]["bboxes"].append(p["bbox"])
        out_dict[img_id]["scores"].append(p.get("quality_score", 1.0))

    labeler.save_pseudo_labels(out_dict, args.save_path)
    print("Done!")

if __name__ == '__main__':
    main()
