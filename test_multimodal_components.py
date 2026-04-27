"""
Test script to verify Multi-Modal REW components
Tests CLIP integration, fusion, and basic functionality
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_clip_wrapper():
    """Test CLIP wrapper functionality."""
    print("=" * 60)
    print("Testing CLIP Wrapper...")
    print("=" * 60)
    
    try:
        from mepu.model.rew.clip_wrapper import CLIPFeatureExtractor
        
        # Initialize CLIP extractor
        clip_extractor = CLIPFeatureExtractor(
            model_name='ViT-B/32',
            device='cpu',  # Use CPU for testing
            freeze=True
        )
        
        print(f"[OK] CLIP model loaded: ViT-B/32")
        print(f"[OK] Visual feature dim: {clip_extractor.visual_dim}")
        print(f"[OK] Text feature dim: {clip_extractor.text_dim}")
        
        # Test visual feature extraction
        dummy_images = torch.randn(2, 3, 224, 224)  # Batch of 2 images
        visual_features = clip_extractor.extract_visual_features(dummy_images)
        print(f"[OK] Visual features shape: {visual_features.shape}")
        
        # Test text feature extraction
        texts = ["a photo of a dog", "a photo of a cat"]
        text_features = clip_extractor.extract_text_features(texts)
        print(f"[OK] Text features shape: {text_features.shape}")
        
        # Test class text features
        class_names = ["person", "car", "dog", "cat"]
        class_features = clip_extractor.get_class_text_features(class_names)
        print(f"[OK] Class features shape: {class_features.shape}")
        
        # Test similarity computation
        similarity = clip_extractor.compute_similarity(visual_features, text_features)
        print(f"[OK] Similarity matrix shape: {similarity.shape}")
        
        print("\n[PASS] CLIP Wrapper test PASSED!\n")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] CLIP Wrapper test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_fusion_module():
    """Test multi-modal fusion module."""
    print("=" * 60)
    print("Testing Fusion Module...")
    print("=" * 60)
    
    try:
        from mepu.model.rew.fusion import MultiModalFusion
        
        # Test different fusion strategies
        fusion_types = ['concat', 'attention', 'gating', 'adaptive']
        
        for fusion_type in fusion_types:
            print(f"\nTesting {fusion_type} fusion...")
            
            fusion = MultiModalFusion(
                soco_dim=2048,
                clip_visual_dim=512,
                clip_text_dim=512,
                fusion_type=fusion_type,
                output_dim=1024
            )
            
            # Create dummy features
            batch_size = 4
            soco_features = torch.randn(batch_size, 2048)
            clip_visual = torch.randn(batch_size, 512)
            clip_text = torch.randn(batch_size, 512)
            
            # Test fusion
            output = fusion(soco_features, clip_visual, clip_text)
            
            print(f"  [OK] Fused features shape: {output['fused'].shape}")
            if output['weights'] is not None:
                print(f"  [OK] Fusion weights shape: {output['weights'].shape}")
        
        print("\n[PASS] Fusion Module test PASSED!\n")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Fusion Module test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_uncertainty_estimator():
    """Test uncertainty estimation module."""
    print("=" * 60)
    print("Testing Uncertainty Estimator...")
    print("=" * 60)
    
    try:
        from mepu.model.uncertainty_estimator import (
            MCDropoutUncertainty,
            TemperatureScaling,
            DynamicThreshold
        )
        
        # Create a simple test model
        test_model = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128)
        )
        
        # Test MC-Dropout
        print("\nTesting MC-Dropout...")
        mc_dropout = MCDropoutUncertainty(test_model, n_samples=5)
        dummy_input = torch.randn(4, 512)
        mc_output = mc_dropout(dummy_input)
        print(f"  [OK] Mean prediction shape: {mc_output['mean'].shape}")
        print(f"  [OK] Uncertainty shape: {mc_output['uncertainty'].shape}")
        print(f"  [OK] Uncertainty values: {mc_output['uncertainty'][:3].tolist()}")
        
        # Test Temperature Scaling
        print("\nTesting Temperature Scaling...")
        temp_scaling = TemperatureScaling()
        logits = torch.randn(10, 5)
        scaled_logits = temp_scaling(logits)
        print(f"  [OK] Scaled logits shape: {scaled_logits.shape}")
        print(f"  [OK] Temperature: {temp_scaling.temperature.item():.4f}")
        
        # Test Dynamic Threshold
        print("\nTesting Dynamic Threshold...")
        dynamic_thresh = DynamicThreshold()
        uncertainties = torch.rand(100)
        threshold = dynamic_thresh.get_adaptive_threshold(uncertainties, method='percentile', percentile=0.7)
        print(f"  [OK] Adaptive threshold (percentile): {threshold:.4f}")
        
        threshold_otsu = dynamic_thresh.get_adaptive_threshold(uncertainties, method='otsu')
        print(f"  [OK] Adaptive threshold (Otsu): {threshold_otsu:.4f}")
        
        print("\n[PASS] Uncertainty Estimator test PASSED!\n")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Uncertainty Estimator test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_pseudo_labeler():
    """Test uncertainty-aware pseudo-labeler."""
    print("=" * 60)
    print("Testing Pseudo-Labeler...")
    print("=" * 60)
    
    try:
        from tools.gen_pseudo_label_uncertainty import UncertaintyAwarePseudoLabeler
        import numpy as np
        
        # Create dummy proposals
        n_proposals = 100
        proposals = [
            {'bbox': [10, 10, 50, 50], 'score': np.random.rand()}
            for _ in range(n_proposals)
        ]
        
        # Create dummy scores
        rew_scores = {
            'combined_score': np.random.rand(n_proposals),
            'visual_score': np.random.rand(n_proposals),
            'semantic_score': np.random.rand(n_proposals)
        }
        uncertainty_scores = np.random.rand(n_proposals)
        
        # Initialize labeler
        labeler = UncertaintyAwarePseudoLabeler(
            uncertainty_threshold=0.5,
            quality_threshold=0.4,
            use_active_learning=True,
            active_learning_budget=20
        )
        
        # Generate pseudo-labels
        print("\nGenerating pseudo-labels...")
        pseudo_labels = labeler.generate_pseudo_labels(
            proposals=proposals,
            rew_scores=rew_scores,
            uncertainty_scores=uncertainty_scores,
            known_cls_num=19,
            keep_type='quality'
        )
        
        print(f"  [OK] Input proposals: {len(proposals)}")
        print(f"  [OK] Output pseudo-labels: {len(pseudo_labels)}")
        if len(pseudo_labels) > 0:
            print(f"  [OK] Sample quality score: {pseudo_labels[0].get('quality_score', 'N/A')}")
            print(f"  [OK] Sample uncertainty: {pseudo_labels[0].get('uncertainty', 'N/A')}")
        
        print("\n[PASS] Pseudo-Labeler test PASSED!\n")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Pseudo-Labeler test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MULTI-MODAL REW COMPONENT TESTS")
    print("=" * 60 + "\n")
    
    results = {}
    
    # Run tests
    results['CLIP Wrapper'] = test_clip_wrapper()
    results['Fusion Module'] = test_fusion_module()
    results['Uncertainty Estimator'] = test_uncertainty_estimator()
    results['Pseudo-Labeler'] = test_pseudo_labeler()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{test_name:.<40} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n*** All tests PASSED! Multi-Modal REW is ready to use! ***")
    else:
        print("\n*** Some tests failed. Please check the errors above. ***")
    
    print("=" * 60 + "\n")
    
    return total_passed == total_tests


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

