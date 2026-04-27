"""
GPU-enabled test script for Multi-Modal REW components
Tests CLIP integration, fusion, and basic functionality on CUDA
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_gpu_availability():
    """Test GPU availability and setup."""
    print("=" * 60)
    print("Testing GPU Setup...")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2)} GB")
        print(f"Current device: {torch.cuda.current_device()}")
        print("\n[PASS] GPU is available and ready!\n")
        return True
    else:
        print("\n[FAIL] GPU not available!\n")
        return False


def test_clip_wrapper_gpu():
    """Test CLIP wrapper on GPU."""
    print("=" * 60)
    print("Testing CLIP Wrapper on GPU...")
    print("=" * 60)
    
    try:
        from mepu.model.rew.clip_wrapper import CLIPFeatureExtractor
        
        # Initialize CLIP extractor on GPU
        clip_extractor = CLIPFeatureExtractor(
            model_name='ViT-B/32',
            device='cuda',
            freeze=True
        )
        
        print(f"[OK] CLIP model loaded on GPU")
        print(f"[OK] Visual feature dim: {clip_extractor.visual_dim}")
        
        # Test visual feature extraction on GPU
        dummy_images = torch.randn(2, 3, 224, 224).cuda()
        visual_features = clip_extractor.extract_visual_features(dummy_images)
        print(f"[OK] Visual features shape: {visual_features.shape}")
        print(f"[OK] Features on GPU: {visual_features.is_cuda}")
        
        # Test text feature extraction
        texts = ["a photo of a dog", "a photo of a cat"]
        text_features = clip_extractor.extract_text_features(texts)
        print(f"[OK] Text features shape: {text_features.shape}")
        print(f"[OK] Text features on GPU: {text_features.is_cuda}")
        
        print("\n[PASS] CLIP Wrapper GPU test PASSED!\n")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] CLIP Wrapper GPU test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_fusion_module_gpu():
    """Test multi-modal fusion on GPU."""
    print("=" * 60)
    print("Testing Fusion Module on GPU...")
    print("=" * 60)
    
    try:
        from mepu.model.rew.fusion import MultiModalFusion
        
        fusion = MultiModalFusion(
            soco_dim=2048,
            clip_visual_dim=512,
            clip_text_dim=512,
            fusion_type='attention',
            output_dim=1024
        ).cuda()
        
        # Create dummy features on GPU
        batch_size = 4
        soco_features = torch.randn(batch_size, 2048).cuda()
        clip_visual = torch.randn(batch_size, 512).cuda()
        clip_text = torch.randn(batch_size, 512).cuda()
        
        # Test fusion
        output = fusion(soco_features, clip_visual, clip_text)
        
        print(f"[OK] Fused features shape: {output['fused'].shape}")
        print(f"[OK] Features on GPU: {output['fused'].is_cuda}")
        
        print("\n[PASS] Fusion Module GPU test PASSED!\n")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Fusion Module GPU test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_memory():
    """Test GPU memory usage."""
    print("=" * 60)
    print("Testing GPU Memory...")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("[SKIP] No GPU available")
        return True
    
    try:
        # Get initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Initial GPU memory: {initial_memory:.2f} MB")
        
        # Allocate some tensors
        test_tensor = torch.randn(1000, 1000, 100).cuda()
        allocated_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"After allocation: {allocated_memory:.2f} MB")
        print(f"Allocated: {allocated_memory - initial_memory:.2f} MB")
        
        # Free memory
        del test_tensor
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"After cleanup: {final_memory:.2f} MB")
        
        print("\n[PASS] GPU Memory test PASSED!\n")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] GPU Memory test FAILED: {e}\n")
        return False


def main():
    """Run all GPU tests."""
    print("\n" + "=" * 60)
    print("MULTI-MODAL REW GPU TESTS")
    print("=" * 60 + "\n")
    
    results = {}
    
    # Run tests
    results['GPU Setup'] = test_gpu_availability()
    
    if results['GPU Setup']:
        results['CLIP Wrapper GPU'] = test_clip_wrapper_gpu()
        results['Fusion Module GPU'] = test_fusion_module_gpu()
        results['GPU Memory'] = test_gpu_memory()
    else:
        print("Skipping GPU tests - GPU not available")
        return False
    
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
        print("\n*** All GPU tests PASSED! Ready for GPU training! ***")
    else:
        print("\n*** Some tests failed. Please check the errors above. ***")
    
    print("=" * 60 + "\n")
    
    return total_passed == total_tests


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
