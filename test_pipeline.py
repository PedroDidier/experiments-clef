import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        from src.config import get_config, update_config
        print("✓ Configuration module imported")
        
        from src.data.dataset import ROCOv2DataHandler
        print("✓ Dataset handler imported")
        
        from src.vectordb.image_vectordb import ImageVectorDB
        print("✓ Vector database imported")
        
        from src.llm.llm_utils import MedicalImageCaptioner
        print("✓ LLM utilities imported")
        
        from src.analysis.cost_analysis import CostAnalyzer
        print("✓ Cost analysis imported")
        
        from src.analysis.evaluation_visualizer import EvaluationVisualizer
        print("✓ Evaluation visualizer imported")
        
        from src.main import MedicalImageCaptioningPipeline
        print("✓ Main pipeline imported")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration...")
    
    try:
        from src.config import get_config, update_config
        
        # Test default configuration
        config = get_config()
        assert config.cache_drive == "D", "Default cache drive should be D"
        assert "huggingface_cache" in str(config.cache_dir), "Cache directory should contain huggingface_cache"
        
        # Test configuration update
        update_config(cache_drive="C")
        config = get_config()
        assert config.cache_drive == "C", "Cache drive should be updated to C"
        
        print("✓ Configuration system working")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_pipeline_initialization():
    """Test pipeline initialization."""
    print("\nTesting pipeline initialization...")
    
    try:
        from src.main import MedicalImageCaptioningPipeline
        
        # Test with minimal configuration
        pipeline = MedicalImageCaptioningPipeline(
            model_name="gpt-5-mini",
            num_validation_samples=5,
            num_rag_examples=2,
            random_seed=42,
            run_cost_analysis=False,
            run_evaluation=False
        )
        
        assert pipeline.model_name == "gpt-5-mini", "Model name should be set correctly"
        assert pipeline.num_validation_samples == 5, "Validation samples should be set correctly"
        assert pipeline.num_rag_examples == 2, "RAG examples should be set correctly"
        
        print("✓ Pipeline initialization working")
        return True
        
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        return False

def main():
    """Main test function."""
    print("Medical Image Captioning Pipeline - Test Suite")
    print("=" * 50)
    
    # Run tests
    tests = [
        test_imports,
        test_configuration,
        test_pipeline_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to use.")
        print("\nTo run the pipeline:")
        print("  python run_pipeline.py --samples 10 --model gpt-5-mini")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
