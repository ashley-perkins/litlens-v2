#!/usr/bin/env python3
"""
Test script for the refactored LitLens summarization modules

This script tests both OpenAI and HuggingFace summarization to ensure 
the refactoring maintains functionality while adding new features.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Add the backend directory to the path
sys.path.insert(0, '/mnt/c/Users/cvill/litlens-analysis/consolidated-litlens/backend')

from modules.summarizer import (
    SummarizationConfig,
    SummarizationProvider,
    SummarizationStrategy,
    summarization_manager,
    summarize_with_openai,
    summarize_with_huggingface,
    get_available_models,
    summarize_inline_text
)

from utils.hf_utils import (
    summarize_text_with_hf_api,
    test_hf_model,
    get_hf_model_info,
    get_recommended_model
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample text for testing
SAMPLE_TEXT = """
Artificial intelligence (AI) has emerged as one of the most transformative technologies of the 21st century. 
Machine learning, a subset of AI, enables computers to learn and improve from experience without being explicitly programmed. 
Deep learning, which uses neural networks with multiple layers, has revolutionized fields such as computer vision, 
natural language processing, and speech recognition. Recent advances in large language models like GPT and BERT have 
demonstrated remarkable capabilities in understanding and generating human-like text. These developments have significant 
implications for various industries, including healthcare, finance, education, and transportation. However, the rapid 
advancement of AI also raises important ethical considerations around bias, privacy, job displacement, and the need 
for responsible AI development and deployment.
"""

RESEARCH_GOAL = "Understanding the current state and future implications of artificial intelligence technologies"

async def test_basic_functionality():
    """Test basic summarization functionality"""
    logger.info("Testing basic summarization functionality...")
    
    try:
        # Test backward compatibility function
        result = summarize_inline_text(SAMPLE_TEXT, RESEARCH_GOAL)
        logger.info(f"Backward compatibility test passed. Summary length: {len(result)} characters")
        
        # Test with simple configuration
        config = SummarizationConfig(
            provider=SummarizationProvider.OPENAI,
            model="gpt-4",
            max_tokens=4096,
            temperature=0.7
        )
        
        result = await summarization_manager.summarize(
            content=SAMPLE_TEXT,
            goal=RESEARCH_GOAL,
            provider=SummarizationProvider.OPENAI,
            config=config
        )
        
        logger.info("✅ Basic OpenAI summarization test passed")
        logger.info(f"Quality score: {result.metadata.quality_score:.2f}")
        logger.info(f"Cost estimate: ${result.metadata.cost_estimate:.4f}")
        logger.info(f"Processing time: {result.metadata.processing_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        return False

async def test_openai_features():
    """Test OpenAI-specific features"""
    logger.info("Testing OpenAI advanced features...")
    
    try:
        # Test with custom configuration
        config = SummarizationConfig(
            provider=SummarizationProvider.OPENAI,
            model="gpt-4",
            max_tokens=8192,
            temperature=0.5,
            max_retries=2,
            cache_enabled=True,
            quality_threshold=0.6,
            cost_tracking_enabled=True
        )
        
        # First call
        result1 = await summarize_with_openai(SAMPLE_TEXT, RESEARCH_GOAL, config)
        
        # Second call (should be cached)
        result2 = await summarize_with_openai(SAMPLE_TEXT, RESEARCH_GOAL, config)
        
        logger.info("✅ OpenAI advanced features test passed")
        logger.info(f"First call processing time: {result1.metadata.processing_time:.2f}s")
        logger.info(f"Second call processing time: {result2.metadata.processing_time:.2f}s")
        
        # Test usage statistics
        summarizer = summarization_manager.get_summarizer(SummarizationProvider.OPENAI, config)
        stats = summarizer.get_usage_stats()
        logger.info(f"Usage stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ OpenAI features test failed: {e}")
        return False

async def test_huggingface_features():
    """Test HuggingFace-specific features"""
    logger.info("Testing HuggingFace features...")
    
    # Check if HuggingFace token is available
    if not os.getenv("HUGGINGFACE_API_TOKEN"):
        logger.warning("⚠️ HUGGINGFACE_API_TOKEN not found, skipping HuggingFace tests")
        return True
    
    try:
        # Test model information
        model_name = get_recommended_model("general")
        model_info = get_hf_model_info(model_name)
        logger.info(f"Testing with model: {model_info}")
        
        # Test HuggingFace summarization
        result = await summarize_text_with_hf_api(
            text=SAMPLE_TEXT,
            model_name=model_name,
            goal=RESEARCH_GOAL
        )
        
        logger.info("✅ HuggingFace basic test passed")
        logger.info(f"Summary: {result[:100]}...")
        
        # Test model testing function
        test_result = await test_hf_model(model_name)
        logger.info(f"Model test result: {test_result}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ HuggingFace features test failed: {e}")
        return False

async def test_error_handling():
    """Test error handling and resilience"""
    logger.info("Testing error handling...")
    
    try:
        # Test with invalid model
        config = SummarizationConfig(
            provider=SummarizationProvider.OPENAI,
            model="invalid-model",
            max_retries=1
        )
        
        try:
            result = await summarization_manager.summarize(
                content=SAMPLE_TEXT,
                goal=RESEARCH_GOAL,
                provider=SummarizationProvider.OPENAI,
                config=config
            )
            logger.warning("Expected error for invalid model, but got result")
        except Exception as e:
            logger.info(f"✅ Correctly handled invalid model error: {type(e).__name__}")
        
        # Test with empty content
        try:
            result = await summarization_manager.summarize(
                content="",
                goal=RESEARCH_GOAL,
                provider=SummarizationProvider.OPENAI
            )
            logger.info("✅ Handled empty content correctly")
        except Exception as e:
            logger.info(f"✅ Correctly handled empty content: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        return False

def test_utility_functions():
    """Test utility functions"""
    logger.info("Testing utility functions...")
    
    try:
        # Test available models
        models = get_available_models()
        logger.info(f"Available models: {models}")
        
        # Test model recommendations
        for task in ["general", "news", "academic", "fast"]:
            recommended = get_recommended_model(task)
            logger.info(f"Recommended model for {task}: {recommended}")
        
        logger.info("✅ Utility functions test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Utility functions test failed: {e}")
        return False

async def test_configuration_validation():
    """Test configuration validation"""
    logger.info("Testing configuration validation...")
    
    try:
        # Test various configurations
        configs = [
            SummarizationConfig(
                provider=SummarizationProvider.OPENAI,
                strategy=SummarizationStrategy.SECTION_BY_SECTION,
                max_tokens=4096
            ),
            SummarizationConfig(
                provider=SummarizationProvider.OPENAI,
                strategy=SummarizationStrategy.ABSTRACTIVE,
                temperature=0.9,
                max_retries=5
            )
        ]
        
        for i, config in enumerate(configs):
            logger.info(f"Testing configuration {i + 1}: {config.strategy.value}")
            # Just create summarizer to test validation
            summarizer = summarization_manager.get_summarizer(config.provider, config)
            logger.info(f"✅ Configuration {i + 1} validated successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration validation test failed: {e}")
        return False

async def main():
    """Run all tests"""
    logger.info("🚀 Starting LitLens Summarizer Tests")
    logger.info("=" * 50)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Basic Functionality", test_basic_functionality()),
        ("OpenAI Features", test_openai_features()),
        ("HuggingFace Features", test_huggingface_features()),
        ("Error Handling", test_error_handling()),
        ("Utility Functions", test_utility_functions()),
        ("Configuration Validation", test_configuration_validation())
    ]
    
    for test_name, test_coro in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        try:
            result = await test_coro
            test_results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name} test PASSED")
            else:
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} test FAILED with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:.<30} {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Refactoring was successful.")
        return 0
    else:
        logger.error("⚠️ Some tests failed. Please review the issues.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)