# backend/examples/embedding_usage.py
"""
Example usage of the unified embedding interface
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the unified embedding interface
from backend.modules.embedding_factory import (
    create_openai_embedder,
    create_huggingface_embedder,
    get_available_models
)
from backend.modules.base_embedder import EmbeddingConfig, EmbeddingProvider


def example_openai_usage():
    """Example usage of OpenAI embedder"""
    print("=== OpenAI Embedder Example ===")
    
    try:
        # Create OpenAI embedder with default settings
        embedder = create_openai_embedder(
            model_name="text-embedding-ada-002",
            batch_size=50,
            cache_size=1000
        )
        
        # Single text embedding
        text = "This is a sample text for embedding"
        embedding = embedder.embed_text(text)
        print(f"Single text embedding dimension: {len(embedding)}")
        
        # Multiple texts embedding
        texts = [
            "First research paper about machine learning",
            "Second paper on natural language processing",
            "Third study on computer vision"
        ]
        
        result = embedder.embed_texts(texts)
        print(f"Batch embedding results:")
        print(f"  - Number of embeddings: {len(result.embeddings)}")
        print(f"  - Embedding dimension: {result.dimensions}")
        print(f"  - Tokens used: {result.tokens_used}")
        print(f"  - Processing time: {result.processing_time:.2f}s")
        print(f"  - Cache hits: {result.cache_hits}")
        
        # Paper embedding (backward compatibility)
        papers = [
            {"content": "Research paper 1 content"},
            {"content": "Research paper 2 content"}
        ]
        
        embedded_papers = embedder.embed_papers(papers)
        print(f"Embedded {len(embedded_papers)} papers")
        
        # Get model information
        model_info = embedder.get_model_info()
        print(f"Model info: {model_info}")
        
        # Get metrics
        metrics = embedder.get_metrics()
        print(f"Metrics: Success rate: {metrics.successful_requests}/{metrics.total_requests}")
        
        # Estimate cost
        cost = embedder.estimate_cost(1000)
        print(f"Estimated cost for 1000 tokens: ${cost:.6f}")
        
    except Exception as e:
        print(f"OpenAI embedder error: {e}")


def example_huggingface_usage():
    """Example usage of HuggingFace embedder"""
    print("\n=== HuggingFace Embedder Example ===")
    
    try:
        # Create HuggingFace embedder with default settings
        embedder = create_huggingface_embedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=16,
            cache_size=500
        )
        
        # Single text embedding
        text = "This is a sample text for embedding using HuggingFace"
        embedding = embedder.embed_text(text)
        print(f"Single text embedding dimension: {len(embedding)}")
        
        # Multiple texts embedding
        texts = [
            "First research paper about deep learning",
            "Second paper on transformer architectures",
            "Third study on attention mechanisms"
        ]
        
        result = embedder.embed_texts(texts)
        print(f"Batch embedding results:")
        print(f"  - Number of embeddings: {len(result.embeddings)}")
        print(f"  - Embedding dimension: {result.dimensions}")
        print(f"  - Tokens used: {result.tokens_used}")
        print(f"  - Processing time: {result.processing_time:.2f}s")
        print(f"  - Cache hits: {result.cache_hits}")
        
        # Get model information
        model_info = embedder.get_model_info()
        print(f"Model info: {model_info}")
        
        # Get memory usage (HuggingFace specific)
        memory_stats = embedder.get_memory_usage()
        print(f"Memory usage: {memory_stats}")
        
        # Clear GPU memory
        embedder.clear_memory()
        
    except Exception as e:
        print(f"HuggingFace embedder error: {e}")


def example_custom_configuration():
    """Example of custom configuration"""
    print("\n=== Custom Configuration Example ===")
    
    # Create custom configuration
    config = EmbeddingConfig(
        provider=EmbeddingProvider.OPENAI,
        model_name="text-embedding-3-small",
        max_tokens=8191,
        batch_size=100,
        cache_size=2000,
        retry_attempts=5,
        retry_delay=1.5,
        timeout=45.0,
        rate_limit_delay=0.2
    )
    
    try:
        # Use custom configuration
        from backend.modules.embedding_factory import create_embedder
        
        embedder = create_embedder(
            provider=EmbeddingProvider.OPENAI,
            model_name="text-embedding-3-small",
            config=config
        )
        
        print(f"Created custom embedder with config: {config}")
        
    except Exception as e:
        print(f"Custom configuration error: {e}")


def example_model_comparison():
    """Example of comparing different models"""
    print("\n=== Model Comparison Example ===")
    
    test_text = "Artificial intelligence and machine learning are transforming the world"
    
    # Compare OpenAI models
    openai_models = ["text-embedding-ada-002", "text-embedding-3-small"]
    
    for model in openai_models:
        try:
            embedder = create_openai_embedder(model_name=model)
            embedding = embedder.embed_text(test_text)
            model_info = embedder.get_model_info()
            
            print(f"OpenAI {model}:")
            print(f"  - Dimension: {len(embedding)}")
            print(f"  - Model specs: {model_info.get('specs', {})}")
            
        except Exception as e:
            print(f"Error with {model}: {e}")
    
    # Compare HuggingFace models
    hf_models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ]
    
    for model in hf_models:
        try:
            embedder = create_huggingface_embedder(model_name=model)
            embedding = embedder.embed_text(test_text)
            model_info = embedder.get_model_info()
            
            print(f"HuggingFace {model}:")
            print(f"  - Dimension: {len(embedding)}")
            print(f"  - Model specs: {model_info.get('specs', {})}")
            
        except Exception as e:
            print(f"Error with {model}: {e}")


def example_available_models():
    """Example of listing available models"""
    print("\n=== Available Models Example ===")
    
    # Get OpenAI models
    openai_models = get_available_models("openai")
    print("OpenAI Models:")
    for model, specs in openai_models.items():
        print(f"  - {model}: {specs}")
    
    # Get HuggingFace models
    hf_models = get_available_models("huggingface")
    print("\nHuggingFace Models:")
    for model, specs in hf_models.items():
        print(f"  - {model}: {specs}")


def example_caching_and_metrics():
    """Example of caching and metrics functionality"""
    print("\n=== Caching and Metrics Example ===")
    
    try:
        embedder = create_openai_embedder(cache_size=100)
        
        # First embedding (cache miss)
        text = "This text will be cached"
        embedding1 = embedder.embed_text(text)
        
        # Second embedding (cache hit)
        embedding2 = embedder.embed_text(text)
        
        # Verify embeddings are the same
        assert embedding1 == embedding2, "Cached embeddings should be identical"
        
        # Get cache statistics
        cache_stats = embedder.get_cache_stats()
        print(f"Cache stats: {cache_stats}")
        
        # Get metrics
        metrics = embedder.get_metrics()
        print(f"Metrics - Total requests: {metrics.total_requests}")
        print(f"Metrics - Cache hits: {metrics.cache_hits}")
        print(f"Metrics - Cache misses: {metrics.cache_misses}")
        
        # Clear cache
        embedder.clear_cache()
        print("Cache cleared")
        
        # Reset metrics
        embedder.reset_metrics()
        print("Metrics reset")
        
    except Exception as e:
        print(f"Caching example error: {e}")


def example_backward_compatibility():
    """Example showing backward compatibility"""
    print("\n=== Backward Compatibility Example ===")
    
    try:
        # Import the legacy interfaces
        from backend.modules import embedder as openai_embedder
        from backend.utils import embedder_hf as hf_embedder
        
        # Test OpenAI backward compatibility
        print("Testing OpenAI backward compatibility...")
        openai_embedding = openai_embedder.embed_text("Test text for OpenAI")
        print(f"OpenAI embedding dimension: {len(openai_embedding)}")
        
        # Test HuggingFace backward compatibility
        print("Testing HuggingFace backward compatibility...")
        hf_embedding = hf_embedder.embed_text("Test text for HuggingFace")
        print(f"HuggingFace embedding dimension: {len(hf_embedding)}")
        
        # Test paper embedding
        papers = [{"content": "Sample paper content"}]
        
        openai_papers = openai_embedder.embed_papers(papers.copy())
        hf_papers = hf_embedder.embed_papers(papers.copy())
        
        print(f"OpenAI papers embedded: {len(openai_papers)}")
        print(f"HuggingFace papers embedded: {len(hf_papers)}")
        
    except Exception as e:
        print(f"Backward compatibility error: {e}")


if __name__ == "__main__":
    # Run all examples
    example_openai_usage()
    example_huggingface_usage()
    example_custom_configuration()
    example_model_comparison()
    example_available_models()
    example_caching_and_metrics()
    example_backward_compatibility()
    
    print("\n=== All Examples Complete ===")