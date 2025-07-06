# backend/modules/__init__.py
"""
LitLens Embedding Modules

This package provides a unified interface for embedding text using both OpenAI and HuggingFace models.
The interface includes advanced features like caching, batch processing, error handling, and metrics.
"""

from .base_embedder import (
    BaseEmbedder,
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingResult,
    EmbeddingMetrics,
    EmbeddingCache
)

# Temporarily disable ALL embedder imports to get basic functionality working
# from .openai_embedder import OpenAIEmbedder
# from .huggingface_embedder import HuggingFaceEmbedder

# from .embedding_factory import (
#     EmbeddingFactory,
#     embedding_factory,
#     create_embedder,
#     create_openai_embedder,
#     create_huggingface_embedder,
#     get_available_models
# )

__all__ = [
    # Base classes
    'BaseEmbedder',
    'EmbeddingConfig',
    'EmbeddingProvider',
    'EmbeddingResult',
    'EmbeddingMetrics',
    'EmbeddingCache',
    
    # Embedder implementations (ALL temporarily disabled)
    # 'OpenAIEmbedder',
    # 'HuggingFaceEmbedder',
    
    # Factory (temporarily disabled)
    # 'EmbeddingFactory',
    # 'embedding_factory',
    # 'create_embedder',
    # 'create_openai_embedder',
    # 'create_huggingface_embedder',
    # 'get_available_models'
]

__version__ = '0.4.0'
__author__ = 'LitLens Development Team'
__description__ = 'Unified embedding interface for LitLens'