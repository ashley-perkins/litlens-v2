# backend/modules/__init__.py
"""
LitLens OpenAI-Only Modules

This package provides OpenAI-powered functionality for:
- Text embeddings using OpenAI Ada-002
- Summarization using GPT-4
- PDF text extraction and processing
"""

from .base_embedder import (
    BaseEmbedder,
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingResult,
    EmbeddingMetrics,
    EmbeddingCache
)

from .openai_embedder import OpenAIEmbedder

__all__ = [
    # Base classes
    'BaseEmbedder',
    'EmbeddingConfig',
    'EmbeddingProvider',
    'EmbeddingResult',
    'EmbeddingMetrics',
    'EmbeddingCache',
    
    # OpenAI embedder
    'OpenAIEmbedder',
]

__version__ = '1.0.0'
__author__ = 'LitLens Development Team'
__description__ = 'OpenAI-powered embedding and processing for LitLens'