# Unified Embedding Interface Documentation

## Overview

The unified embedding interface provides a consistent, robust, and efficient way to work with both OpenAI and HuggingFace embedding models in the LitLens system. This interface includes advanced features like caching, batch processing, error handling, metrics, and more.

## Key Features

- **Unified Interface**: Common API for both OpenAI and HuggingFace models
- **Backward Compatibility**: Maintains existing function signatures
- **Caching**: Intelligent caching to reduce API calls and improve performance
- **Batch Processing**: Efficient handling of multiple texts
- **Error Handling**: Comprehensive error handling with retry logic
- **Metrics**: Detailed metrics and monitoring capabilities
- **Memory Management**: Optimized memory usage, especially for HuggingFace models
- **Rate Limiting**: Graceful handling of API rate limits
- **Thread Safety**: Safe for concurrent operations
- **Type Safety**: Full type hints throughout

## Architecture

### Core Components

1. **BaseEmbedder**: Abstract base class defining the common interface
2. **OpenAIEmbedder**: OpenAI-specific implementation
3. **HuggingFaceEmbedder**: HuggingFace-specific implementation
4. **EmbeddingFactory**: Factory for creating embedders
5. **EmbeddingConfig**: Configuration management
6. **EmbeddingCache**: Thread-safe caching layer
7. **EmbeddingMetrics**: Metrics collection and reporting

### Class Hierarchy

```
BaseEmbedder (Abstract)
├── OpenAIEmbedder
└── HuggingFaceEmbedder
```

## Quick Start

### Basic Usage

```python
from backend.modules.embedding_factory import create_openai_embedder, create_huggingface_embedder

# Create OpenAI embedder
openai_embedder = create_openai_embedder(model_name="text-embedding-ada-002")

# Create HuggingFace embedder
hf_embedder = create_huggingface_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Embed single text
embedding = openai_embedder.embed_text("Your text here")

# Embed multiple texts
texts = ["Text 1", "Text 2", "Text 3"]
result = openai_embedder.embed_texts(texts)
```

### Backward Compatibility

The new interface maintains backward compatibility with existing code:

```python
# Legacy OpenAI usage still works
from backend.modules import embedder
embedding = embedder.embed_text("Your text")

# Legacy HuggingFace usage still works
from backend.utils import embedder_hf
embedding = embedder_hf.embed_text("Your text")
```

## Configuration

### EmbeddingConfig

```python
from backend.modules.base_embedder import EmbeddingConfig, EmbeddingProvider

config = EmbeddingConfig(
    provider=EmbeddingProvider.OPENAI,
    model_name="text-embedding-ada-002",
    max_tokens=8191,
    batch_size=100,
    cache_size=1000,
    retry_attempts=3,
    retry_delay=1.0,
    timeout=30.0,
    rate_limit_delay=0.1
)
```

### Configuration Parameters

- **provider**: Embedding provider (OPENAI or HUGGINGFACE)
- **model_name**: Name of the embedding model
- **max_tokens**: Maximum tokens per text
- **batch_size**: Batch size for processing multiple texts
- **cache_size**: Maximum number of cached embeddings
- **retry_attempts**: Number of retry attempts on failure
- **retry_delay**: Initial delay between retries (exponential backoff)
- **timeout**: Request timeout in seconds
- **rate_limit_delay**: Delay between requests to avoid rate limits
- **embedding_dimension**: Expected embedding dimension (auto-detected if not set)

## Supported Models

### OpenAI Models

- `text-embedding-ada-002` (1536 dimensions)
- `text-embedding-3-small` (1536 dimensions)
- `text-embedding-3-large` (3072 dimensions)

### HuggingFace Models

- `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- `sentence-transformers/all-mpnet-base-v2` (768 dimensions)
- `sentence-transformers/all-MiniLM-L12-v2` (384 dimensions)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions)
- `BAAI/bge-small-en-v1.5` (384 dimensions)
- `BAAI/bge-base-en-v1.5` (768 dimensions)
- `BAAI/bge-large-en-v1.5` (1024 dimensions)

## API Reference

### BaseEmbedder

#### Methods

- `embed_text(text: str, use_cache: bool = True) -> List[float]`
- `embed_texts(texts: List[str], use_cache: bool = True) -> EmbeddingResult`
- `embed_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]`
- `embed_goal_and_papers(goal: str, papers: List[Dict[str, Any]]) -> Tuple[List[float], List[List[float]]]`
- `get_metrics() -> EmbeddingMetrics`
- `get_cache_stats() -> Dict[str, Any]`
- `clear_cache() -> None`
- `reset_metrics() -> None`

### OpenAIEmbedder

Additional methods:
- `estimate_cost(token_count: int) -> float`
- `get_model_info() -> Dict[str, Any]`

### HuggingFaceEmbedder

Additional methods:
- `get_memory_usage() -> Dict[str, Any]`
- `clear_memory() -> None`
- `get_model_info() -> Dict[str, Any]`

### EmbeddingFactory

- `create_embedder(provider, model_name, config=None, **kwargs) -> BaseEmbedder`
- `create_openai_embedder(model_name, api_key=None, **kwargs) -> OpenAIEmbedder`
- `create_huggingface_embedder(model_name, device=None, **kwargs) -> HuggingFaceEmbedder`
- `get_available_models(provider) -> Dict[str, Any]`

## Features in Detail

### Caching

The caching system automatically stores embedding results to avoid redundant API calls:

```python
# First call hits the API
embedding1 = embedder.embed_text("Sample text")

# Second call uses cache
embedding2 = embedder.embed_text("Sample text")

# Check cache statistics
cache_stats = embedder.get_cache_stats()
print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
```

### Batch Processing

Efficient batch processing with configurable batch sizes:

```python
texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
result = embedder.embed_texts(texts)

print(f"Processed {len(result.embeddings)} texts")
print(f"Processing time: {result.processing_time:.2f}s")
print(f"Cache hits: {result.cache_hits}")
```

### Error Handling and Retries

Automatic retry with exponential backoff:

```python
config = EmbeddingConfig(
    provider=EmbeddingProvider.OPENAI,
    model_name="text-embedding-ada-002",
    retry_attempts=5,
    retry_delay=1.0  # Will retry with 1s, 2s, 4s, 8s, 16s delays
)
```

### Metrics

Comprehensive metrics collection:

```python
metrics = embedder.get_metrics()
print(f"Total requests: {metrics.total_requests}")
print(f"Success rate: {metrics.successful_requests / metrics.total_requests:.2%}")
print(f"Total tokens: {metrics.total_tokens}")
print(f"Average time per request: {metrics.total_time / metrics.total_requests:.2f}s")
```

### Memory Management (HuggingFace)

Optimized memory usage for HuggingFace models:

```python
# Check GPU memory usage
memory_stats = hf_embedder.get_memory_usage()
print(f"GPU allocated: {memory_stats.get('gpu_allocated', 0)} bytes")

# Clear GPU memory cache
hf_embedder.clear_memory()
```

## Best Practices

### 1. Choose the Right Model

- Use OpenAI for highest quality embeddings
- Use HuggingFace for cost-effective, local processing
- Consider embedding dimensions for your use case

### 2. Optimize Batch Sizes

- OpenAI: Use larger batch sizes (50-100)
- HuggingFace: Use smaller batch sizes (8-32) based on GPU memory

### 3. Configure Caching

- Set appropriate cache sizes based on your use case
- Clear cache periodically to free memory
- Monitor cache hit rates

### 4. Handle Errors Gracefully

- Set appropriate retry attempts and delays
- Monitor rate limits and adjust delays
- Use fallback strategies

### 5. Monitor Performance

- Track metrics regularly
- Monitor token usage and costs
- Optimize batch sizes based on performance

## Migration Guide

### From Legacy OpenAI Embedder

```python
# Old way
from backend.modules import embedder
embedding = embedder.embed_text("text")

# New way (optional - old way still works)
from backend.modules.embedding_factory import create_openai_embedder
embedder = create_openai_embedder()
embedding = embedder.embed_text("text")
```

### From Legacy HuggingFace Embedder

```python
# Old way
from backend.utils import embedder_hf
embedding = embedder_hf.embed_text("text")

# New way (optional - old way still works)
from backend.modules.embedding_factory import create_huggingface_embedder
embedder = create_huggingface_embedder()
embedding = embedder.embed_text("text")
```

## Troubleshooting

### Common Issues

1. **API Key Missing**: Ensure `OPENAI_API_KEY` is set in environment
2. **Rate Limits**: Increase `rate_limit_delay` in configuration
3. **Memory Issues**: Reduce batch sizes for HuggingFace models
4. **Model Not Found**: Check model name spelling and availability

### Performance Optimization

1. **Enable Caching**: Use appropriate cache sizes
2. **Batch Processing**: Group texts for batch processing
3. **Memory Management**: Clear GPU memory regularly for HuggingFace
4. **Rate Limiting**: Adjust delays to avoid rate limits

## Testing

Run the test suite:

```bash
python -m pytest backend/tests/test_embeddings.py -v
```

## Examples

See `backend/examples/embedding_usage.py` for comprehensive examples of all features.

## Environment Variables

- `OPENAI_API_KEY`: Required for OpenAI embeddings
- `CUDA_VISIBLE_DEVICES`: Control GPU usage for HuggingFace models

## Dependencies

### Core Dependencies
- `openai`: OpenAI API client
- `sentence-transformers`: HuggingFace sentence transformers
- `transformers`: HuggingFace transformers
- `torch`: PyTorch for HuggingFace models
- `numpy`: Numerical operations
- `tiktoken`: Token counting for OpenAI

### Optional Dependencies
- `faiss-cpu`: For similarity search
- `scikit-learn`: For metrics calculations

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the examples
3. Check the test suite for expected behavior
4. Refer to the API documentation