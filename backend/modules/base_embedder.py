# backend/modules/base_embedder.py
"""
Unified embedding interface for LitLens
Provides abstract base class and common functionality for OpenAI and HuggingFace embedders
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import logging
import threading
import time
from functools import lru_cache
import numpy as np


class EmbeddingProvider(Enum):
    """Supported embedding providers"""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""
    provider: EmbeddingProvider
    model_name: str
    max_tokens: int
    batch_size: int = 32
    cache_size: int = 1000
    retry_attempts: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    rate_limit_delay: float = 0.1
    embedding_dimension: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.retry_attempts < 0:
            raise ValueError("retry_attempts must be non-negative")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")


@dataclass
class EmbeddingResult:
    """Result of embedding operation"""
    embeddings: List[List[float]]
    dimensions: int
    provider: str
    model: str
    tokens_used: int
    processing_time: float
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass
class EmbeddingMetrics:
    """Metrics for embedding operations"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    rate_limit_hits: int = 0
    
    def add_request(self, success: bool, tokens: int, time_taken: float, from_cache: bool = False):
        """Add metrics for a single request"""
        self.total_requests += 1
        self.total_tokens += tokens
        self.total_time += time_taken
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            
        if from_cache:
            self.cache_hits += 1
        else:
            self.cache_misses += 1


class EmbeddingCache:
    """Thread-safe cache for embedding results"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, List[float]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def _generate_key(self, text: str, model: str, provider: str) -> str:
        """Generate cache key for text and model combination"""
        content = f"{provider}:{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, model: str, provider: str) -> Optional[List[float]]:
        """Get embedding from cache"""
        key = self._generate_key(text, model, provider)
        with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                return self._cache[key]
            return None
    
    def set(self, text: str, model: str, provider: str, embedding: List[float]) -> None:
        """Set embedding in cache"""
        key = self._generate_key(text, model, provider)
        with self._lock:
            # Remove oldest entries if cache is full
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._access_times.keys(), key=self._access_times.get)
                del self._cache[oldest_key]
                del self._access_times[oldest_key]
            
            self._cache[key] = embedding
            self._access_times[key] = time.time()
    
    def clear(self) -> None:
        """Clear all cached embeddings"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        with self._lock:
            return len(self._cache)


class BaseEmbedder(ABC):
    """Abstract base class for all embedding providers"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.metrics = EmbeddingMetrics()
        self.cache = EmbeddingCache(config.cache_size)
        self._lock = threading.RLock()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate embedding configuration"""
        if not self.config.model_name:
            raise ValueError("model_name cannot be empty")
        if self.config.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
    
    @abstractmethod
    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text string (implementation specific)"""
        pass
    
    @abstractmethod
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts (implementation specific)"""
        pass
    
    @abstractmethod
    def _get_token_count(self, text: str) -> int:
        """Get token count for text (implementation specific)"""
        pass
    
    @abstractmethod
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens (implementation specific)"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension for the model"""
        pass
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding"""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Truncate if necessary
        token_count = self._get_token_count(text)
        if token_count > self.config.max_tokens:
            self.logger.warning(
                f"Truncating text from {token_count} to {self.config.max_tokens} tokens"
            )
            text = self._truncate_text(text, self.config.max_tokens)
        
        return text.strip()
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic"""
        last_exception = None
        
        for attempt in range(self.config.retry_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.retry_attempts:
                    delay = self.config.retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(f"All {self.config.retry_attempts + 1} attempts failed")
        
        raise last_exception
    
    def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """Embed a single text string"""
        start_time = time.time()
        
        try:
            # Preprocess text
            text = self._preprocess_text(text)
            
            # Check cache first
            if use_cache:
                cached_embedding = self.cache.get(
                    text, self.config.model_name, self.config.provider.value
                )
                if cached_embedding is not None:
                    processing_time = time.time() - start_time
                    self.metrics.add_request(True, self._get_token_count(text), processing_time, from_cache=True)
                    return cached_embedding
            
            # Generate embedding
            embedding = self._retry_with_backoff(self._embed_single, text)
            
            # Validate embedding dimension
            if self.config.embedding_dimension:
                if len(embedding) != self.config.embedding_dimension:
                    raise ValueError(
                        f"Expected embedding dimension {self.config.embedding_dimension}, "
                        f"got {len(embedding)}"
                    )
            
            # Cache result
            if use_cache:
                self.cache.set(text, self.config.model_name, self.config.provider.value, embedding)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.add_request(True, self._get_token_count(text), processing_time, from_cache=False)
            
            return embedding
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.add_request(False, 0, processing_time, from_cache=False)
            self.logger.error(f"Failed to embed text: {e}")
            raise
    
    def embed_texts(self, texts: List[str], use_cache: bool = True) -> EmbeddingResult:
        """Embed multiple texts efficiently"""
        if not texts:
            raise ValueError("texts cannot be empty")
        
        start_time = time.time()
        all_embeddings = []
        total_tokens = 0
        cache_hits = 0
        cache_misses = 0
        
        try:
            # Process texts in batches
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                batch_embeddings = []
                uncached_texts = []
                uncached_indices = []
                
                # Check cache for each text in batch
                for j, text in enumerate(batch):
                    text = self._preprocess_text(text)
                    
                    if use_cache:
                        cached_embedding = self.cache.get(
                            text, self.config.model_name, self.config.provider.value
                        )
                        if cached_embedding is not None:
                            batch_embeddings.append(cached_embedding)
                            cache_hits += 1
                            continue
                    
                    uncached_texts.append(text)
                    uncached_indices.append(j)
                    cache_misses += 1
                
                # Embed uncached texts
                if uncached_texts:
                    uncached_embeddings = self._retry_with_backoff(self._embed_batch, uncached_texts)
                    
                    # Cache new embeddings
                    if use_cache:
                        for text, embedding in zip(uncached_texts, uncached_embeddings):
                            self.cache.set(
                                text, self.config.model_name, self.config.provider.value, embedding
                            )
                    
                    # Merge cached and uncached embeddings
                    for idx, embedding in zip(uncached_indices, uncached_embeddings):
                        while len(batch_embeddings) <= idx:
                            batch_embeddings.append(None)
                        batch_embeddings[idx] = embedding
                
                all_embeddings.extend(batch_embeddings)
                
                # Count tokens
                for text in batch:
                    total_tokens += self._get_token_count(text)
                
                # Rate limiting
                if self.config.rate_limit_delay > 0:
                    time.sleep(self.config.rate_limit_delay)
            
            # Validate all embeddings have same dimension
            if all_embeddings:
                expected_dim = len(all_embeddings[0])
                for i, embedding in enumerate(all_embeddings):
                    if len(embedding) != expected_dim:
                        raise ValueError(
                            f"Inconsistent embedding dimensions: "
                            f"embedding {i} has {len(embedding)} dims, expected {expected_dim}"
                        )
            
            processing_time = time.time() - start_time
            
            # Update metrics
            self.metrics.add_request(True, total_tokens, processing_time, from_cache=False)
            
            return EmbeddingResult(
                embeddings=all_embeddings,
                dimensions=len(all_embeddings[0]) if all_embeddings else 0,
                provider=self.config.provider.value,
                model=self.config.model_name,
                tokens_used=total_tokens,
                processing_time=processing_time,
                cache_hits=cache_hits,
                cache_misses=cache_misses
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.add_request(False, total_tokens, processing_time, from_cache=False)
            self.logger.error(f"Failed to embed texts: {e}")
            raise
    
    def embed_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Embed papers and add embeddings to paper objects (backward compatibility)"""
        if not papers:
            return papers
        
        # Extract content from papers
        texts = []
        for paper in papers:
            if "content" not in paper:
                raise ValueError("Paper must have 'content' field")
            texts.append(paper["content"])
        
        # Generate embeddings
        result = self.embed_texts(texts)
        
        # Add embeddings to papers
        for paper, embedding in zip(papers, result.embeddings):
            paper["embedding"] = embedding
        
        return papers
    
    def embed_goal_and_papers(self, goal: str, papers: List[Dict[str, Any]]) -> Tuple[List[float], List[List[float]]]:
        """Embed goal and papers (backward compatibility)"""
        # Embed goal
        goal_embedding = self.embed_text(goal)
        
        # Embed papers
        paper_texts = [paper["content"] for paper in papers]
        paper_result = self.embed_texts(paper_texts)
        
        return goal_embedding, paper_result.embeddings
    
    def get_metrics(self) -> EmbeddingMetrics:
        """Get embedding metrics"""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset embedding metrics"""
        self.metrics = EmbeddingMetrics()
    
    def clear_cache(self) -> None:
        """Clear embedding cache"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": self.cache.size(),
            "max_size": self.cache.max_size,
            "hit_rate": self.metrics.cache_hits / max(1, self.metrics.total_requests),
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses
        }