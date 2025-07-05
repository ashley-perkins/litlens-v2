# backend/modules/openai_embedder.py
"""
OpenAI embedder implementation using the unified embedding interface
"""

from typing import List, Dict, Any, Optional
import tiktoken
import logging
import time
from openai import OpenAI
from openai.types import CreateEmbeddingResponse

from .base_embedder import BaseEmbedder, EmbeddingConfig, EmbeddingProvider


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedder implementation"""
    
    # Model specifications
    MODEL_SPECS = {
        "text-embedding-ada-002": {
            "max_tokens": 8191,
            "dimension": 1536,
            "cost_per_1k_tokens": 0.0001
        },
        "text-embedding-3-small": {
            "max_tokens": 8191,
            "dimension": 1536,
            "cost_per_1k_tokens": 0.00002
        },
        "text-embedding-3-large": {
            "max_tokens": 8191,
            "dimension": 3072,
            "cost_per_1k_tokens": 0.00013
        }
    }
    
    def __init__(self, config: EmbeddingConfig, api_key: str):
        """Initialize OpenAI embedder"""
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        # Set default values for OpenAI
        if config.provider != EmbeddingProvider.OPENAI:
            raise ValueError("Config must have provider set to OPENAI")
        
        # Set embedding dimension if not provided
        if config.embedding_dimension is None and config.model_name in self.MODEL_SPECS:
            config.embedding_dimension = self.MODEL_SPECS[config.model_name]["dimension"]
        
        super().__init__(config)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(config.model_name)
        except KeyError:
            # Fallback to ada-002 tokenizer for unknown models
            self.logger.warning(f"Unknown model {config.model_name}, using ada-002 tokenizer")
            self.tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
        
        self.logger.info(f"Initialized OpenAI embedder with model: {config.model_name}")
    
    def _get_token_count(self, text: str) -> int:
        """Get token count for text"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            self.logger.warning(f"Failed to count tokens: {e}")
            # Fallback estimation (rough approximation)
            return len(text.split()) * 4 // 3
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens"""
        try:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
            
            truncated_tokens = tokens[:max_tokens]
            return self.tokenizer.decode(truncated_tokens)
        except Exception as e:
            self.logger.warning(f"Failed to truncate text: {e}")
            # Fallback: truncate by characters (rough approximation)
            estimated_chars = max_tokens * 4
            return text[:estimated_chars]
    
    def _handle_openai_response(self, response: CreateEmbeddingResponse) -> List[List[float]]:
        """Handle OpenAI API response"""
        embeddings = []
        for embedding_data in response.data:
            embeddings.append(embedding_data.embedding)
        return embeddings
    
    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text string using OpenAI API"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.config.model_name,
                timeout=self.config.timeout
            )
            
            embeddings = self._handle_openai_response(response)
            if not embeddings:
                raise ValueError("No embeddings returned from OpenAI API")
            
            return embeddings[0]
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            # Check for rate limiting
            if "rate_limit" in str(e).lower():
                self.metrics.rate_limit_hits += 1
                self.logger.warning("Rate limit hit, increasing delay")
                time.sleep(self.config.retry_delay * 2)
            raise
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts using OpenAI API"""
        if not texts:
            return []
        
        try:
            # OpenAI API supports batch embedding
            response = self.client.embeddings.create(
                input=texts,
                model=self.config.model_name,
                timeout=self.config.timeout
            )
            
            embeddings = self._handle_openai_response(response)
            
            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Mismatch in embedding count: expected {len(texts)}, got {len(embeddings)}"
                )
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"OpenAI batch API error: {e}")
            # Check for rate limiting
            if "rate_limit" in str(e).lower():
                self.metrics.rate_limit_hits += 1
                self.logger.warning("Rate limit hit, increasing delay")
                time.sleep(self.config.retry_delay * 2)
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension for the model"""
        if self.config.embedding_dimension:
            return self.config.embedding_dimension
        
        if self.config.model_name in self.MODEL_SPECS:
            return self.MODEL_SPECS[self.config.model_name]["dimension"]
        
        # Default fallback
        return 1536
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "provider": self.config.provider.value,
            "model": self.config.model_name,
            "dimension": self.get_embedding_dimension(),
            "max_tokens": self.config.max_tokens,
            "specs": self.MODEL_SPECS.get(self.config.model_name, {})
        }
    
    def estimate_cost(self, token_count: int) -> float:
        """Estimate cost for embedding tokens"""
        if self.config.model_name not in self.MODEL_SPECS:
            return 0.0
        
        cost_per_1k = self.MODEL_SPECS[self.config.model_name]["cost_per_1k_tokens"]
        return (token_count / 1000) * cost_per_1k
    
    @classmethod
    def create_default_config(cls, model_name: str = "text-embedding-ada-002") -> EmbeddingConfig:
        """Create default configuration for OpenAI embedder"""
        max_tokens = 8191
        embedding_dimension = None
        
        if model_name in cls.MODEL_SPECS:
            max_tokens = cls.MODEL_SPECS[model_name]["max_tokens"]
            embedding_dimension = cls.MODEL_SPECS[model_name]["dimension"]
        
        return EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model_name=model_name,
            max_tokens=max_tokens,
            batch_size=100,  # OpenAI supports up to 2048 inputs per batch
            cache_size=1000,
            retry_attempts=3,
            retry_delay=1.0,
            timeout=30.0,
            rate_limit_delay=0.1,
            embedding_dimension=embedding_dimension
        )