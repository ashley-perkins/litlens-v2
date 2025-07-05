# backend/modules/huggingface_embedder.py
"""
HuggingFace embedder implementation using the unified embedding interface
"""

from typing import List, Dict, Any, Optional
import logging
import gc
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np

from .base_embedder import BaseEmbedder, EmbeddingConfig, EmbeddingProvider


class HuggingFaceEmbedder(BaseEmbedder):
    """HuggingFace embedder implementation"""
    
    # Model specifications
    MODEL_SPECS = {
        "sentence-transformers/all-MiniLM-L6-v2": {
            "max_tokens": 256,
            "dimension": 384,
            "type": "sentence_transformer"
        },
        "sentence-transformers/all-mpnet-base-v2": {
            "max_tokens": 384,
            "dimension": 768,
            "type": "sentence_transformer"
        },
        "sentence-transformers/all-MiniLM-L12-v2": {
            "max_tokens": 256,
            "dimension": 384,
            "type": "sentence_transformer"
        },
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
            "max_tokens": 128,
            "dimension": 384,
            "type": "sentence_transformer"
        },
        "BAAI/bge-small-en-v1.5": {
            "max_tokens": 512,
            "dimension": 384,
            "type": "sentence_transformer"
        },
        "BAAI/bge-base-en-v1.5": {
            "max_tokens": 512,
            "dimension": 768,
            "type": "sentence_transformer"
        },
        "BAAI/bge-large-en-v1.5": {
            "max_tokens": 512,
            "dimension": 1024,
            "type": "sentence_transformer"
        }
    }
    
    def __init__(self, config: EmbeddingConfig, device: Optional[str] = None):
        """Initialize HuggingFace embedder"""
        if config.provider != EmbeddingProvider.HUGGINGFACE:
            raise ValueError("Config must have provider set to HUGGINGFACE")
        
        # Set embedding dimension if not provided
        if config.embedding_dimension is None and config.model_name in self.MODEL_SPECS:
            config.embedding_dimension = self.MODEL_SPECS[config.model_name]["dimension"]
        
        super().__init__(config)
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self._load_model()
        
        self.logger.info(f"Initialized HuggingFace embedder with model: {config.model_name}")
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            model_type = self.MODEL_SPECS.get(self.config.model_name, {}).get("type", "sentence_transformer")
            
            if model_type == "sentence_transformer" or "sentence-transformers" in self.config.model_name:
                # Use SentenceTransformer for sentence-transformers models
                self.model = SentenceTransformer(self.config.model_name, device=self.device)
                self.tokenizer = self.model.tokenizer
            else:
                # Use AutoModel for other HuggingFace models
                self.model = AutoModel.from_pretrained(self.config.model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                self.model.to(self.device)
                self.model.eval()
            
        except Exception as e:
            self.logger.error(f"Failed to load model {self.config.model_name}: {e}")
            raise
    
    def _get_token_count(self, text: str) -> int:
        """Get token count for text"""
        try:
            if self.tokenizer is None:
                # Fallback estimation
                return len(text.split()) * 4 // 3
            
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)
        except Exception as e:
            self.logger.warning(f"Failed to count tokens: {e}")
            # Fallback estimation
            return len(text.split()) * 4 // 3
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to max tokens"""
        try:
            if self.tokenizer is None:
                # Fallback: truncate by characters (rough approximation)
                estimated_chars = max_tokens * 4
                return text[:estimated_chars]
            
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) <= max_tokens:
                return text
            
            # Account for special tokens
            max_content_tokens = max_tokens - 2  # CLS and SEP tokens
            truncated_tokens = tokens[1:max_content_tokens + 1]  # Skip CLS, keep content
            return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        except Exception as e:
            self.logger.warning(f"Failed to truncate text: {e}")
            # Fallback: truncate by characters (rough approximation)
            estimated_chars = max_tokens * 4
            return text[:estimated_chars]
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for transformer models"""
        token_embeddings = model_output[0]  # First element contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _embed_with_sentence_transformer(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using SentenceTransformer"""
        try:
            # Handle memory efficiently for large batches
            if len(texts) > self.config.batch_size:
                all_embeddings = []
                for i in range(0, len(texts), self.config.batch_size):
                    batch = texts[i:i + self.config.batch_size]
                    batch_embeddings = self.model.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=min(len(batch), 32)  # Internal batch size
                    )
                    all_embeddings.extend(batch_embeddings.tolist())
                    
                    # Clean up memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                return all_embeddings
            else:
                embeddings = self.model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=min(len(texts), 32)
                )
                return embeddings.tolist()
                
        except Exception as e:
            self.logger.error(f"SentenceTransformer embedding failed: {e}")
            raise
    
    def _embed_with_auto_model(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using AutoModel"""
        try:
            all_embeddings = []
            
            with torch.no_grad():
                for i in range(0, len(texts), self.config.batch_size):
                    batch = texts[i:i + self.config.batch_size]
                    
                    # Tokenize batch
                    encoded_input = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=self.config.max_tokens
                    )
                    
                    # Move to device
                    for key in encoded_input:
                        encoded_input[key] = encoded_input[key].to(self.device)
                    
                    # Get model output
                    model_output = self.model(**encoded_input)
                    
                    # Apply mean pooling
                    embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                    
                    # Normalize embeddings
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                    # Convert to list
                    batch_embeddings = embeddings.cpu().numpy().tolist()
                    all_embeddings.extend(batch_embeddings)
                    
                    # Clean up memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"AutoModel embedding failed: {e}")
            raise
    
    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text string"""
        embeddings = self._embed_batch([text])
        return embeddings[0]
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts"""
        if not texts:
            return []
        
        try:
            # Choose embedding method based on model type
            if isinstance(self.model, SentenceTransformer):
                return self._embed_with_sentence_transformer(texts)
            else:
                return self._embed_with_auto_model(texts)
                
        except Exception as e:
            self.logger.error(f"Batch embedding failed: {e}")
            raise
        finally:
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension for the model"""
        if self.config.embedding_dimension:
            return self.config.embedding_dimension
        
        if self.config.model_name in self.MODEL_SPECS:
            return self.MODEL_SPECS[self.config.model_name]["dimension"]
        
        # Try to get dimension from model
        try:
            if isinstance(self.model, SentenceTransformer):
                return self.model.get_sentence_embedding_dimension()
            else:
                return self.model.config.hidden_size
        except:
            # Default fallback
            return 768
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "provider": self.config.provider.value,
            "model": self.config.model_name,
            "dimension": self.get_embedding_dimension(),
            "max_tokens": self.config.max_tokens,
            "device": self.device,
            "specs": self.MODEL_SPECS.get(self.config.model_name, {})
        }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        memory_stats = {}
        
        if torch.cuda.is_available():
            memory_stats.update({
                "gpu_allocated": torch.cuda.memory_allocated(self.device),
                "gpu_cached": torch.cuda.memory_reserved(self.device),
                "gpu_max_allocated": torch.cuda.max_memory_allocated(self.device),
                "gpu_max_cached": torch.cuda.max_memory_reserved(self.device)
            })
        
        return memory_stats
    
    def clear_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @classmethod
    def create_default_config(cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingConfig:
        """Create default configuration for HuggingFace embedder"""
        max_tokens = 256
        embedding_dimension = None
        
        if model_name in cls.MODEL_SPECS:
            max_tokens = cls.MODEL_SPECS[model_name]["max_tokens"]
            embedding_dimension = cls.MODEL_SPECS[model_name]["dimension"]
        
        return EmbeddingConfig(
            provider=EmbeddingProvider.HUGGINGFACE,
            model_name=model_name,
            max_tokens=max_tokens,
            batch_size=16,  # Smaller batch size for memory efficiency
            cache_size=1000,
            retry_attempts=3,
            retry_delay=1.0,
            timeout=60.0,  # Longer timeout for model inference
            rate_limit_delay=0.0,  # No rate limiting for local models
            embedding_dimension=embedding_dimension
        )