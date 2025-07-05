# backend/modules/embedding_factory.py
"""
Factory for creating embedders and managing configurations
"""

from typing import Dict, Any, Optional, Union
import os
import logging
from dotenv import load_dotenv

from .base_embedder import BaseEmbedder, EmbeddingConfig, EmbeddingProvider
from .openai_embedder import OpenAIEmbedder
from .huggingface_embedder import HuggingFaceEmbedder


class EmbeddingFactory:
    """Factory for creating embedders"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        load_dotenv()
    
    def create_embedder(
        self,
        provider: Union[str, EmbeddingProvider],
        model_name: str,
        config: Optional[EmbeddingConfig] = None,
        **kwargs
    ) -> BaseEmbedder:
        """Create an embedder instance"""
        
        # Convert string to enum if needed
        if isinstance(provider, str):
            try:
                provider = EmbeddingProvider(provider.lower())
            except ValueError:
                raise ValueError(f"Unknown provider: {provider}")
        
        # Create default config if not provided
        if config is None:
            config = self._create_default_config(provider, model_name, **kwargs)
        
        # Create embedder based on provider
        if provider == EmbeddingProvider.OPENAI:
            return self._create_openai_embedder(config, **kwargs)
        elif provider == EmbeddingProvider.HUGGINGFACE:
            return self._create_huggingface_embedder(config, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _create_default_config(
        self,
        provider: EmbeddingProvider,
        model_name: str,
        **kwargs
    ) -> EmbeddingConfig:
        """Create default configuration for provider"""
        
        if provider == EmbeddingProvider.OPENAI:
            config = OpenAIEmbedder.create_default_config(model_name)
        elif provider == EmbeddingProvider.HUGGINGFACE:
            config = HuggingFaceEmbedder.create_default_config(model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Override with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def _create_openai_embedder(self, config: EmbeddingConfig, **kwargs) -> OpenAIEmbedder:
        """Create OpenAI embedder"""
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        return OpenAIEmbedder(config, api_key)
    
    def _create_huggingface_embedder(self, config: EmbeddingConfig, **kwargs) -> HuggingFaceEmbedder:
        """Create HuggingFace embedder"""
        device = kwargs.get("device")
        return HuggingFaceEmbedder(config, device)
    
    def create_openai_embedder(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        **kwargs
    ) -> OpenAIEmbedder:
        """Convenience method to create OpenAI embedder"""
        return self.create_embedder(
            EmbeddingProvider.OPENAI,
            model_name,
            api_key=api_key,
            **kwargs
        )
    
    def create_huggingface_embedder(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        **kwargs
    ) -> HuggingFaceEmbedder:
        """Convenience method to create HuggingFace embedder"""
        return self.create_embedder(
            EmbeddingProvider.HUGGINGFACE,
            model_name,
            device=device,
            **kwargs
        )
    
    def get_available_models(self, provider: Union[str, EmbeddingProvider]) -> Dict[str, Any]:
        """Get available models for a provider"""
        if isinstance(provider, str):
            provider = EmbeddingProvider(provider.lower())
        
        if provider == EmbeddingProvider.OPENAI:
            return OpenAIEmbedder.MODEL_SPECS
        elif provider == EmbeddingProvider.HUGGINGFACE:
            return HuggingFaceEmbedder.MODEL_SPECS
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def validate_config(self, config: EmbeddingConfig) -> bool:
        """Validate embedding configuration"""
        try:
            # Basic validation
            if not config.model_name:
                raise ValueError("model_name is required")
            
            if config.max_tokens <= 0:
                raise ValueError("max_tokens must be positive")
            
            if config.batch_size <= 0:
                raise ValueError("batch_size must be positive")
            
            if config.retry_attempts < 0:
                raise ValueError("retry_attempts must be non-negative")
            
            # Provider-specific validation
            if config.provider == EmbeddingProvider.OPENAI:
                if not os.getenv("OPENAI_API_KEY"):
                    self.logger.warning("OPENAI_API_KEY not found in environment")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False


# Global factory instance
embedding_factory = EmbeddingFactory()


def create_embedder(
    provider: Union[str, EmbeddingProvider],
    model_name: str,
    config: Optional[EmbeddingConfig] = None,
    **kwargs
) -> BaseEmbedder:
    """Global function to create embedders"""
    return embedding_factory.create_embedder(provider, model_name, config, **kwargs)


def create_openai_embedder(
    model_name: str = "text-embedding-ada-002",
    api_key: Optional[str] = None,
    **kwargs
) -> OpenAIEmbedder:
    """Global function to create OpenAI embedders"""
    return embedding_factory.create_openai_embedder(model_name, api_key, **kwargs)


def create_huggingface_embedder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    **kwargs
) -> HuggingFaceEmbedder:
    """Global function to create HuggingFace embedders"""
    return embedding_factory.create_huggingface_embedder(model_name, device, **kwargs)


def get_available_models(provider: Union[str, EmbeddingProvider]) -> Dict[str, Any]:
    """Global function to get available models"""
    return embedding_factory.get_available_models(provider)