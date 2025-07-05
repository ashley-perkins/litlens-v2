# backend/tests/test_embeddings.py
"""
Comprehensive test suite for the unified embedding interface
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import time
from typing import List, Dict, Any

# Import the modules under test
from backend.modules.base_embedder import (
    BaseEmbedder, EmbeddingConfig, EmbeddingProvider, EmbeddingResult,
    EmbeddingMetrics, EmbeddingCache
)
from backend.modules.openai_embedder import OpenAIEmbedder
from backend.modules.huggingface_embedder import HuggingFaceEmbedder
from backend.modules.embedding_factory import EmbeddingFactory, create_embedder


class TestEmbeddingCache(unittest.TestCase):
    """Test the embedding cache functionality"""
    
    def setUp(self):
        self.cache = EmbeddingCache(max_size=3)
    
    def test_cache_set_get(self):
        """Test basic cache set and get operations"""
        embedding = [0.1, 0.2, 0.3]
        self.cache.set("test text", "model", "provider", embedding)
        
        result = self.cache.get("test text", "model", "provider")
        self.assertEqual(result, embedding)
    
    def test_cache_miss(self):
        """Test cache miss returns None"""
        result = self.cache.get("nonexistent", "model", "provider")
        self.assertIsNone(result)
    
    def test_cache_eviction(self):
        """Test cache eviction when max size is reached"""
        # Fill cache to max capacity
        for i in range(3):
            self.cache.set(f"text{i}", "model", "provider", [float(i)])
        
        # Add one more item (should evict oldest)
        self.cache.set("text3", "model", "provider", [3.0])
        
        # First item should be evicted
        result = self.cache.get("text0", "model", "provider")
        self.assertIsNone(result)
        
        # Last item should be present
        result = self.cache.get("text3", "model", "provider")
        self.assertEqual(result, [3.0])
    
    def test_cache_clear(self):
        """Test cache clear functionality"""
        self.cache.set("test", "model", "provider", [1.0])
        self.cache.clear()
        
        result = self.cache.get("test", "model", "provider")
        self.assertIsNone(result)
        self.assertEqual(self.cache.size(), 0)


class TestEmbeddingMetrics(unittest.TestCase):
    """Test the embedding metrics functionality"""
    
    def setUp(self):
        self.metrics = EmbeddingMetrics()
    
    def test_add_successful_request(self):
        """Test adding a successful request"""
        self.metrics.add_request(success=True, tokens=100, time_taken=1.5)
        
        self.assertEqual(self.metrics.total_requests, 1)
        self.assertEqual(self.metrics.successful_requests, 1)
        self.assertEqual(self.metrics.failed_requests, 0)
        self.assertEqual(self.metrics.total_tokens, 100)
        self.assertEqual(self.metrics.total_time, 1.5)
    
    def test_add_failed_request(self):
        """Test adding a failed request"""
        self.metrics.add_request(success=False, tokens=50, time_taken=0.5)
        
        self.assertEqual(self.metrics.total_requests, 1)
        self.assertEqual(self.metrics.successful_requests, 0)
        self.assertEqual(self.metrics.failed_requests, 1)
        self.assertEqual(self.metrics.total_tokens, 50)
        self.assertEqual(self.metrics.total_time, 0.5)
    
    def test_cache_hits_misses(self):
        """Test cache hit/miss tracking"""
        self.metrics.add_request(success=True, tokens=100, time_taken=1.0, from_cache=True)
        self.metrics.add_request(success=True, tokens=100, time_taken=1.0, from_cache=False)
        
        self.assertEqual(self.metrics.cache_hits, 1)
        self.assertEqual(self.metrics.cache_misses, 1)


class TestEmbeddingConfig(unittest.TestCase):
    """Test the embedding configuration"""
    
    def test_valid_config(self):
        """Test creating a valid configuration"""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model_name="test-model",
            max_tokens=1000,
            batch_size=32
        )
        
        self.assertEqual(config.provider, EmbeddingProvider.OPENAI)
        self.assertEqual(config.model_name, "test-model")
        self.assertEqual(config.max_tokens, 1000)
        self.assertEqual(config.batch_size, 32)
    
    def test_invalid_batch_size(self):
        """Test invalid batch size raises error"""
        with self.assertRaises(ValueError):
            EmbeddingConfig(
                provider=EmbeddingProvider.OPENAI,
                model_name="test-model",
                max_tokens=1000,
                batch_size=-1
            )
    
    def test_invalid_retry_attempts(self):
        """Test invalid retry attempts raises error"""
        with self.assertRaises(ValueError):
            EmbeddingConfig(
                provider=EmbeddingProvider.OPENAI,
                model_name="test-model",
                max_tokens=1000,
                retry_attempts=-1
            )


class MockEmbedder(BaseEmbedder):
    """Mock embedder for testing base functionality"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.embedding_dimension = 3
    
    def _embed_single(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]
    
    def _get_token_count(self, text: str) -> int:
        return len(text.split())
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        words = text.split()
        return " ".join(words[:max_tokens])
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dimension


class TestBaseEmbedder(unittest.TestCase):
    """Test the base embedder functionality"""
    
    def setUp(self):
        self.config = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model_name="test-model",
            max_tokens=10,
            batch_size=2,
            cache_size=5
        )
        self.embedder = MockEmbedder(self.config)
    
    def test_embed_single_text(self):
        """Test embedding a single text"""
        result = self.embedder.embed_text("test text")
        self.assertEqual(result, [0.1, 0.2, 0.3])
    
    def test_embed_multiple_texts(self):
        """Test embedding multiple texts"""
        texts = ["text1", "text2", "text3"]
        result = self.embedder.embed_texts(texts)
        
        self.assertIsInstance(result, EmbeddingResult)
        self.assertEqual(len(result.embeddings), 3)
        self.assertEqual(result.dimensions, 3)
    
    def test_text_truncation(self):
        """Test text truncation when exceeding max tokens"""
        long_text = " ".join(["word"] * 20)  # 20 words, max is 10
        result = self.embedder.embed_text(long_text)
        self.assertEqual(result, [0.1, 0.2, 0.3])
    
    def test_caching(self):
        """Test embedding caching"""
        # First call should cache the result
        result1 = self.embedder.embed_text("test text")
        
        # Second call should use cache
        result2 = self.embedder.embed_text("test text")
        
        self.assertEqual(result1, result2)
        self.assertEqual(self.embedder.cache.size(), 1)
    
    def test_batch_processing(self):
        """Test batch processing with different batch sizes"""
        texts = ["text1", "text2", "text3", "text4", "text5"]
        result = self.embedder.embed_texts(texts)
        
        self.assertEqual(len(result.embeddings), 5)
        for embedding in result.embeddings:
            self.assertEqual(embedding, [0.1, 0.2, 0.3])
    
    def test_empty_text_error(self):
        """Test error handling for empty text"""
        with self.assertRaises(ValueError):
            self.embedder.embed_text("")
    
    def test_empty_texts_error(self):
        """Test error handling for empty text list"""
        with self.assertRaises(ValueError):
            self.embedder.embed_texts([])
    
    def test_embed_papers(self):
        """Test embedding papers"""
        papers = [
            {"content": "paper 1 content"},
            {"content": "paper 2 content"}
        ]
        
        result = self.embedder.embed_papers(papers)
        
        self.assertEqual(len(result), 2)
        for paper in result:
            self.assertIn("embedding", paper)
            self.assertEqual(paper["embedding"], [0.1, 0.2, 0.3])
    
    def test_embed_goal_and_papers(self):
        """Test embedding goal and papers"""
        goal = "research goal"
        papers = [{"content": "paper content"}]
        
        goal_embedding, paper_embeddings = self.embedder.embed_goal_and_papers(goal, papers)
        
        self.assertEqual(goal_embedding, [0.1, 0.2, 0.3])
        self.assertEqual(len(paper_embeddings), 1)
        self.assertEqual(paper_embeddings[0], [0.1, 0.2, 0.3])


class TestOpenAIEmbedder(unittest.TestCase):
    """Test OpenAI embedder implementation"""
    
    def setUp(self):
        self.config = OpenAIEmbedder.create_default_config()
        self.api_key = "test-api-key"
    
    @patch('backend.modules.openai_embedder.OpenAI')
    def test_initialization(self, mock_openai):
        """Test OpenAI embedder initialization"""
        embedder = OpenAIEmbedder(self.config, self.api_key)
        
        self.assertEqual(embedder.config.provider, EmbeddingProvider.OPENAI)
        mock_openai.assert_called_once_with(api_key=self.api_key)
    
    def test_no_api_key_error(self):
        """Test error when no API key is provided"""
        with self.assertRaises(ValueError):
            OpenAIEmbedder(self.config, "")
    
    def test_wrong_provider_error(self):
        """Test error when wrong provider is set"""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.HUGGINGFACE,
            model_name="test-model",
            max_tokens=1000
        )
        
        with self.assertRaises(ValueError):
            OpenAIEmbedder(config, self.api_key)
    
    @patch('backend.modules.openai_embedder.OpenAI')
    def test_get_embedding_dimension(self, mock_openai):
        """Test getting embedding dimension"""
        embedder = OpenAIEmbedder(self.config, self.api_key)
        dimension = embedder.get_embedding_dimension()
        
        self.assertEqual(dimension, 1536)  # Default for ada-002
    
    @patch('backend.modules.openai_embedder.OpenAI')
    def test_model_info(self, mock_openai):
        """Test getting model information"""
        embedder = OpenAIEmbedder(self.config, self.api_key)
        info = embedder.get_model_info()
        
        self.assertEqual(info["provider"], "openai")
        self.assertEqual(info["model"], "text-embedding-ada-002")
        self.assertEqual(info["dimension"], 1536)
    
    @patch('backend.modules.openai_embedder.OpenAI')
    def test_cost_estimation(self, mock_openai):
        """Test cost estimation"""
        embedder = OpenAIEmbedder(self.config, self.api_key)
        cost = embedder.estimate_cost(1000)
        
        self.assertGreater(cost, 0)
        self.assertIsInstance(cost, float)


class TestHuggingFaceEmbedder(unittest.TestCase):
    """Test HuggingFace embedder implementation"""
    
    def setUp(self):
        self.config = HuggingFaceEmbedder.create_default_config()
    
    @patch('backend.modules.huggingface_embedder.SentenceTransformer')
    def test_initialization(self, mock_sentence_transformer):
        """Test HuggingFace embedder initialization"""
        mock_model = Mock()
        mock_model.tokenizer = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        embedder = HuggingFaceEmbedder(self.config)
        
        self.assertEqual(embedder.config.provider, EmbeddingProvider.HUGGINGFACE)
        self.assertIsNotNone(embedder.model)
    
    def test_wrong_provider_error(self):
        """Test error when wrong provider is set"""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.OPENAI,
            model_name="test-model",
            max_tokens=1000
        )
        
        with self.assertRaises(ValueError):
            HuggingFaceEmbedder(config)
    
    @patch('backend.modules.huggingface_embedder.SentenceTransformer')
    def test_get_embedding_dimension(self, mock_sentence_transformer):
        """Test getting embedding dimension"""
        mock_model = Mock()
        mock_model.tokenizer = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        embedder = HuggingFaceEmbedder(self.config)
        dimension = embedder.get_embedding_dimension()
        
        self.assertEqual(dimension, 384)
    
    @patch('backend.modules.huggingface_embedder.SentenceTransformer')
    def test_model_info(self, mock_sentence_transformer):
        """Test getting model information"""
        mock_model = Mock()
        mock_model.tokenizer = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        embedder = HuggingFaceEmbedder(self.config)
        info = embedder.get_model_info()
        
        self.assertEqual(info["provider"], "huggingface")
        self.assertEqual(info["model"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertIn("device", info)


class TestEmbeddingFactory(unittest.TestCase):
    """Test the embedding factory"""
    
    def setUp(self):
        self.factory = EmbeddingFactory()
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_create_openai_embedder(self):
        """Test creating OpenAI embedder through factory"""
        with patch('backend.modules.openai_embedder.OpenAI'):
            embedder = self.factory.create_openai_embedder()
            self.assertIsInstance(embedder, OpenAIEmbedder)
    
    @patch('backend.modules.huggingface_embedder.SentenceTransformer')
    def test_create_huggingface_embedder(self, mock_sentence_transformer):
        """Test creating HuggingFace embedder through factory"""
        mock_model = Mock()
        mock_model.tokenizer = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        embedder = self.factory.create_huggingface_embedder()
        self.assertIsInstance(embedder, HuggingFaceEmbedder)
    
    def test_get_available_models(self):
        """Test getting available models"""
        openai_models = self.factory.get_available_models("openai")
        hf_models = self.factory.get_available_models("huggingface")
        
        self.assertIsInstance(openai_models, dict)
        self.assertIsInstance(hf_models, dict)
        self.assertIn("text-embedding-ada-002", openai_models)
        self.assertIn("sentence-transformers/all-MiniLM-L6-v2", hf_models)
    
    def test_invalid_provider(self):
        """Test error for invalid provider"""
        with self.assertRaises(ValueError):
            self.factory.create_embedder("invalid_provider", "test-model")


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing interfaces"""
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch('backend.modules.embedder.OpenAI')
    def test_openai_embedder_compatibility(self, mock_openai):
        """Test OpenAI embedder backward compatibility"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client = Mock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Import and test the module
        from backend.modules import embedder
        
        # Test single text embedding
        result = embedder.embed_text("test text")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        
        # Test papers embedding
        papers = [{"content": "paper 1"}, {"content": "paper 2"}]
        result = embedder.embed_papers(papers)
        self.assertEqual(len(result), 2)
        for paper in result:
            self.assertIn("embedding", paper)
        
        # Test goal and papers embedding
        goal_emb, paper_embs = embedder.embed_goal_and_papers("goal", papers)
        self.assertIsInstance(goal_emb, list)
        self.assertIsInstance(paper_embs, list)
        self.assertEqual(len(paper_embs), 2)
    
    @patch('backend.utils.embedder_hf.SentenceTransformer')
    def test_huggingface_embedder_compatibility(self, mock_sentence_transformer):
        """Test HuggingFace embedder backward compatibility"""
        # Mock SentenceTransformer
        mock_model = Mock()
        mock_model.encode.return_value = Mock()
        mock_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
        mock_sentence_transformer.return_value = mock_model
        
        # Import and test the module
        from backend.utils import embedder_hf
        
        # Test single text embedding
        result = embedder_hf.embed_text("test text")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        
        # Test papers embedding
        papers = [{"content": "paper 1"}, {"content": "paper 2"}]
        result = embedder_hf.embed_papers(papers)
        self.assertEqual(len(result), 2)
        for paper in result:
            self.assertIn("embedding", paper)


if __name__ == '__main__':
    unittest.main()