# utils/embedder_hf.py
# v0.4.0 - Unified Embedding Interface with Enhanced Features
# Maintains backward compatibility with existing function signatures

from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, models 
import logging
import os

# Import new unified embedding interface
from backend.modules.embedding_factory import create_huggingface_embedder, EmbeddingProvider
from backend.modules.base_embedder import EmbeddingConfig

try:
    import nltk
    nltk.data.path.append("/tmp/nltk_data")
    nltk.download("punkt", download_dir="/tmp/nltk_data")
except Exception as e:
    logging.error(f"❌ NLTK download failed: {e}")
    # Don't raise - allow the module to load even if NLTK fails
    pass

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Global embedder instance (lazy initialization)
_embedder = None

# Legacy model for backward compatibility
_legacy_model = None

def get_embedder():
    """Get or create the global embedder instance"""
    global _embedder
    if _embedder is None:
        try:
            _embedder = create_huggingface_embedder(
                model_name=MODEL_NAME,
                max_tokens=256,
                batch_size=16,
                cache_size=1000,
                retry_attempts=3
            )
            logger.info("Initialized unified HuggingFace embedder")
        except Exception as e:
            logger.error(f"Failed to initialize unified embedder: {e}")
            _embedder = None
    return _embedder

def get_legacy_model():
    """Get or create the legacy model for fallback"""
    global _legacy_model
    if _legacy_model is None:
        try:
            # Manually create the SentenceTransformer model
            word_embedding_model = models.Transformer(MODEL_NAME)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            _legacy_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            logger.info("Initialized legacy HuggingFace model")
        except Exception as e:
            logger.error(f"Failed to initialize legacy model: {e}")
            _legacy_model = None
    return _legacy_model

def embed_text(text: str) -> List[float]:
    """Embed a single text string (backward compatible)"""
    embedder = get_embedder()
    
    if embedder is not None:
        # Use new unified interface
        try:
            return embedder.embed_text(text)
        except Exception as e:
            logger.warning(f"Unified embedder failed, falling back to legacy: {e}")
    
    # Fallback to legacy implementation
    legacy_model = get_legacy_model()
    if legacy_model is None:
        raise RuntimeError("Both unified and legacy embedders failed to initialize")
    
    try:
        logging.info("🔵 Generating HF embedding with legacy model...")
        embedding = legacy_model.encode(text, convert_to_numpy=True).tolist()
        return embedding
    except Exception as e:
        logging.error(f"❌ HF embedding failed: {e}")
        raise

def embed_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Embed papers and add embeddings to paper objects (backward compatible)"""
    embedder = get_embedder()
    
    if embedder is not None:
        # Use new unified interface with batch processing
        try:
            logger.info("🔵 Generating embeddings with unified interface...")
            result = embedder.embed_papers(papers)
            logger.info(f"✅ Embeddings complete. Processed {len(papers)} papers.")
            return result
        except Exception as e:
            logger.warning(f"Unified embedder failed, falling back to legacy: {e}")
    
    # Fallback to legacy implementation
    logger.info("🔵 Generating embeddings with legacy model...")
    for paper in papers:
        paper["embedding"] = embed_text(paper["content"])
    logger.info(f"✅ Embeddings complete. Processed {len(papers)} papers.")
    return papers

def embed_goal_and_papers(goal: str, papers: List[Dict[str, Any]]) -> Tuple[List[float], List[List[float]]]:
    """Embed goal and papers (backward compatible)"""
    embedder = get_embedder()
    
    if embedder is not None:
        # Use new unified interface with batch processing
        try:
            logger.info("🔵 Embedding research goal and papers with unified interface...")
            goal_embedding, paper_embeddings = embedder.embed_goal_and_papers(goal, papers)
            logger.info(f"✅ Embedded goal and {len(paper_embeddings)} papers.")
            return goal_embedding, paper_embeddings
        except Exception as e:
            logger.warning(f"Unified embedder failed, falling back to legacy: {e}")
    
    # Fallback to legacy implementation
    logger.info("🔵 Embedding research goal and papers with legacy model...")
    goal_embedding = embed_text(goal)
    paper_embeddings = [embed_text(p["content"]) for p in papers]
    logger.info(f"✅ Embedded goal and {len(paper_embeddings)} papers.")
    return goal_embedding, paper_embeddings


# === UTILITY FUNCTIONS ===
def get_embedding_stats() -> Dict[str, Any]:
    """Get embedding statistics from the unified interface"""
    embedder = get_embedder()
    if embedder is not None:
        return {
            "metrics": embedder.get_metrics(),
            "cache_stats": embedder.get_cache_stats(),
            "model_info": embedder.get_model_info(),
            "memory_usage": embedder.get_memory_usage()
        }
    return {"error": "Unified embedder not available"}


def clear_embedding_cache() -> None:
    """Clear the embedding cache"""
    embedder = get_embedder()
    if embedder is not None:
        embedder.clear_cache()
        logger.info("Embedding cache cleared")


def reset_embedding_metrics() -> None:
    """Reset embedding metrics"""
    embedder = get_embedder()
    if embedder is not None:
        embedder.reset_metrics()
        logger.info("Embedding metrics reset")


def clear_memory() -> None:
    """Clear GPU memory cache"""
    embedder = get_embedder()
    if embedder is not None:
        embedder.clear_memory()
        logger.info("GPU memory cache cleared")


def get_model_info() -> Dict[str, Any]:
    """Get model information"""
    embedder = get_embedder()
    if embedder is not None:
        return embedder.get_model_info()
    return {"error": "Unified embedder not available"}


def switch_model(model_name: str) -> None:
    """Switch to a different HuggingFace model"""
    global _embedder
    try:
        _embedder = create_huggingface_embedder(
            model_name=model_name,
            max_tokens=256,
            batch_size=16,
            cache_size=1000,
            retry_attempts=3
        )
        logger.info(f"Switched to model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to switch to model {model_name}: {e}")
        raise