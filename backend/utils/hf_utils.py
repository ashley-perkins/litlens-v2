"""
Enhanced HuggingFace Utils Module

This module provides integration with the unified summarization system for HuggingFace models.
It maintains backward compatibility while leveraging the new robust error handling and features.

Version: 2.0.0
"""

import httpx
import os
import logging
from typing import Optional, Dict, Any

# Import the new unified summarization system
from backend.modules.summarizer import (
    SummarizationConfig,
    SummarizationProvider,
    HuggingFaceSummarizer,
    summarization_manager,
    get_available_models
)

# Configure logging
logger = logging.getLogger(__name__)

# Check for API token
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not API_TOKEN:
    logger.warning("HUGGINGFACE_API_TOKEN not found in environment variables. HuggingFace functionality will be limited.")

# Available models for quick reference
AVAILABLE_HF_MODELS = get_available_models()["huggingface"]

async def summarize_text_with_hf_api(
    text: str, 
    model_name: str = "philschmid/bart-large-cnn-samsum",
    goal: str = "General summarization",
    **kwargs
) -> str:
    """
    Enhanced HuggingFace API integration with improved error handling
    
    This function now uses the unified summarization system for better reliability,
    retry logic, and error handling while maintaining backward compatibility.
    
    Args:
        text: Text to summarize
        model_name: HuggingFace model to use
        goal: Research goal or context for summarization
        **kwargs: Additional configuration parameters
    
    Returns:
        str: Summarized text
    
    Raises:
        Exception: If summarization fails after retries
    """
    try:
        # Create configuration for HuggingFace summarization
        config = SummarizationConfig(
            provider=SummarizationProvider.HUGGINGFACE,
            model=model_name,
            **kwargs
        )
        
        # Use the unified summarization manager
        result = await summarization_manager.summarize(
            content=text,
            goal=goal,
            provider=SummarizationProvider.HUGGINGFACE,
            config=config
        )
        
        logger.debug(f"Successfully summarized text using model '{model_name}'")
        logger.debug(f"Quality score: {result.metadata.quality_score:.2f}")
        logger.debug(f"Processing time: {result.metadata.processing_time:.2f}s")
        
        return result.content
        
    except Exception as e:
        logger.error(f"❌ HuggingFace API error: {e}")
        raise

async def batch_summarize_texts(
    texts: list[str],
    model_name: str = "philschmid/bart-large-cnn-samsum",
    goal: str = "General summarization",
    progress_callback: Optional[callable] = None,
    **kwargs
) -> list[Dict[str, Any]]:
    """
    Batch summarize multiple texts with the unified system
    
    Args:
        texts: List of texts to summarize
        model_name: HuggingFace model to use
        goal: Research goal or context for summarization
        progress_callback: Optional callback for progress updates
        **kwargs: Additional configuration parameters
    
    Returns:
        List of dictionaries with 'text', 'summary', 'metadata' keys
    """
    results = []
    
    config = SummarizationConfig(
        provider=SummarizationProvider.HUGGINGFACE,
        model=model_name,
        progress_callback=progress_callback,
        **kwargs
    )
    
    for i, text in enumerate(texts):
        if progress_callback:
            progress_callback(i + 1, len(texts), f"Summarizing text {i + 1}/{len(texts)}")
        
        try:
            result = await summarization_manager.summarize(
                content=text,
                goal=goal,
                provider=SummarizationProvider.HUGGINGFACE,
                config=config
            )
            
            results.append({
                'text': text,
                'summary': result.content,
                'metadata': {
                    'quality_score': result.metadata.quality_score,
                    'processing_time': result.metadata.processing_time,
                    'tokens_used': result.metadata.tokens_used,
                    'cost_estimate': result.metadata.cost_estimate,
                }
            })
            
        except Exception as e:
            logger.error(f"Failed to summarize text {i + 1}: {e}")
            results.append({
                'text': text,
                'summary': f"Error: {str(e)}",
                'metadata': {'error': str(e)}
            })
    
    return results

def get_hf_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a HuggingFace model
    
    Args:
        model_name: Name of the model
    
    Returns:
        Dictionary with model information
    """
    model_info = {
        "name": model_name,
        "available": model_name in AVAILABLE_HF_MODELS,
        "type": "summarization",
        "provider": "huggingface"
    }
    
    # Add specific model information
    if "bart" in model_name.lower():
        model_info["architecture"] = "BART"
        model_info["description"] = "Bidirectional and Auto-Regressive Transformer for text summarization"
    elif "pegasus" in model_name.lower():
        model_info["architecture"] = "Pegasus"
        model_info["description"] = "Pre-training with Extracted Gap-sentences for Abstractive Summarization"
    elif "dialogpt" in model_name.lower():
        model_info["architecture"] = "DialoGPT"
        model_info["description"] = "Large-scale pretrained dialogue response generation model"
    else:
        model_info["architecture"] = "Unknown"
        model_info["description"] = "Custom or unknown model architecture"
    
    return model_info

async def test_hf_model(model_name: str, test_text: str = None) -> Dict[str, Any]:
    """
    Test a HuggingFace model with sample text
    
    Args:
        model_name: Name of the model to test
        test_text: Optional test text (uses default if not provided)
    
    Returns:
        Dictionary with test results
    """
    if test_text is None:
        test_text = (
            "Artificial intelligence (AI) is intelligence demonstrated by machines, "
            "in contrast to the natural intelligence displayed by humans and animals. "
            "Leading AI textbooks define the field as the study of 'intelligent agents': "
            "any device that perceives its environment and takes actions that maximize "
            "its chance of successfully achieving its goals."
        )
    
    try:
        start_time = __import__('time').time()
        
        result = await summarize_text_with_hf_api(
            text=test_text,
            model_name=model_name,
            goal="Testing model performance"
        )
        
        end_time = __import__('time').time()
        
        return {
            "success": True,
            "model": model_name,
            "input_length": len(test_text),
            "output_length": len(result),
            "summary": result,
            "processing_time": round(end_time - start_time, 2),
            "compression_ratio": round(len(result) / len(test_text), 2)
        }
        
    except Exception as e:
        return {
            "success": False,
            "model": model_name,
            "error": str(e)
        }

# Legacy compatibility - maintain the original function signature
async def summarize_text_with_hf_api_legacy(text: str, model_name: str = "philschmid/bart-large-cnn-samsum"):
    """
    Legacy function for backward compatibility
    
    This maintains the exact same signature and behavior as the original function
    but uses the new unified system under the hood.
    """
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    payload = {"inputs": text}
    url = f"https://api-inference.huggingface.co/models/{model_name}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            logging.debug(f"sending to HF model '{model_name}': {text[:300]}")
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, list) and result and "summary_text" in result[0]:
                return result[0]["summary_text"]
            else:
                raise ValueError(f"Unexpected response format: {result}")
                
    except Exception as e:
        logging.error(f"❌ HuggingFace API error: {e}")
        logging.debug(f"📦 Full HF response: {response.text if 'response' in locals() else 'No response'}")
        raise

# Utility functions
def list_available_models() -> list[str]:
    """Get list of available HuggingFace models"""
    return AVAILABLE_HF_MODELS.copy()

def is_model_available(model_name: str) -> bool:
    """Check if a model is in the available models list"""
    return model_name in AVAILABLE_HF_MODELS

def get_recommended_model(task_type: str = "general") -> str:
    """Get recommended model for different task types"""
    recommendations = {
        "general": "philschmid/bart-large-cnn-samsum",
        "news": "facebook/bart-large-cnn",
        "dialogue": "microsoft/DialoGPT-medium",
        "academic": "google/pegasus-xsum",
        "fast": "sshleifer/distilbart-cnn-12-6"
    }
    
    return recommendations.get(task_type.lower(), recommendations["general"])

