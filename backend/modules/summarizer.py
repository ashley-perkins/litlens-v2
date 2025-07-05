"""
LitLens Unified Summarization Module

This module provides a unified interface for text summarization using multiple AI providers
including OpenAI GPT-4 and HuggingFace models. It includes robust error handling, 
circuit breakers, caching, and quality validation.

Version: 2.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta
from functools import wraps

from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
import httpx

from backend.modules import chunker
from backend.utils.pdf_utils import extract_title_from_text, extract_pdf_metadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client globally for backward compatibility
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    logger.warning(f"Could not initialize OpenAI client: {e}")
    client = None


class SummarizationProvider(Enum):
    """Supported summarization providers"""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class SummarizationStrategy(Enum):
    """Available summarization strategies"""
    SECTION_BY_SECTION = "section_by_section"
    HIERARCHICAL = "hierarchical"
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"


class QualityMetric(Enum):
    """Quality metrics for summary validation"""
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    COMPLETENESS = "completeness"
    FACTUAL_ACCURACY = "factual_accuracy"


@dataclass
class SummarizationConfig:
    """Configuration for summarization operations"""
    provider: SummarizationProvider = SummarizationProvider.OPENAI
    strategy: SummarizationStrategy = SummarizationStrategy.SECTION_BY_SECTION
    model: str = "gpt-4"
    max_tokens: int = 8192
    buffer_tokens: int = 500
    temperature: float = 0.7
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_factor: float = 2.0
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour
    quality_threshold: float = 0.7
    cost_tracking_enabled: bool = True
    progress_callback: Optional[Callable] = None


@dataclass
class SummaryMetadata:
    """Metadata about a summary"""
    provider: str
    model: str
    strategy: str
    tokens_used: int
    cost_estimate: float
    processing_time: float
    quality_score: float
    timestamp: datetime
    chunk_count: int
    original_length: int
    summary_length: int


@dataclass
class SummaryResult:
    """Result of a summarization operation"""
    content: str
    metadata: SummaryMetadata
    quality_metrics: Dict[str, float]
    chunks_processed: List[Dict[str, Any]]
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker"""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN


class SummarizationCache:
    """Simple in-memory cache for summaries"""
    
    def __init__(self, ttl: int = 3600):
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.ttl = ttl
    
    def _generate_key(self, content: str, config: SummarizationConfig, goal: str) -> str:
        """Generate cache key from content and config"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        config_hash = hashlib.md5(
            f"{config.provider.value}:{config.model}:{config.strategy.value}:{goal}".encode()
        ).hexdigest()
        return f"{content_hash}_{config_hash}"
    
    def get(self, content: str, config: SummarizationConfig, goal: str) -> Optional[Any]:
        """Get cached result if available and not expired"""
        key = self._generate_key(content, config, goal)
        if key in self.cache:
            result, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                return result
            else:
                del self.cache[key]
        return None
    
    def set(self, content: str, config: SummarizationConfig, goal: str, result: Any) -> None:
        """Store result in cache"""
        key = self._generate_key(content, config, goal)
        self.cache[key] = (result, datetime.now())
    
    def clear(self) -> None:
        """Clear all cached entries"""
        self.cache.clear()


def circuit_breaker(func):
    """Decorator to implement circuit breaker pattern"""
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_circuit_breaker_state'):
            self._circuit_breaker_state = CircuitBreakerState()
        
        state = self._circuit_breaker_state
        now = datetime.now()
        
        # Check if circuit breaker should be opened
        if state.state == "OPEN":
            if (state.last_failure_time and 
                now - state.last_failure_time > timedelta(seconds=self.config.circuit_breaker_recovery_timeout)):
                state.state = "HALF_OPEN"
                logger.info("Circuit breaker moved to HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - too many failures")
        
        try:
            result = func(self, *args, **kwargs)
            
            # Reset circuit breaker on success
            if state.state == "HALF_OPEN":
                state.state = "CLOSED"
                state.failure_count = 0
                logger.info("Circuit breaker reset to CLOSED state")
            
            return result
            
        except Exception as e:
            state.failure_count += 1
            state.last_failure_time = now
            
            if state.failure_count >= self.config.circuit_breaker_failure_threshold:
                state.state = "OPEN"
                logger.warning(f"Circuit breaker opened after {state.failure_count} failures")
            
            raise e
    
    return wrapper


def count_message_tokens(messages: List[Dict[str, str]], model: str = "gpt-4") -> int:
    """Count tokens in a list of messages for a specific model"""
    try:
        enc = tiktoken.encoding_for_model(model)
        tokens_per_message = 4  # every message structure adds overhead
        tokens = 0
        for msg in messages:
            tokens += tokens_per_message
            for key, value in msg.items():
                tokens += len(enc.encode(str(value)))
        tokens += 2  # reply primer
        return tokens
    except Exception as e:
        logger.warning(f"Error counting tokens for model {model}: {e}")
        # Fallback to approximate count
        total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
        return int(total_chars / 4)  # Rough approximation


class BaseSummarizer(ABC):
    """Abstract base class for all summarizers"""
    
    def __init__(self, config: SummarizationConfig):
        self.config = config
        self.cache = SummarizationCache(config.cache_ttl) if config.cache_enabled else None
        self._circuit_breaker_state = CircuitBreakerState()
        self._usage_stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'average_quality': 0.0
        }
    
    @abstractmethod
    async def _summarize_chunk(self, content: str, context: Dict[str, Any]) -> str:
        """Summarize a single chunk of content"""
        pass
    
    @abstractmethod
    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost for token usage"""
        pass
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate configuration for this provider"""
        pass
    
    def _calculate_quality_score(self, original: str, summary: str, context: Dict[str, Any]) -> float:
        """Calculate quality score for a summary"""
        # Simple heuristic-based quality scoring
        if not summary or not original:
            return 0.0
        
        # Length ratio (summary should be significantly shorter)
        length_ratio = len(summary) / len(original)
        length_score = 1.0 if 0.1 <= length_ratio <= 0.5 else max(0.0, 1.0 - abs(length_ratio - 0.3))
        
        # Keyword preservation (check if important terms are preserved)
        original_words = set(original.lower().split())
        summary_words = set(summary.lower().split())
        important_words = {word for word in original_words if len(word) > 5}
        preserved_ratio = len(important_words & summary_words) / max(len(important_words), 1)
        
        # Completeness check (summary should contain key information)
        completeness_score = min(1.0, preserved_ratio * 2)
        
        # Weighted average
        quality_score = (length_score * 0.3 + preserved_ratio * 0.4 + completeness_score * 0.3)
        
        return min(1.0, max(0.0, quality_score))
    
    def _update_usage_stats(self, tokens: int, cost: float, quality: float) -> None:
        """Update usage statistics"""
        self._usage_stats['total_requests'] += 1
        self._usage_stats['total_tokens'] += tokens
        self._usage_stats['total_cost'] += cost
        
        # Update rolling average quality
        total_requests = self._usage_stats['total_requests']
        current_avg = self._usage_stats['average_quality']
        self._usage_stats['average_quality'] = (
            (current_avg * (total_requests - 1) + quality) / total_requests
        )
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return self._usage_stats.copy()
    
    @circuit_breaker
    async def summarize(self, content: str, goal: str, paper_metadata: Optional[Dict[str, Any]] = None) -> SummaryResult:
        """Main summarization method"""
        start_time = time.time()
        
        # Validate configuration
        self._validate_config()
        
        # Check cache first
        if self.cache:
            cached_result = self.cache.get(content, self.config, goal)
            if cached_result:
                logger.info("Returning cached summary")
                return cached_result
        
        # Initialize context
        context = {
            'goal': goal,
            'metadata': paper_metadata or {},
            'strategy': self.config.strategy.value,
            'model': self.config.model
        }
        
        # Process content based on strategy
        if self.config.strategy == SummarizationStrategy.SECTION_BY_SECTION:
            result = await self._summarize_section_by_section(content, context)
        else:
            # Default to section-by-section for now
            result = await self._summarize_section_by_section(content, context)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        quality_score = self._calculate_quality_score(content, result.content, context)
        
        # Update metadata
        result.metadata.processing_time = processing_time
        result.metadata.quality_score = quality_score
        result.metadata.timestamp = datetime.now()
        
        # Update usage statistics
        if self.config.cost_tracking_enabled:
            self._update_usage_stats(
                result.metadata.tokens_used,
                result.metadata.cost_estimate,
                quality_score
            )
        
        # Validate quality
        if quality_score < self.config.quality_threshold:
            result.warnings.append(f"Quality score {quality_score:.2f} below threshold {self.config.quality_threshold}")
        
        # Cache result
        if self.cache:
            self.cache.set(content, self.config, goal, result)
        
        return result
    
    async def _summarize_section_by_section(self, content: str, context: Dict[str, Any]) -> SummaryResult:
        """Summarize content section by section"""
        # Chunk the content
        chunks = chunker.chunk_text(content, max_tokens=self.config.max_tokens - self.config.buffer_tokens)
        
        if not chunks:
            logger.warning("No chunks generated from content")
            return SummaryResult(
                content="No content to summarize",
                metadata=SummaryMetadata(
                    provider=self.config.provider.value,
                    model=self.config.model,
                    strategy=self.config.strategy.value,
                    tokens_used=0,
                    cost_estimate=0.0,
                    processing_time=0.0,
                    quality_score=0.0,
                    timestamp=datetime.now(),
                    chunk_count=0,
                    original_length=len(content),
                    summary_length=0
                ),
                quality_metrics={},
                chunks_processed=[]
            )
        
        # Process chunks
        summaries = []
        chunks_processed = []
        total_tokens = 0
        
        for i, chunk in enumerate(chunks):
            if self.config.progress_callback:
                self.config.progress_callback(i + 1, len(chunks), f"Processing chunk {i + 1}/{len(chunks)}")
            
            try:
                chunk_summary = await self._summarize_chunk(chunk['content'], context)
                summaries.append(chunk_summary)
                
                # Estimate tokens for this chunk
                chunk_tokens = count_message_tokens([{'content': chunk['content']}], self.config.model)
                total_tokens += chunk_tokens
                
                chunks_processed.append({
                    'title': chunk.get('title', f'Section {i + 1}'),
                    'original_length': len(chunk['content']),
                    'summary_length': len(chunk_summary),
                    'tokens_used': chunk_tokens
                })
                
            except Exception as e:
                logger.error(f"Error processing chunk {i + 1}: {e}")
                chunks_processed.append({
                    'title': chunk.get('title', f'Section {i + 1}'),
                    'error': str(e)
                })
        
        # Combine summaries
        combined_summary = "\n\n".join(summaries)
        
        # Create result
        result = SummaryResult(
            content=combined_summary,
            metadata=SummaryMetadata(
                provider=self.config.provider.value,
                model=self.config.model,
                strategy=self.config.strategy.value,
                tokens_used=total_tokens,
                cost_estimate=self._estimate_cost(total_tokens),
                processing_time=0.0,  # Will be set by caller
                quality_score=0.0,   # Will be calculated by caller
                timestamp=datetime.now(),
                chunk_count=len(chunks),
                original_length=len(content),
                summary_length=len(combined_summary)
            ),
            quality_metrics={},
            chunks_processed=chunks_processed
        )
        
        return result


class OpenAISummarizer(BaseSummarizer):
    """OpenAI GPT-4 based summarizer with enhanced error handling"""
    
    def __init__(self, config: SummarizationConfig):
        super().__init__(config)
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)
    
    def _validate_config(self) -> None:
        """Validate OpenAI-specific configuration"""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        # Validate model
        valid_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
        if self.config.model not in valid_models:
            logger.warning(f"Model {self.config.model} may not be supported. Valid models: {valid_models}")
    
    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost for OpenAI API usage"""
        # Pricing as of 2024 (approximate)
        if self.config.model.startswith("gpt-4"):
            input_cost_per_1k = 0.03
            output_cost_per_1k = 0.06
        else:  # gpt-3.5-turbo
            input_cost_per_1k = 0.001
            output_cost_per_1k = 0.002
        
        # Assume 70% input tokens, 30% output tokens
        input_tokens = int(tokens * 0.7)
        output_tokens = int(tokens * 0.3)
        
        cost = (input_tokens / 1000 * input_cost_per_1k) + (output_tokens / 1000 * output_cost_per_1k)
        return round(cost, 6)
    
    async def _summarize_chunk(self, content: str, context: Dict[str, Any]) -> str:
        """Summarize a single chunk using OpenAI API"""
        metadata = context.get('metadata', {})
        goal = context.get('goal', '')
        
        prompt = self._build_prompt(content, goal, metadata)
        
        messages = [{"role": "user", "content": prompt}]
        
        # Check token limits
        token_count = count_message_tokens(messages, self.config.model)
        allowed_tokens = self.config.max_tokens - self.config.buffer_tokens
        
        if token_count > allowed_tokens:
            logger.warning(f"Trimming content from {token_count} to {allowed_tokens} tokens")
            content = self._trim_content(content, allowed_tokens, prompt, goal, metadata)
            prompt = self._build_prompt(content, goal, metadata)
            messages = [{"role": "user", "content": prompt}]
        
        # Implement retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=min(4096, self.config.max_tokens // 2)  # Reserve tokens for response
                )
                
                summary = response.choices[0].message.content.strip()
                return summary
                
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Handle specific OpenAI errors
                if "rate_limit" in error_str:
                    wait_time = self.config.retry_delay * (self.config.backoff_factor ** attempt)
                    logger.warning(f"Rate limit hit. Waiting {wait_time:.1f}s before retry {attempt + 1}/{self.config.max_retries}")
                    await asyncio.sleep(wait_time)
                    continue
                elif "quota" in error_str or "billing" in error_str:
                    logger.error("OpenAI quota exceeded or billing issue")
                    raise e
                elif "invalid_request" in error_str:
                    logger.error(f"Invalid request to OpenAI API: {e}")
                    raise e
                else:
                    if attempt < self.config.max_retries:
                        wait_time = self.config.retry_delay * (self.config.backoff_factor ** attempt)
                        logger.warning(f"OpenAI API error: {e}. Retrying in {wait_time:.1f}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise e
        
        # If all retries failed
        raise Exception(f"Failed to get summary after {self.config.max_retries} retries. Last error: {last_exception}")
    
    def _build_prompt(self, content: str, goal: str, metadata: Dict[str, Any]) -> str:
        """Build the summarization prompt"""
        prompt = (
            f"You are an expert scientific writer assisting with a literature review.\n\n"
            f"Research Goal: {goal}\n\n"
        )
        
        if metadata:
            prompt += f"Paper Metadata:\n"
            for key, value in metadata.items():
                if value:
                    prompt += f"- {key.title()}: {value}\n"
            prompt += "\n"
        
        prompt += (
            f"Task:\n"
            f"- Summarize the following section of a research paper.\n"
            f"- Focus on relevance to the research goal.\n"
            f"- Use clear, academic bullet points (Markdown format).\n"
            f"- End with a [Reviewer Note] evaluating the section's usefulness.\n"
            f"- Be concise but comprehensive.\n\n"
            f"Content to summarize:\n{content}"
        )
        
        return prompt
    
    def _trim_content(self, content: str, allowed_tokens: int, prompt_template: str, goal: str, metadata: Dict[str, Any]) -> str:
        """Trim content to fit within token limits"""
        try:
            enc = tiktoken.encoding_for_model(self.config.model)
            
            # Calculate tokens for prompt without content
            prompt_base = self._build_prompt("", goal, metadata)
            base_tokens = count_message_tokens([{"role": "user", "content": prompt_base}], self.config.model)
            
            # Calculate available tokens for content
            available_tokens = allowed_tokens - base_tokens
            
            # Trim content to fit
            content_tokens = enc.encode(content)
            if len(content_tokens) > available_tokens:
                trimmed_tokens = content_tokens[:available_tokens]
                trimmed_content = enc.decode(trimmed_tokens)
                logger.info(f"Trimmed content from {len(content_tokens)} to {len(trimmed_tokens)} tokens")
                return trimmed_content
            
            return content
            
        except Exception as e:
            logger.warning(f"Error trimming content: {e}")
            # Fallback to character-based trimming
            try:
                char_ratio = available_tokens / len(content_tokens) if 'content_tokens' in locals() and content_tokens else 0.8
                trimmed_length = int(len(content) * char_ratio)
                return content[:trimmed_length]
            except:
                # Final fallback - return first 80% of content
                return content[:int(len(content) * 0.8)]


class HuggingFaceSummarizer(BaseSummarizer):
    """HuggingFace API based summarizer"""
    
    def __init__(self, config: SummarizationConfig):
        super().__init__(config)
        self.api_token = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize HuggingFace client"""
        self.api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not self.api_token:
            raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables")
    
    def _validate_config(self) -> None:
        """Validate HuggingFace-specific configuration"""
        if not self.api_token:
            raise ValueError("HuggingFace API token not available")
        
        # Default to BART model if not specified
        if not self.config.model:
            self.config.model = "philschmid/bart-large-cnn-samsum"
    
    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost for HuggingFace API usage"""
        # HuggingFace Inference API is often free for moderate usage
        # This is a placeholder for when they implement pricing
        return 0.0
    
    async def _summarize_chunk(self, content: str, context: Dict[str, Any]) -> str:
        """Summarize a single chunk using HuggingFace API"""
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        # Prepare payload with parameters if available
        payload = {"inputs": content}
        if hasattr(self.config, 'temperature') and self.config.temperature != 0.7:
            payload["parameters"] = {"temperature": self.config.temperature}
        
        url = f"https://api-inference.huggingface.co/models/{self.config.model}"
        
        # Implement retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    # Handle different response formats
                    if isinstance(result, list) and result and "summary_text" in result[0]:
                        return result[0]["summary_text"]
                    elif isinstance(result, dict) and "summary_text" in result:
                        return result["summary_text"]
                    elif isinstance(result, dict) and "error" in result:
                        error_msg = result["error"]
                        if "loading" in error_msg.lower():
                            raise httpx.HTTPStatusError("Model loading", request=None, response=type('obj', (object,), {'status_code': 503})())
                        raise Exception(f"HuggingFace API error: {error_msg}")
                    else:
                        raise ValueError(f"Unexpected response format: {result}")
                        
            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code == 503:
                    # Model is loading
                    wait_time = self.config.retry_delay * (self.config.backoff_factor ** attempt)
                    logger.warning(f"Model loading. Waiting {wait_time:.1f}s before retry {attempt + 1}/{self.config.max_retries}")
                    await asyncio.sleep(wait_time)
                    continue
                elif e.response.status_code == 429:
                    # Rate limit
                    wait_time = self.config.retry_delay * (self.config.backoff_factor ** attempt)
                    logger.warning(f"Rate limit hit. Waiting {wait_time:.1f}s before retry {attempt + 1}/{self.config.max_retries}")
                    await asyncio.sleep(wait_time)
                    continue
                elif e.response.status_code == 400:
                    # Bad request - likely input too long
                    logger.error(f"Bad request to HuggingFace API: {e.response.text}")
                    raise Exception(f"Invalid input to HuggingFace API: {e.response.text}")
                else:
                    raise e
                    
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    wait_time = self.config.retry_delay * (self.config.backoff_factor ** attempt)
                    logger.warning(f"HuggingFace API error: {e}. Retrying in {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise e
        
        # If all retries failed
        raise Exception(f"Failed to get summary after {self.config.max_retries} retries. Last error: {last_exception}")


class SummarizationManager:
    """Manager class for handling different summarization providers"""
    
    def __init__(self):
        self.summarizers: Dict[SummarizationProvider, BaseSummarizer] = {}
        self.default_config = SummarizationConfig()
    
    def get_summarizer(self, provider: SummarizationProvider, config: Optional[SummarizationConfig] = None) -> BaseSummarizer:
        """Get or create a summarizer for the specified provider"""
        if config is None:
            config = self.default_config
        
        # Create new summarizer (don't cache to allow different configs)
        if provider == SummarizationProvider.OPENAI:
            return OpenAISummarizer(config)
        elif provider == SummarizationProvider.HUGGINGFACE:
            return HuggingFaceSummarizer(config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def summarize(self, content: str, goal: str, 
                       provider: SummarizationProvider = SummarizationProvider.OPENAI,
                       config: Optional[SummarizationConfig] = None,
                       paper_metadata: Optional[Dict[str, Any]] = None) -> SummaryResult:
        """Summarize content using the specified provider"""
        summarizer = self.get_summarizer(provider, config)
        return await summarizer.summarize(content, goal, paper_metadata)
    
    def get_usage_stats(self, provider: SummarizationProvider) -> Dict[str, Any]:
        """Get usage statistics for a provider"""
        if provider in self.summarizers:
            return self.summarizers[provider].get_usage_stats()
        return {}


# Global manager instance
summarization_manager = SummarizationManager()


# Backward compatibility functions
def summarize_papers(relevant_papers, goal):
    """Legacy function for backward compatibility - uses new unified interface"""
    print("🟣 Starting summarization...\n")
    summaries = []
    
    # Use the new unified interface
    config = SummarizationConfig(
        provider=SummarizationProvider.OPENAI,
        model="gpt-4",
        max_tokens=8192,
        buffer_tokens=500
    )
    
    summarizer = OpenAISummarizer(config)
    
    for paper in relevant_papers:
        print(f"🔍 Summarizing: {paper.get('title', 'Untitled')} ({paper['filename']})")
        
        metadata = paper.get("metadata") or extract_pdf_metadata(paper.get('content', ''))
        paper_title = metadata.get("title") or extract_title_from_text(paper['content'])
        paper['title'] = paper_title
        paper['metadata'] = metadata
        
        print(f"Extracted title: {paper_title}")
        
        try:
            # Use async function synchronously for backward compatibility
            import asyncio
            result = asyncio.run(summarizer.summarize(
                content=paper['content'],
                goal=goal,
                paper_metadata=metadata
            ))
            
            print(f"📄 Processed {result.metadata.chunk_count} sections")
            
            summaries.append({
                "filename": paper['filename'],
                "title": paper_title,
                "summary": result.content,
                "chunks": result.chunks_processed,
                "metadata": metadata,
                "goal": goal,
                "quality_score": result.metadata.quality_score,
                "cost_estimate": result.metadata.cost_estimate
            })
            
        except Exception as e:
            logger.error(f"Error summarizing {paper['filename']}: {e}")
            summaries.append({
                "filename": paper['filename'],
                "title": paper_title,
                "summary": f"Error: {str(e)}",
                "chunks": [],
                "metadata": metadata,
                "goal": goal,
                "error": str(e)
            })
    
    print("\n✅ All papers summarized.\n")
    return summaries

def summarize_inline_text(content: str, goal: str) -> str:
    """Legacy function for backward compatibility - uses new unified interface"""
    try:
        # Use the new unified interface
        config = SummarizationConfig(
            provider=SummarizationProvider.OPENAI,
            model="gpt-4",
            max_tokens=8192,
            strategy=SummarizationStrategy.ABSTRACTIVE
        )
        
        summarizer = OpenAISummarizer(config)
        
        # Use async function synchronously for backward compatibility
        import asyncio
        result = asyncio.run(summarizer.summarize(
            content=content,
            goal=goal
        ))
        
        return result.content
        
    except Exception as e:
        logger.error(f"Error in inline summarization: {e}")
        # Fallback to old method if new interface fails
        if client is None:
            raise Exception("OpenAI client not available and summarization failed")
        
        prompt = (
            f"You are an expert scientific writer assisting with a literature review.\n\n"
            f"Research Goal: {goal}\n\n"
            f"Task:\n"
            f"- Summarize the content below in 3–5 bullet points.\n"
            f"- Keep tone academic, format Markdown.\n"
            f"- End with a [Reviewer Note] about how well this aligns with the goal.\n\n"
            f"Content:\n{content}"
        )
        
        enc = tiktoken.encoding_for_model("gpt-4")
        messages = [{"role": "user", "content": prompt}]
        if count_message_tokens(messages) > 7692:
            allowed = 7692 - count_message_tokens([{"role": "user", "content": prompt.replace(content, '')}])
            content_trimmed = enc.decode(enc.encode(content)[:allowed])
            prompt = prompt.replace(content, content_trimmed)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content.strip()

# Convenience functions for external use
async def summarize_with_openai(content: str, goal: str, config: Optional[SummarizationConfig] = None) -> SummaryResult:
    """Convenience function for OpenAI summarization"""
    if config is None:
        config = SummarizationConfig(provider=SummarizationProvider.OPENAI)
    return await summarization_manager.summarize(content, goal, SummarizationProvider.OPENAI, config)

async def summarize_with_huggingface(content: str, goal: str, config: Optional[SummarizationConfig] = None) -> SummaryResult:
    """Convenience function for HuggingFace summarization"""
    if config is None:
        config = SummarizationConfig(provider=SummarizationProvider.HUGGINGFACE)
    return await summarization_manager.summarize(content, goal, SummarizationProvider.HUGGINGFACE, config)

def get_available_models() -> Dict[str, List[str]]:
    """Get list of available models for each provider"""
    return {
        "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        "huggingface": [
            "philschmid/bart-large-cnn-samsum",
            "facebook/bart-large-cnn",
            "microsoft/DialoGPT-medium",
            "google/pegasus-xsum",
            "sshleifer/distilbart-cnn-12-6"
        ]
    }

# Enhanced HuggingFace integration from hf_utils.py
async def summarize_text_with_hf_api(text: str, model_name: str = "philschmid/bart-large-cnn-samsum") -> str:
    """Direct HuggingFace API integration with improved error handling"""
    config = SummarizationConfig(
        provider=SummarizationProvider.HUGGINGFACE,
        model=model_name
    )
    
    summarizer = HuggingFaceSummarizer(config)
    
    try:
        result = await summarizer.summarize(text, "General summarization")
        return result.content
    except Exception as e:
        logger.error(f"HuggingFace summarization failed: {e}")
        raise