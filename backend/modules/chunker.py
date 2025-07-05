# v0.4.0 - Enhanced Chunking with Multiple Strategies and Intelligent Boundaries

import re
import tiktoken
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from backend.config import ChunkerConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Enumeration of available chunking strategies."""
    SECTION_BASED = "section_based"
    PARAGRAPH_BASED = "paragraph_based"
    SENTENCE_BASED = "sentence_based"
    SLIDING_WINDOW = "sliding_window"
    SEMANTIC_BOUNDARY = "semantic_boundary"
    HYBRID = "hybrid"


@dataclass
class ChunkingConfig:
    """Configuration for chunking operations."""
    max_tokens: int = 3000
    overlap_tokens: int = 200
    min_chunk_size: int = 100
    strategy: ChunkingStrategy = ChunkingStrategy.SECTION_BASED
    preserve_metadata: bool = True
    respect_sentence_boundaries: bool = True
    merge_short_chunks: bool = True
    model_name: str = "gpt-4"


@dataclass
class ChunkMetadata:
    """Metadata for individual chunks."""
    chunk_id: int
    title: str
    original_section: str
    token_count: int
    character_count: int
    start_position: int
    end_position: int
    overlap_start: int = 0
    overlap_end: int = 0
    source_boundaries: List[str] = None
    
    def __post_init__(self):
        if self.source_boundaries is None:
            self.source_boundaries = []


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format for backward compatibility."""
        return {
            "title": self.metadata.title,
            "content": self.content,
            "id": self.metadata.chunk_id,
            "metadata": {
                "original_section": self.metadata.original_section,
                "token_count": self.metadata.token_count,
                "character_count": self.metadata.character_count,
                "start_position": self.metadata.start_position,
                "end_position": self.metadata.end_position,
                "overlap_start": self.metadata.overlap_start,
                "overlap_end": self.metadata.overlap_end,
                "source_boundaries": self.metadata.source_boundaries
            }
        }


class BaseChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.encoder = tiktoken.encoding_for_model(config.model_name)
    
    @abstractmethod
    def chunk_text(self, text: str, sections: List[Dict[str, Any]]) -> List[TextChunk]:
        """Abstract method to chunk text based on strategy."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the configured encoder."""
        return len(self.encoder.encode(text))
    
    def find_sentence_boundaries(self, text: str) -> List[int]:
        """Find sentence boundaries in text."""
        sentences = re.split(r'[.!?]+\s+', text)
        boundaries = []
        pos = 0
        for sentence in sentences[:-1]:
            pos += len(sentence) + 1
            boundaries.append(pos)
        return boundaries
    
    def merge_short_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Merge chunks that are too short with adjacent chunks."""
        if not self.config.merge_short_chunks:
            return chunks
        
        merged_chunks = []
        current_chunk = None
        
        for chunk in chunks:
            if chunk.metadata.token_count < self.config.min_chunk_size:
                if current_chunk is None:
                    current_chunk = chunk
                else:
                    # Merge with previous chunk
                    merged_content = current_chunk.content + "\n\n" + chunk.content
                    merged_metadata = ChunkMetadata(
                        chunk_id=current_chunk.metadata.chunk_id,
                        title=current_chunk.metadata.title,
                        original_section=current_chunk.metadata.original_section,
                        token_count=self.count_tokens(merged_content),
                        character_count=len(merged_content),
                        start_position=current_chunk.metadata.start_position,
                        end_position=chunk.metadata.end_position,
                        source_boundaries=current_chunk.metadata.source_boundaries + chunk.metadata.source_boundaries
                    )
                    current_chunk = TextChunk(merged_content, merged_metadata)
            else:
                if current_chunk is not None:
                    merged_chunks.append(current_chunk)
                    current_chunk = None
                merged_chunks.append(chunk)
        
        if current_chunk is not None:
            merged_chunks.append(current_chunk)
        
        return merged_chunks

class SectionBasedChunker(BaseChunkingStrategy):
    """Section-based chunking strategy."""
    
    def chunk_text(self, text: str, sections: List[Dict[str, Any]]) -> List[TextChunk]:
        """Chunk text based on document sections."""
        chunks = []
        current_chunk = ""
        current_tokens = 0
        current_title = ""
        chunk_id = 1
        
        for section in sections:
            section_title = section["title"]
            section_content = section["content"]
            section_text = f"{section_title}\n{section_content}"
            section_tokens = self.count_tokens(section_text)
            
            if section_tokens > self.config.max_tokens:
                # Handle oversized sections by breaking into paragraphs
                paragraphs = section_content.split("\n\n")
                for para in paragraphs:
                    para_text = f"{section_title}\n{para.strip()}"
                    para_tokens = self.count_tokens(para_text)
                    
                    if para_tokens > self.config.max_tokens:
                        logger.warning(f"Skipping oversized paragraph (> {self.config.max_tokens} tokens)")
                        continue
                    
                    if current_tokens + para_tokens <= self.config.max_tokens:
                        current_chunk += "\n\n" + para_text
                        current_tokens += para_tokens
                    else:
                        if current_chunk:
                            chunks.append(self._create_chunk(current_chunk.strip(), current_title, chunk_id))
                            chunk_id += 1
                        current_chunk = para_text
                        current_title = section_title
                        current_tokens = para_tokens
                
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk.strip(), current_title, chunk_id))
                    chunk_id += 1
                    current_chunk = ""
                    current_tokens = 0
            else:
                if current_tokens + section_tokens <= self.config.max_tokens and section_title == current_title:
                    current_chunk += "\n\n" + section_text
                    current_tokens += section_tokens
                else:
                    if current_chunk:
                        chunks.append(self._create_chunk(current_chunk.strip(), current_title, chunk_id))
                        chunk_id += 1
                    current_chunk = section_text
                    current_title = section_title
                    current_tokens = section_tokens
        
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk.strip(), current_title, chunk_id))
        
        return self.merge_short_chunks(chunks)
    
    def _create_chunk(self, content: str, title: str, chunk_id: int) -> TextChunk:
        """Create a TextChunk with metadata."""
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            title=title,
            original_section=title,
            token_count=self.count_tokens(content),
            character_count=len(content),
            start_position=0,  # Would need full text to calculate
            end_position=len(content),
            source_boundaries=[title]
        )
        return TextChunk(content, metadata)


class SlidingWindowChunker(BaseChunkingStrategy):
    """Sliding window chunking strategy with configurable overlap."""
    
    def chunk_text(self, text: str, sections: List[Dict[str, Any]]) -> List[TextChunk]:
        """Chunk text using sliding window approach."""
        chunks = []
        chunk_id = 1
        
        # Combine all sections into one text
        full_text = "\n\n".join([f"{s['title']}\n{s['content']}" for s in sections])
        
        # Find sentence boundaries if configured
        if self.config.respect_sentence_boundaries:
            boundaries = self.find_sentence_boundaries(full_text)
        else:
            boundaries = list(range(0, len(full_text), self.config.max_tokens // 4))
        
        start_pos = 0
        while start_pos < len(full_text):
            # Find end position
            end_pos = start_pos + self.config.max_tokens * 4  # Approximate character count
            
            # Adjust to sentence boundary if needed
            if self.config.respect_sentence_boundaries and end_pos < len(full_text):
                for boundary in boundaries:
                    if boundary > end_pos:
                        end_pos = boundary
                        break
            
            chunk_text = full_text[start_pos:end_pos]
            
            # Skip if chunk is too small
            if self.count_tokens(chunk_text) < self.config.min_chunk_size:
                break
            
            # Trim to token limit
            while self.count_tokens(chunk_text) > self.config.max_tokens:
                chunk_text = chunk_text[:-100]  # Remove characters from end
            
            if chunk_text.strip():
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    title=f"Chunk {chunk_id}",
                    original_section="Multiple sections",
                    token_count=self.count_tokens(chunk_text),
                    character_count=len(chunk_text),
                    start_position=start_pos,
                    end_position=start_pos + len(chunk_text)
                )
                chunks.append(TextChunk(chunk_text.strip(), metadata))
                chunk_id += 1
            
            # Move start position with overlap
            start_pos = end_pos - self.config.overlap_tokens * 4
        
        return chunks


class DocumentChunker:
    """Main document chunker with multiple strategies."""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.strategy_map = {
            ChunkingStrategy.SECTION_BASED: SectionBasedChunker,
            ChunkingStrategy.SLIDING_WINDOW: SlidingWindowChunker,
            # Add more strategies as needed
        }
    
    def detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """Detect sections in the document text."""
        pattern_str = '|'.join([re.escape(title) for title in ChunkerConfig.SECTION_TITLES])
        section_pattern = re.compile(rf'^((?:\d+\.\s+)?(?:{pattern_str}))', re.IGNORECASE | re.MULTILINE)
        
        splits = section_pattern.split(text)
        logger.info(f"Detected {len(splits)//2} sections after splitting")
        
        sections = []
        for i in range(1, len(splits), 2):
            title = splits[i].strip() if i < len(splits) and splits[i] else f"Section {i//2 + 1}"
            content = splits[i+1] if i+1 < len(splits) else ""
            sections.append({
                "id": i//2 + 1,
                "title": title,
                "content": content.strip()
            })
        
        return sections
    
    def chunk_document(self, text: str) -> List[Dict[str, Any]]:
        """Chunk document using configured strategy."""
        # Detect sections
        sections = self.detect_sections(text)
        
        # Get appropriate chunking strategy
        strategy_class = self.strategy_map.get(self.config.strategy)
        if not strategy_class:
            raise ValueError(f"Unsupported chunking strategy: {self.config.strategy}")
        
        strategy = strategy_class(self.config)
        
        # Chunk the text
        chunks = strategy.chunk_text(text, sections)
        
        # Validate chunks
        self._validate_chunks(chunks)
        
        # Convert to backward-compatible format
        result = [chunk.to_dict() for chunk in chunks]
        
        logger.info(f"Total Chunks Created: {len(result)}")
        return result
    
    def _validate_chunks(self, chunks: List[TextChunk]) -> None:
        """Validate chunks for token limits and quality."""
        if len(chunks) == 1:
            logger.warning("Only 1 chunk created — check chunker settings or file length.")
        
        for chunk in chunks:
            if chunk.metadata.token_count > self.config.max_tokens:
                logger.warning(f"Chunk {chunk.metadata.chunk_id} exceeds token limit: {chunk.metadata.token_count} tokens")


# Backward compatibility function
def chunk_text(text: str, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
    """Backward compatible chunking function."""
    if max_tokens is None:
        max_tokens = 3000
    
    config = ChunkingConfig(max_tokens=max_tokens)
    chunker = DocumentChunker(config)
    return chunker.chunk_document(text)