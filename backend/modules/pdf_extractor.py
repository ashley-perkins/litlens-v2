"""
Enhanced PDF Extractor Module for LitLens

This module provides robust PDF text extraction capabilities with comprehensive error handling,
metadata extraction, progress tracking, and configuration options. It supports multiple extraction
engines and handles various PDF formats including encrypted and corrupted files.

Author: LitLens Development Team
Version: 2.0.0
"""

import os
import logging
import time
import re
import gc
import tempfile
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

# Configure logging first
logger = logging.getLogger(__name__)

# PDF processing libraries
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available. Some features may be limited.")

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.pdfpage import PDFPage
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfparser import PDFSyntaxError
    from pdfminer.psparser import PSException
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    logger.warning("PDFMiner not available. Some features may be limited.")

# Local imports (after logger is configured)
try:
    from backend.utils.pdf_utils import extract_pdf_metadata
    PDF_UTILS_AVAILABLE = True
except ImportError:
    PDF_UTILS_AVAILABLE = False
    logger.warning("PDF utils not available due to missing dependencies.")
    
    # Provide a fallback for extract_pdf_metadata
    def extract_pdf_metadata(text):
        """Fallback metadata extraction when pdf_utils is not available."""
        import re
        metadata = {}
        
        # Simple year extraction
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            metadata['year'] = year_match.group(0)
        
        # Simple journal extraction
        journal_match = re.search(r'(Journal|Proceedings|Conference)[^\n]{0,100}', text, re.IGNORECASE)
        if journal_match:
            metadata['journal'] = journal_match.group(0).strip()
        
        return metadata

try:
    from backend.config import ChunkerConfig
except ImportError:
    # ChunkerConfig is not used in the current implementation
    # This import was likely from the old code
    ChunkerConfig = None


class ExtractionEngine(Enum):
    """Supported PDF extraction engines."""
    PDFMINER = "pdfminer"
    PYMUPDF = "pymupdf"
    AUTO = "auto"


class PDFStatus(Enum):
    """PDF processing status codes."""
    SUCCESS = "success"
    FAILED = "failed"
    ENCRYPTED = "encrypted"
    CORRUPTED = "corrupted"
    EMPTY = "empty"
    TOO_SMALL = "too_small"
    UNSUPPORTED = "unsupported"


@dataclass
class ExtractionConfig:
    """Configuration for PDF extraction parameters."""
    
    # Content filtering
    min_content_length: int = 100
    max_content_length: int = 10_000_000  # 10MB text limit
    
    # Extraction engines
    primary_engine: ExtractionEngine = ExtractionEngine.AUTO
    fallback_engine: ExtractionEngine = ExtractionEngine.PYMUPDF
    
    # Performance settings
    max_pages: Optional[int] = None
    memory_limit_mb: int = 512
    timeout_seconds: int = 300
    
    # Text cleaning options
    clean_text: bool = True
    normalize_whitespace: bool = True
    remove_headers_footers: bool = True
    
    # Metadata extraction
    extract_metadata: bool = True
    extract_sections: bool = True
    
    # Error handling
    skip_encrypted: bool = False
    skip_corrupted: bool = True
    max_retries: int = 2
    
    # Logging and progress
    enable_progress: bool = True
    log_level: str = "INFO"
    
    # File validation
    allowed_extensions: List[str] = field(default_factory=lambda: ['.pdf'])
    max_file_size_mb: int = 100


@dataclass
class ExtractionResult:
    """Result of PDF text extraction."""
    
    # Basic information
    filename: str
    file_path: str
    status: PDFStatus
    
    # Content
    content: str
    content_length: int
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Processing info
    processing_time: float = 0.0
    engine_used: str = ""
    pages_processed: int = 0
    
    # Error information
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format for backward compatibility."""
        return {
            "filename": self.filename,
            "content": self.content,
            "metadata": self.metadata,
            "status": self.status.value,
            "processing_time": self.processing_time,
            "engine_used": self.engine_used,
            "pages_processed": self.pages_processed,
            "content_length": self.content_length,
            "error_message": self.error_message,
            "warnings": self.warnings
        }


class PDFExtractor:
    """Enhanced PDF text extractor with multiple engine support and robust error handling."""
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        """Initialize PDF extractor with configuration."""
        self.config = config or ExtractionConfig()
        self._setup_logging()
        self._extraction_stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "encrypted": 0,
            "corrupted": 0,
            "empty": 0,
            "total_processing_time": 0.0
        }
    
    def _setup_logging(self):
        """Configure logging for the extractor."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    @contextmanager
    def _memory_monitor(self):
        """Context manager to monitor memory usage."""
        try:
            yield
        finally:
            gc.collect()
    
    def _validate_file(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """Validate PDF file before processing."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"
        
        if file_path.suffix.lower() not in self.config.allowed_extensions:
            return False, f"Unsupported file extension: {file_path.suffix}"
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            return False, f"File too large: {file_size_mb:.1f}MB (max: {self.config.max_file_size_mb}MB)"
        
        return True, ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not self.config.clean_text:
            return text
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove common headers/footers patterns
        if self.config.remove_headers_footers:
            text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'^Page \d+ of \d+\s*$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def _extract_with_pdfminer(self, file_path: Path) -> Tuple[str, int]:
        """Extract text using PDFMiner."""
        if not PDFMINER_AVAILABLE:
            raise ImportError("PDFMiner not available. Install pdfminer.six to use this engine.")
        
        try:
            # Try simple extraction first
            text = pdfminer_extract_text(str(file_path))
            pages_processed = 0
            
            # Count pages if possible
            try:
                with open(file_path, 'rb') as fp:
                    pages_processed = len(list(PDFPage.get_pages(fp)))
            except:
                pass
            
            return text, pages_processed
            
        except (PDFSyntaxError, PSException) as e:
            logger.warning(f"PDFMiner failed for {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"PDFMiner extraction failed for {file_path}: {e}")
            raise
    
    def _extract_with_pymupdf(self, file_path: Path) -> Tuple[str, int]:
        """Extract text using PyMuPDF."""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF not available. Install PyMuPDF to use this engine.")
        
        try:
            text_parts = []
            pages_processed = 0
            
            with fitz.open(str(file_path)) as doc:
                total_pages = len(doc)
                max_pages = self.config.max_pages or total_pages
                
                for page_num in range(min(max_pages, total_pages)):
                    page = doc[page_num]
                    page_text = page.get_text()
                    
                    if page_text.strip():
                        text_parts.append(page_text)
                    
                    pages_processed += 1
                    
                    # Check memory usage periodically
                    if pages_processed % 50 == 0:
                        gc.collect()
            
            return '\n'.join(text_parts), pages_processed
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {file_path}: {e}")
            raise
    
    def _extract_text_with_engine(self, file_path: Path, engine: ExtractionEngine) -> Tuple[str, int]:
        """Extract text using specified engine."""
        if engine == ExtractionEngine.PDFMINER:
            return self._extract_with_pdfminer(file_path)
        elif engine == ExtractionEngine.PYMUPDF:
            return self._extract_with_pymupdf(file_path)
        else:
            raise ValueError(f"Unknown extraction engine: {engine}")
    
    def _detect_pdf_issues(self, file_path: Path) -> Optional[PDFStatus]:
        """Detect common PDF issues before extraction."""
        if not PYMUPDF_AVAILABLE:
            # Skip issue detection if PyMuPDF is not available
            return None
        
        try:
            with fitz.open(str(file_path)) as doc:
                if doc.is_encrypted:
                    return PDFStatus.ENCRYPTED
                
                if len(doc) == 0:
                    return PDFStatus.EMPTY
                
                # Check if first few pages have readable text
                sample_text = ""
                for page_num in range(min(3, len(doc))):
                    page_text = doc[page_num].get_text()
                    sample_text += page_text
                
                if len(sample_text.strip()) < 10:
                    return PDFStatus.EMPTY
                
        except Exception as e:
            logger.warning(f"PDF issue detection failed for {file_path}: {e}")
            return PDFStatus.CORRUPTED
        
        return None
    
    def extract_single_pdf(self, file_path: Union[str, Path]) -> ExtractionResult:
        """Extract text from a single PDF file."""
        file_path = Path(file_path)
        start_time = time.time()
        
        # Initialize result
        result = ExtractionResult(
            filename=file_path.name,
            file_path=str(file_path),
            status=PDFStatus.FAILED,
            content="",
            content_length=0
        )
        
        try:
            # Validate file
            is_valid, error_msg = self._validate_file(file_path)
            if not is_valid:
                result.error_message = error_msg
                result.status = PDFStatus.UNSUPPORTED
                return result
            
            # Detect PDF issues
            pdf_issue = self._detect_pdf_issues(file_path)
            if pdf_issue:
                result.status = pdf_issue
                result.error_message = f"PDF issue detected: {pdf_issue.value}"
                
                # Skip processing based on config
                if pdf_issue == PDFStatus.ENCRYPTED and self.config.skip_encrypted:
                    return result
                if pdf_issue == PDFStatus.CORRUPTED and self.config.skip_corrupted:
                    return result
            
            # Determine extraction engine
            engine = self.config.primary_engine
            if engine == ExtractionEngine.AUTO:
                # Choose the first available engine
                if PDFMINER_AVAILABLE:
                    engine = ExtractionEngine.PDFMINER
                elif PYMUPDF_AVAILABLE:
                    engine = ExtractionEngine.PYMUPDF
                else:
                    result.error_message = "No PDF extraction engines available. Install pdfminer.six or PyMuPDF."
                    result.status = PDFStatus.FAILED
                    return result
            
            # Extract text with retry logic
            text = ""
            pages_processed = 0
            engine_used = ""
            
            for attempt in range(self.config.max_retries + 1):
                try:
                    with self._memory_monitor():
                        text, pages_processed = self._extract_text_with_engine(file_path, engine)
                        engine_used = engine.value
                        break
                        
                except Exception as e:
                    logger.warning(f"Extraction attempt {attempt + 1} failed with {engine.value}: {e}")
                    
                    if attempt < self.config.max_retries:
                        # Try fallback engine if available and different
                        fallback_engine = self.config.fallback_engine
                        if (engine != fallback_engine and 
                            ((fallback_engine == ExtractionEngine.PDFMINER and PDFMINER_AVAILABLE) or
                             (fallback_engine == ExtractionEngine.PYMUPDF and PYMUPDF_AVAILABLE))):
                            engine = fallback_engine
                            continue
                    
                    # All attempts failed
                    result.error_message = f"All extraction attempts failed. Last error: {e}"
                    result.status = PDFStatus.FAILED
                    return result
            
            # Clean text
            if text:
                text = self._clean_text(text)
            
            # Validate content length
            content_length = len(text.strip())
            if content_length < self.config.min_content_length:
                result.status = PDFStatus.TOO_SMALL
                result.error_message = f"Content too small: {content_length} characters"
                result.content = text
                result.content_length = content_length
                return result
            
            if content_length > self.config.max_content_length:
                result.warnings.append(f"Content truncated from {content_length} to {self.config.max_content_length} characters")
                text = text[:self.config.max_content_length]
                content_length = len(text)
            
            # Extract metadata
            metadata = {}
            if self.config.extract_metadata:
                try:
                    metadata = extract_pdf_metadata(text)
                    
                    # Add technical metadata using PyMuPDF if available
                    if PYMUPDF_AVAILABLE:
                        with fitz.open(str(file_path)) as doc:
                            doc_metadata = doc.metadata
                            if doc_metadata:
                                metadata.update({
                                    'title': doc_metadata.get('title', ''),
                                    'author': doc_metadata.get('author', ''),
                                    'subject': doc_metadata.get('subject', ''),
                                    'creator': doc_metadata.get('creator', ''),
                                    'producer': doc_metadata.get('producer', ''),
                                    'creation_date': doc_metadata.get('creationDate', ''),
                                    'modification_date': doc_metadata.get('modDate', ''),
                                    'total_pages': len(doc)
                                })
                except Exception as e:
                    result.warnings.append(f"Metadata extraction failed: {e}")
            
            # Update result
            result.content = text
            result.content_length = content_length
            result.metadata = metadata
            result.status = PDFStatus.SUCCESS
            result.engine_used = engine_used
            result.pages_processed = pages_processed
            
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path}: {e}")
            result.error_message = f"Unexpected error: {e}"
            result.status = PDFStatus.FAILED
        
        finally:
            result.processing_time = time.time() - start_time
            
            # Update statistics
            self._extraction_stats["total_files"] += 1
            self._extraction_stats["total_processing_time"] += result.processing_time
            
            if result.status == PDFStatus.SUCCESS:
                self._extraction_stats["successful"] += 1
            elif result.status == PDFStatus.ENCRYPTED:
                self._extraction_stats["encrypted"] += 1
            elif result.status == PDFStatus.CORRUPTED:
                self._extraction_stats["corrupted"] += 1
            elif result.status in [PDFStatus.EMPTY, PDFStatus.TOO_SMALL]:
                self._extraction_stats["empty"] += 1
            else:
                self._extraction_stats["failed"] += 1
        
        return result
    
    def extract_from_folder(self, folder_path: Union[str, Path], 
                          progress_callback: Optional[Callable[[int, int], None]] = None) -> List[ExtractionResult]:
        """Extract text from all PDF files in a folder."""
        folder_path = Path(folder_path)
        
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Invalid folder path: {folder_path}")
        
        # Find all PDF files
        pdf_files = []
        for ext in self.config.allowed_extensions:
            pdf_files.extend(folder_path.glob(f"**/*{ext}"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {folder_path}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = []
        for i, pdf_file in enumerate(pdf_files):
            if self.config.enable_progress:
                logger.info(f"Processing {i+1}/{len(pdf_files)}: {pdf_file.name}")
            
            result = self.extract_single_pdf(pdf_file)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(pdf_files))
        
        self._log_extraction_summary(results)
        return results
    
    def _log_extraction_summary(self, results: List[ExtractionResult]):
        """Log summary of extraction results."""
        if not results:
            return
        
        total = len(results)
        successful = sum(1 for r in results if r.status == PDFStatus.SUCCESS)
        failed = sum(1 for r in results if r.status == PDFStatus.FAILED)
        encrypted = sum(1 for r in results if r.status == PDFStatus.ENCRYPTED)
        corrupted = sum(1 for r in results if r.status == PDFStatus.CORRUPTED)
        empty = sum(1 for r in results if r.status in [PDFStatus.EMPTY, PDFStatus.TOO_SMALL])
        
        avg_time = sum(r.processing_time for r in results) / total
        total_content = sum(r.content_length for r in results)
        
        logger.info(f"Extraction Summary:")
        logger.info(f"  Total files: {total}")
        logger.info(f"  Successful: {successful} ({successful/total*100:.1f}%)")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Encrypted: {encrypted}")
        logger.info(f"  Corrupted: {corrupted}")
        logger.info(f"  Empty/Too small: {empty}")
        logger.info(f"  Average processing time: {avg_time:.2f}s")
        logger.info(f"  Total content extracted: {total_content:,} characters")
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return self._extraction_stats.copy()


# Module-level functions for backward compatibility
_default_extractor = PDFExtractor()


def extract_papers(pdf_folder: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Extract text from all PDF files in a folder (backward compatible).
    
    Args:
        pdf_folder: Path to folder containing PDF files
        
    Returns:
        List of dictionaries with filename and content
    """
    results = _default_extractor.extract_from_folder(pdf_folder)
    
    # Convert to backward compatible format
    papers = []
    for result in results:
        if result.status == PDFStatus.SUCCESS:
            papers.append({
                "filename": result.filename,
                "content": result.content
            })
        else:
            # Log issues but maintain backward compatibility
            if result.status not in [PDFStatus.TOO_SMALL, PDFStatus.EMPTY]:
                logger.warning(f"Skipping {result.filename}: {result.error_message}")
    
    logger.info(f"Extracted {len(papers)} papers from {len(results)} files")
    return papers


def extract_text_from_folder(folder_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Extract text from all PDF files in a folder (backward compatible).
    
    Args:
        folder_path: Path to folder containing PDF files
        
    Returns:
        List of dictionaries with filename and content
    """
    return extract_papers(folder_path)


def extract_text_from_pdf(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract text from a single PDF file (backward compatible).
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Dictionary with filename and content
    """
    result = _default_extractor.extract_single_pdf(file_path)
    
    return {
        "filename": result.filename,
        "content": result.content if result.status == PDFStatus.SUCCESS else ""
    }


# Advanced functions for new functionality
def extract_with_config(file_path: Union[str, Path], config: ExtractionConfig) -> ExtractionResult:
    """
    Extract text from PDF with custom configuration.
    
    Args:
        file_path: Path to PDF file
        config: Extraction configuration
        
    Returns:
        ExtractionResult with detailed information
    """
    extractor = PDFExtractor(config)
    return extractor.extract_single_pdf(file_path)


def batch_extract_with_progress(folder_path: Union[str, Path], 
                               config: Optional[ExtractionConfig] = None,
                               progress_callback: Optional[Callable[[int, int], None]] = None) -> List[ExtractionResult]:
    """
    Extract text from folder with progress tracking.
    
    Args:
        folder_path: Path to folder containing PDF files
        config: Optional extraction configuration
        progress_callback: Optional callback function for progress updates
        
    Returns:
        List of ExtractionResult objects
    """
    extractor = PDFExtractor(config)
    return extractor.extract_from_folder(folder_path, progress_callback)