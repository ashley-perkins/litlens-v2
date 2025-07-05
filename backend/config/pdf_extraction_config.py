"""
PDF Extraction Configuration Module

This module provides configuration classes and defaults for PDF text extraction.
It allows users to customize extraction behavior without modifying the main code.

Usage:
    from backend.config.pdf_extraction_config import get_default_config, ProductionConfig
    
    # Use default configuration
    config = get_default_config()
    
    # Use production configuration
    config = ProductionConfig()
    
    # Customize configuration
    config.min_content_length = 200
    config.primary_engine = ExtractionEngine.PYMUPDF
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class ExtractionEngine(Enum):
    """Supported PDF extraction engines."""
    PDFMINER = "pdfminer"
    PYMUPDF = "pymupdf"
    AUTO = "auto"


@dataclass
class PDFExtractionConfig:
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
class DevelopmentConfig(PDFExtractionConfig):
    """Development configuration with verbose logging and relaxed limits."""
    
    min_content_length: int = 50
    max_file_size_mb: int = 50
    enable_progress: bool = True
    log_level: str = "DEBUG"
    max_retries: int = 3
    skip_encrypted: bool = False
    skip_corrupted: bool = False


@dataclass
class ProductionConfig(PDFExtractionConfig):
    """Production configuration with optimized settings."""
    
    min_content_length: int = 200
    max_file_size_mb: int = 200
    enable_progress: bool = False
    log_level: str = "WARNING"
    max_retries: int = 1
    skip_encrypted: bool = True
    skip_corrupted: bool = True
    memory_limit_mb: int = 1024
    timeout_seconds: int = 120


@dataclass
class FastConfig(PDFExtractionConfig):
    """Fast configuration for quick processing."""
    
    primary_engine: ExtractionEngine = ExtractionEngine.PYMUPDF
    fallback_engine: ExtractionEngine = ExtractionEngine.PDFMINER
    clean_text: bool = False
    extract_metadata: bool = False
    extract_sections: bool = False
    max_retries: int = 1
    enable_progress: bool = False
    log_level: str = "ERROR"


@dataclass
class HighQualityConfig(PDFExtractionConfig):
    """High quality configuration for thorough extraction."""
    
    primary_engine: ExtractionEngine = ExtractionEngine.PDFMINER
    fallback_engine: ExtractionEngine = ExtractionEngine.PYMUPDF
    clean_text: bool = True
    normalize_whitespace: bool = True
    remove_headers_footers: bool = True
    extract_metadata: bool = True
    extract_sections: bool = True
    max_retries: int = 3
    skip_encrypted: bool = False
    skip_corrupted: bool = False
    log_level: str = "INFO"


def get_default_config() -> PDFExtractionConfig:
    """Get default PDF extraction configuration."""
    return PDFExtractionConfig()


def get_config_by_name(config_name: str) -> PDFExtractionConfig:
    """
    Get configuration by name.
    
    Args:
        config_name: Name of the configuration (default, development, production, fast, high_quality)
        
    Returns:
        PDFExtractionConfig instance
        
    Raises:
        ValueError: If config_name is not recognized
    """
    config_map = {
        "default": PDFExtractionConfig,
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "fast": FastConfig,
        "high_quality": HighQualityConfig
    }
    
    config_class = config_map.get(config_name.lower())
    if not config_class:
        available_configs = ", ".join(config_map.keys())
        raise ValueError(f"Unknown config name: {config_name}. Available: {available_configs}")
    
    return config_class()


# Environment-based configuration loading
def load_config_from_env() -> PDFExtractionConfig:
    """
    Load configuration from environment variables.
    
    Environment variables:
        PDF_EXTRACTION_CONFIG: Config name (default, development, production, fast, high_quality)
        PDF_MIN_CONTENT_LENGTH: Minimum content length
        PDF_MAX_FILE_SIZE_MB: Maximum file size in MB
        PDF_ENABLE_PROGRESS: Enable progress logging (true/false)
        PDF_LOG_LEVEL: Log level (DEBUG, INFO, WARNING, ERROR)
        PDF_PRIMARY_ENGINE: Primary extraction engine (pdfminer, pymupdf, auto)
        PDF_SKIP_ENCRYPTED: Skip encrypted PDFs (true/false)
        PDF_SKIP_CORRUPTED: Skip corrupted PDFs (true/false)
        PDF_MAX_RETRIES: Maximum number of retry attempts
        PDF_CLEAN_TEXT: Enable text cleaning (true/false)
        PDF_EXTRACT_METADATA: Enable metadata extraction (true/false)
    
    Returns:
        PDFExtractionConfig instance
    """
    import os
    
    # Get base configuration
    config_name = os.getenv("PDF_EXTRACTION_CONFIG", "default")
    config = get_config_by_name(config_name)
    
    # Override with environment variables
    if os.getenv("PDF_MIN_CONTENT_LENGTH"):
        config.min_content_length = int(os.getenv("PDF_MIN_CONTENT_LENGTH"))
    
    if os.getenv("PDF_MAX_FILE_SIZE_MB"):
        config.max_file_size_mb = int(os.getenv("PDF_MAX_FILE_SIZE_MB"))
    
    if os.getenv("PDF_ENABLE_PROGRESS"):
        config.enable_progress = os.getenv("PDF_ENABLE_PROGRESS").lower() == "true"
    
    if os.getenv("PDF_LOG_LEVEL"):
        config.log_level = os.getenv("PDF_LOG_LEVEL").upper()
    
    if os.getenv("PDF_PRIMARY_ENGINE"):
        engine_name = os.getenv("PDF_PRIMARY_ENGINE").upper()
        config.primary_engine = ExtractionEngine[engine_name]
    
    if os.getenv("PDF_SKIP_ENCRYPTED"):
        config.skip_encrypted = os.getenv("PDF_SKIP_ENCRYPTED").lower() == "true"
    
    if os.getenv("PDF_SKIP_CORRUPTED"):
        config.skip_corrupted = os.getenv("PDF_SKIP_CORRUPTED").lower() == "true"
    
    if os.getenv("PDF_MAX_RETRIES"):
        config.max_retries = int(os.getenv("PDF_MAX_RETRIES"))
    
    if os.getenv("PDF_CLEAN_TEXT"):
        config.clean_text = os.getenv("PDF_CLEAN_TEXT").lower() == "true"
    
    if os.getenv("PDF_EXTRACT_METADATA"):
        config.extract_metadata = os.getenv("PDF_EXTRACT_METADATA").lower() == "true"
    
    return config