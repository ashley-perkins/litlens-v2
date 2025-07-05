# PDF Extractor Module Improvements

## Overview

The PDF extractor module has been completely refactored to provide robust, high-performance PDF text extraction capabilities with comprehensive error handling, progress tracking, and configuration options.

## Key Improvements

### 1. **Enhanced Error Handling & Robustness**
- **Multiple extraction engines**: Support for both PDFMiner and PyMuPDF with automatic fallback
- **Comprehensive error detection**: Identifies encrypted, corrupted, empty, and malformed PDFs
- **Retry logic**: Configurable retry attempts with fallback engine switching
- **Graceful degradation**: Continues processing other files when individual files fail

### 2. **Comprehensive Logging & Progress Tracking**
- **Structured logging**: Detailed logging with configurable levels (DEBUG, INFO, WARNING, ERROR)
- **Progress callbacks**: Real-time progress tracking for batch operations
- **Processing statistics**: Detailed statistics about extraction results
- **Performance metrics**: Processing time, memory usage, and throughput tracking

### 3. **Performance Optimizations**
- **Memory management**: Periodic garbage collection and memory monitoring
- **Streaming processing**: Page-by-page processing for large PDFs
- **Configurable limits**: File size limits, page limits, and timeout controls
- **Engine selection**: Automatic engine selection based on PDF characteristics

### 4. **Metadata Extraction**
- **PDF metadata**: Title, author, subject, creator, dates, page count
- **Content metadata**: Extracted from text using existing utilities
- **Technical metadata**: Engine used, processing time, file size
- **Section detection**: Identification of document sections and structure

### 5. **Text Cleaning & Preprocessing**
- **Whitespace normalization**: Configurable whitespace cleanup
- **Header/footer removal**: Automatic removal of common artifacts
- **Content validation**: Minimum/maximum content length validation
- **Encoding handling**: Robust handling of different text encodings

### 6. **Type Hints & Documentation**
- **Full type annotations**: Complete type hints for all functions and classes
- **Comprehensive docstrings**: Detailed documentation with examples
- **Data classes**: Structured configuration and result objects
- **API documentation**: Clear interface specifications

### 7. **Edge Case Handling**
- **Encrypted PDFs**: Configurable handling of password-protected files
- **Corrupted files**: Detection and handling of damaged PDFs
- **Empty files**: Identification of files with no extractable content
- **Large files**: Memory-efficient processing of large documents
- **Network files**: Support for various file path formats

### 8. **Configuration Options**
- **Flexible configuration**: Extensive configuration options via dataclasses
- **Preset configurations**: Pre-built configs for common use cases
- **Environment variables**: Configuration via environment variables
- **Runtime customization**: Dynamic configuration changes

## Backward Compatibility

The module maintains 100% backward compatibility with the existing API:

```python
# Old API still works exactly the same
from backend.modules.pdf_extractor import extract_papers, extract_text_from_folder, extract_text_from_pdf

papers = extract_papers("pdf_folder")
papers = extract_text_from_folder("pdf_folder")
paper = extract_text_from_pdf("file.pdf")
```

## New Advanced API

### Basic Usage with Configuration

```python
from backend.modules.pdf_extractor import PDFExtractor, ExtractionConfig, ExtractionEngine

# Create custom configuration
config = ExtractionConfig(
    min_content_length=200,
    primary_engine=ExtractionEngine.PYMUPDF,
    clean_text=True,
    extract_metadata=True,
    max_retries=3
)

# Create extractor
extractor = PDFExtractor(config)

# Extract from folder with detailed results
results = extractor.extract_from_folder("pdf_folder")

# Process results
for result in results:
    if result.status == PDFStatus.SUCCESS:
        print(f"Successfully extracted {result.content_length} characters from {result.filename}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Engine used: {result.engine_used}")
        print(f"Metadata: {result.metadata}")
    else:
        print(f"Failed to extract from {result.filename}: {result.error_message}")
```

### Progress Tracking

```python
from backend.modules.pdf_extractor import batch_extract_with_progress

def progress_callback(current, total):
    print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")

results = batch_extract_with_progress(
    "pdf_folder",
    config=config,
    progress_callback=progress_callback
)
```

### Configuration Presets

```python
from backend.config.pdf_extraction_config import (
    DevelopmentConfig,
    ProductionConfig,
    FastConfig,
    HighQualityConfig
)

# Use preset configurations
dev_config = DevelopmentConfig()      # Verbose logging, relaxed limits
prod_config = ProductionConfig()      # Optimized for production
fast_config = FastConfig()            # Fast processing, minimal features
quality_config = HighQualityConfig()  # Maximum quality extraction
```

## Configuration Options

### ExtractionConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_content_length` | int | 100 | Minimum content length to consider valid |
| `max_content_length` | int | 10,000,000 | Maximum content length (truncate if exceeded) |
| `primary_engine` | ExtractionEngine | AUTO | Primary extraction engine |
| `fallback_engine` | ExtractionEngine | PYMUPDF | Fallback extraction engine |
| `max_pages` | int | None | Maximum pages to process (None = all) |
| `memory_limit_mb` | int | 512 | Memory limit in MB |
| `timeout_seconds` | int | 300 | Processing timeout in seconds |
| `clean_text` | bool | True | Enable text cleaning |
| `normalize_whitespace` | bool | True | Normalize whitespace |
| `remove_headers_footers` | bool | True | Remove headers/footers |
| `extract_metadata` | bool | True | Extract PDF metadata |
| `extract_sections` | bool | True | Extract document sections |
| `skip_encrypted` | bool | False | Skip encrypted PDFs |
| `skip_corrupted` | bool | True | Skip corrupted PDFs |
| `max_retries` | int | 2 | Maximum retry attempts |
| `enable_progress` | bool | True | Enable progress logging |
| `log_level` | str | "INFO" | Logging level |
| `allowed_extensions` | List[str] | ['.pdf'] | Allowed file extensions |
| `max_file_size_mb` | int | 100 | Maximum file size in MB |

### Environment Variables

Set these environment variables to configure the extractor:

```bash
export PDF_EXTRACTION_CONFIG=production
export PDF_MIN_CONTENT_LENGTH=200
export PDF_MAX_FILE_SIZE_MB=50
export PDF_ENABLE_PROGRESS=true
export PDF_LOG_LEVEL=INFO
export PDF_PRIMARY_ENGINE=pymupdf
export PDF_SKIP_ENCRYPTED=true
export PDF_SKIP_CORRUPTED=true
export PDF_MAX_RETRIES=2
export PDF_CLEAN_TEXT=true
export PDF_EXTRACT_METADATA=true
```

Then load the configuration:

```python
from backend.config.pdf_extraction_config import load_config_from_env

config = load_config_from_env()
extractor = PDFExtractor(config)
```

## Status Codes

The new extractor returns detailed status information:

| Status | Description |
|--------|-------------|
| `SUCCESS` | Extraction completed successfully |
| `FAILED` | Extraction failed due to errors |
| `ENCRYPTED` | PDF is password-protected |
| `CORRUPTED` | PDF file is corrupted or malformed |
| `EMPTY` | PDF has no extractable content |
| `TOO_SMALL` | Extracted content is below minimum length |
| `UNSUPPORTED` | File format not supported |

## Performance Improvements

### Speed Optimizations
- **Engine selection**: Automatic selection of fastest engine for each PDF
- **Memory management**: Reduced memory usage through streaming and cleanup
- **Batch processing**: Optimized batch operations with progress tracking
- **Caching**: Efficient caching of extraction results

### Memory Optimizations
- **Streaming extraction**: Process large PDFs page by page
- **Garbage collection**: Periodic cleanup of unused memory
- **Memory limits**: Configurable memory usage limits
- **Resource cleanup**: Proper cleanup of file handles and resources

### Error Recovery
- **Retry logic**: Automatic retry with different engines
- **Fallback processing**: Graceful degradation when primary engine fails
- **Partial extraction**: Extract what's possible when some pages fail
- **Error isolation**: Continue processing when individual files fail

## Migration Guide

### From Old API to New API

#### Old Code:
```python
from backend.modules.pdf_extractor import extract_papers

papers = extract_papers("pdf_folder")
for paper in papers:
    print(f"File: {paper['filename']}")
    print(f"Content: {paper['content'][:100]}...")
```

#### New Code (Backward Compatible):
```python
from backend.modules.pdf_extractor import extract_papers

# This still works exactly the same
papers = extract_papers("pdf_folder")
for paper in papers:
    print(f"File: {paper['filename']}")
    print(f"Content: {paper['content'][:100]}...")
```

#### New Code (Advanced Features):
```python
from backend.modules.pdf_extractor import PDFExtractor, ExtractionConfig

config = ExtractionConfig(extract_metadata=True)
extractor = PDFExtractor(config)

results = extractor.extract_from_folder("pdf_folder")
for result in results:
    if result.status == PDFStatus.SUCCESS:
        print(f"File: {result.filename}")
        print(f"Content: {result.content[:100]}...")
        print(f"Metadata: {result.metadata}")
        print(f"Processing time: {result.processing_time:.2f}s")
    else:
        print(f"Failed: {result.filename} - {result.error_message}")
```

## Testing

Run the example script to test the new functionality:

```bash
python examples/pdf_extraction_examples.py
```

This will demonstrate:
1. Basic usage (backward compatible)
2. Advanced configuration
3. Single file extraction with detailed results
4. Progress tracking
5. Different extraction engines
6. Configuration presets
7. Error handling

## Dependencies

The enhanced extractor uses the existing dependencies:
- `pdfminer.six` (primary extraction engine)
- `PyMuPDF` (fallback extraction engine)
- `pathlib` (file path handling)
- `dataclasses` (configuration objects)
- `typing` (type hints)

No additional dependencies are required.

## Conclusion

The refactored PDF extractor provides a robust, high-performance solution for PDF text extraction with comprehensive error handling, detailed logging, and flexible configuration options. It maintains 100% backward compatibility while adding powerful new features for advanced use cases.

The module is now production-ready and can handle various PDF formats reliably, making it suitable for large-scale document processing pipelines.