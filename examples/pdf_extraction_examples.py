#!/usr/bin/env python3
"""
PDF Extraction Examples

This script demonstrates various ways to use the enhanced PDF extractor module.
It shows both basic usage (backward compatible) and advanced usage with the new features.

Usage:
    python examples/pdf_extraction_examples.py
"""

import sys
import os
from pathlib import Path
import tempfile
from typing import List, Dict, Any

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.modules.pdf_extractor import (
    PDFExtractor, 
    ExtractionConfig, 
    ExtractionEngine, 
    PDFStatus,
    extract_papers,
    extract_text_from_folder,
    extract_text_from_pdf,
    extract_with_config,
    batch_extract_with_progress
)


def example_basic_usage():
    """Example 1: Basic usage (backward compatible)"""
    print("=" * 60)
    print("Example 1: Basic Usage (Backward Compatible)")
    print("=" * 60)
    
    # This works exactly like the old version
    folder_path = "test_pdfs"  # Replace with your PDF folder
    
    try:
        # Method 1: Using the old function
        papers = extract_papers(folder_path)
        print(f"Extracted {len(papers)} papers using extract_papers()")
        
        # Method 2: Using the old function (alternative)
        papers2 = extract_text_from_folder(folder_path)
        print(f"Extracted {len(papers2)} papers using extract_text_from_folder()")
        
        # Method 3: Extract single PDF
        if papers:
            single_pdf_path = os.path.join(folder_path, papers[0]['filename'])
            if os.path.exists(single_pdf_path):
                paper = extract_text_from_pdf(single_pdf_path)
                print(f"Extracted single paper: {paper['filename']}")
        
    except Exception as e:
        print(f"Error in basic usage: {e}")
    
    print()


def example_advanced_configuration():
    """Example 2: Advanced configuration"""
    print("=" * 60)
    print("Example 2: Advanced Configuration")
    print("=" * 60)
    
    # Create custom configuration
    config = ExtractionConfig(
        min_content_length=200,                    # Require at least 200 characters
        primary_engine=ExtractionEngine.PYMUPDF,  # Use PyMuPDF as primary engine
        fallback_engine=ExtractionEngine.PDFMINER, # Fallback to PDFMiner
        clean_text=True,                          # Clean extracted text
        extract_metadata=True,                    # Extract metadata
        max_retries=3,                           # Retry up to 3 times
        log_level="INFO",                        # Set log level
        enable_progress=True,                    # Show progress
        skip_encrypted=False,                    # Don't skip encrypted PDFs
        max_file_size_mb=50                      # Limit file size to 50MB
    )
    
    # Create extractor with custom configuration
    extractor = PDFExtractor(config)
    
    # Extract from a folder
    folder_path = "test_pdfs"
    try:
        results = extractor.extract_from_folder(folder_path)
        
        print(f"Processed {len(results)} files")
        for result in results[:3]:  # Show first 3 results
            print(f"File: {result.filename}")
            print(f"Status: {result.status.value}")
            print(f"Content length: {result.content_length}")
            print(f"Processing time: {result.processing_time:.2f}s")
            print(f"Engine used: {result.engine_used}")
            if result.warnings:
                print(f"Warnings: {result.warnings}")
            print("-" * 40)
        
        # Get extraction statistics
        stats = extractor.get_extraction_stats()
        print(f"Extraction Statistics: {stats}")
        
    except Exception as e:
        print(f"Error in advanced configuration: {e}")
    
    print()


def example_single_file_extraction():
    """Example 3: Single file extraction with detailed results"""
    print("=" * 60)
    print("Example 3: Single File Extraction with Detailed Results")
    print("=" * 60)
    
    config = ExtractionConfig(
        extract_metadata=True,
        clean_text=True,
        log_level="DEBUG"
    )
    
    extractor = PDFExtractor(config)
    
    # Find a PDF file to test
    test_file = None
    for folder in ["test_pdfs", ".", ".."]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith('.pdf'):
                    test_file = os.path.join(folder, file)
                    break
            if test_file:
                break
    
    if test_file:
        result = extractor.extract_single_pdf(test_file)
        
        print(f"File: {result.filename}")
        print(f"Status: {result.status.value}")
        print(f"Content length: {result.content_length}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Engine used: {result.engine_used}")
        print(f"Pages processed: {result.pages_processed}")
        
        if result.metadata:
            print("Metadata:")
            for key, value in result.metadata.items():
                print(f"  {key}: {value}")
        
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        
        if result.error_message:
            print(f"Error: {result.error_message}")
        
        # Show first 200 characters of content
        if result.content:
            print(f"Content preview: {result.content[:200]}...")
        
    else:
        print("No PDF file found for testing")
    
    print()


def example_progress_tracking():
    """Example 4: Progress tracking during batch extraction"""
    print("=" * 60)
    print("Example 4: Progress Tracking During Batch Extraction")
    print("=" * 60)
    
    def progress_callback(current: int, total: int):
        """Progress callback function"""
        percentage = (current / total) * 100
        print(f"Progress: {current}/{total} ({percentage:.1f}%)")
    
    config = ExtractionConfig(
        enable_progress=True,
        log_level="INFO"
    )
    
    folder_path = "test_pdfs"
    try:
        results = batch_extract_with_progress(
            folder_path, 
            config=config, 
            progress_callback=progress_callback
        )
        
        print(f"Completed processing {len(results)} files")
        
        # Summary of results
        successful = sum(1 for r in results if r.status == PDFStatus.SUCCESS)
        failed = sum(1 for r in results if r.status == PDFStatus.FAILED)
        encrypted = sum(1 for r in results if r.status == PDFStatus.ENCRYPTED)
        
        print(f"Results: {successful} successful, {failed} failed, {encrypted} encrypted")
        
    except Exception as e:
        print(f"Error in progress tracking: {e}")
    
    print()


def example_different_engines():
    """Example 5: Testing different extraction engines"""
    print("=" * 60)
    print("Example 5: Testing Different Extraction Engines")
    print("=" * 60)
    
    # Find a test PDF file
    test_file = None
    for folder in ["test_pdfs", ".", ".."]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith('.pdf'):
                    test_file = os.path.join(folder, file)
                    break
            if test_file:
                break
    
    if not test_file:
        print("No PDF file found for testing")
        return
    
    engines = [ExtractionEngine.PDFMINER, ExtractionEngine.PYMUPDF]
    
    for engine in engines:
        print(f"Testing with {engine.value}:")
        
        config = ExtractionConfig(
            primary_engine=engine,
            max_retries=1,
            log_level="WARNING"
        )
        
        result = extract_with_config(test_file, config)
        
        print(f"  Status: {result.status.value}")
        print(f"  Content length: {result.content_length}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        print(f"  Engine used: {result.engine_used}")
        
        if result.error_message:
            print(f"  Error: {result.error_message}")
        
        print()


def example_configuration_presets():
    """Example 6: Using configuration presets"""
    print("=" * 60)
    print("Example 6: Using Configuration Presets")
    print("=" * 60)
    
    try:
        from backend.config.pdf_extraction_config import (
            get_default_config,
            DevelopmentConfig,
            ProductionConfig,
            FastConfig,
            HighQualityConfig,
            get_config_by_name
        )
        
        configs = {
            "Default": get_default_config(),
            "Development": DevelopmentConfig(),
            "Production": ProductionConfig(),
            "Fast": FastConfig(),
            "High Quality": HighQualityConfig()
        }
        
        for name, config in configs.items():
            print(f"{name} Configuration:")
            print(f"  Min content length: {config.min_content_length}")
            print(f"  Primary engine: {config.primary_engine.value}")
            print(f"  Clean text: {config.clean_text}")
            print(f"  Extract metadata: {config.extract_metadata}")
            print(f"  Max retries: {config.max_retries}")
            print(f"  Log level: {config.log_level}")
            print()
        
        # Example of using a preset by name
        fast_config = get_config_by_name("fast")
        print(f"Fast config loaded: {fast_config.primary_engine.value}")
        
    except ImportError:
        print("Configuration presets not available (config file not found)")
    
    print()


def example_error_handling():
    """Example 7: Error handling and edge cases"""
    print("=" * 60)
    print("Example 7: Error Handling and Edge Cases")
    print("=" * 60)
    
    config = ExtractionConfig(
        skip_encrypted=False,
        skip_corrupted=False,
        max_retries=2,
        log_level="INFO"
    )
    
    extractor = PDFExtractor(config)
    
    # Test with non-existent file
    result = extractor.extract_single_pdf("nonexistent.pdf")
    print(f"Non-existent file - Status: {result.status.value}, Error: {result.error_message}")
    
    # Test with invalid file extension
    if os.path.exists("README.md"):
        result = extractor.extract_single_pdf("README.md")
        print(f"Invalid extension - Status: {result.status.value}, Error: {result.error_message}")
    
    # Test with empty folder
    with tempfile.TemporaryDirectory() as temp_dir:
        results = extractor.extract_from_folder(temp_dir)
        print(f"Empty folder - Found {len(results)} files")
    
    print()


def main():
    """Run all examples"""
    print("PDF Extraction Examples")
    print("=" * 60)
    print()
    
    examples = [
        example_basic_usage,
        example_advanced_configuration,
        example_single_file_extraction,
        example_progress_tracking,
        example_different_engines,
        example_configuration_presets,
        example_error_handling
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            print()
    
    print("=" * 60)
    print("All examples completed!")


if __name__ == "__main__":
    main()