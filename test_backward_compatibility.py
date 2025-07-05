#!/usr/bin/env python3
"""
Test script to verify backward compatibility of the refactored PDF extractor.
This script tests that the old API still works exactly as expected.
"""

import os
import sys
import tempfile
import traceback

# Add current directory to path
sys.path.insert(0, '.')

def test_import_compatibility():
    """Test that all old imports still work"""
    print("Testing import compatibility...")
    
    try:
        # Test importing the old functions
        from backend.modules.pdf_extractor import (
            extract_papers,
            extract_text_from_folder,
            extract_text_from_pdf
        )
        print("✓ All backward compatibility imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_function_signatures():
    """Test that function signatures are still compatible"""
    print("\nTesting function signatures...")
    
    try:
        from backend.modules.pdf_extractor import (
            extract_papers,
            extract_text_from_folder,
            extract_text_from_pdf
        )
        
        # Test that functions can be called with the old signature
        # We'll use a non-existent folder to avoid actual processing
        test_folder = "non_existent_folder"
        test_file = "non_existent_file.pdf"
        
        # These should not raise signature errors (they might raise other errors)
        try:
            extract_papers(test_folder)
        except Exception as e:
            if "takes" in str(e) and "positional argument" in str(e):
                print(f"✗ extract_papers signature changed: {e}")
                return False
        
        try:
            extract_text_from_folder(test_folder)
        except Exception as e:
            if "takes" in str(e) and "positional argument" in str(e):
                print(f"✗ extract_text_from_folder signature changed: {e}")
                return False
        
        try:
            extract_text_from_pdf(test_file)
        except Exception as e:
            if "takes" in str(e) and "positional argument" in str(e):
                print(f"✗ extract_text_from_pdf signature changed: {e}")
                return False
        
        print("✓ All function signatures are compatible")
        return True
        
    except Exception as e:
        print(f"✗ Function signature test failed: {e}")
        traceback.print_exc()
        return False

def test_return_format():
    """Test that return formats are still compatible"""
    print("\nTesting return format compatibility...")
    
    try:
        from backend.modules.pdf_extractor import (
            extract_papers,
            extract_text_from_folder,
            extract_text_from_pdf
        )
        
        # Create a temporary empty folder
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test folder functions with empty folder
            papers1 = extract_papers(temp_dir)
            papers2 = extract_text_from_folder(temp_dir)
            
            # Should return empty list
            if not isinstance(papers1, list):
                print(f"✗ extract_papers should return list, got {type(papers1)}")
                return False
            
            if not isinstance(papers2, list):
                print(f"✗ extract_text_from_folder should return list, got {type(papers2)}")
                return False
            
            # Test single file function with non-existent file
            result = extract_text_from_pdf("non_existent_file.pdf")
            
            if not isinstance(result, dict):
                print(f"✗ extract_text_from_pdf should return dict, got {type(result)}")
                return False
            
            if "filename" not in result or "content" not in result:
                print(f"✗ extract_text_from_pdf should return dict with 'filename' and 'content' keys")
                return False
        
        print("✓ All return formats are compatible")
        return True
        
    except Exception as e:
        print(f"✗ Return format test failed: {e}")
        traceback.print_exc()
        return False

def test_new_features_available():
    """Test that new features are available"""
    print("\nTesting new features availability...")
    
    try:
        from backend.modules.pdf_extractor import (
            PDFExtractor,
            ExtractionConfig,
            ExtractionEngine,
            PDFStatus,
            ExtractionResult
        )
        
        # Test that we can create the new classes
        config = ExtractionConfig()
        extractor = PDFExtractor(config)
        
        # Test enums
        engines = list(ExtractionEngine)
        statuses = list(PDFStatus)
        
        print(f"✓ New features available: {len(engines)} engines, {len(statuses)} status codes")
        return True
        
    except Exception as e:
        print(f"✗ New features test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all compatibility tests"""
    print("PDF Extractor Backward Compatibility Tests")
    print("=" * 50)
    
    tests = [
        test_import_compatibility,
        test_function_signatures,
        test_return_format,
        test_new_features_available
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All backward compatibility tests passed!")
        return True
    else:
        print("✗ Some backward compatibility tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)