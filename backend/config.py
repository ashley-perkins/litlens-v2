# config.py

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY not found in environment")

# Additional API configurations
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "30"))

# Processing settings
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))

# File processing settings
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
ALLOWED_FILE_TYPES = os.getenv("ALLOWED_FILE_TYPES", "pdf,txt,md,docx").split(",")
TEMP_DIR = os.getenv("TEMP_DIR", "temp")

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "litlens.log")
LOG_MAX_SIZE = int(os.getenv("LOG_MAX_SIZE", "10485760"))  # 10MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "3"))

# Cache settings
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "True").lower() == "true"
CACHE_DIR = os.getenv("CACHE_DIR", "cache")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour

# Development settings
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
PROFILE_PERFORMANCE = os.getenv("PROFILE_PERFORMANCE", "False").lower() == "true"

# Chunker Settings
class ChunkerConfig:
    MAX_TOKENS = 1000
    
    # Enhanced chunking parameters
    DEFAULT_MAX_TOKENS = 3000
    DEFAULT_OVERLAP_TOKENS = 200
    MIN_CHUNK_SIZE = 100
    
    # Chunking strategies
    CHUNKING_STRATEGIES = [
        "section_based",
        "paragraph_based", 
        "sentence_based",
        "sliding_window",
        "semantic_boundary",
        "hybrid"
    ]
    
    # Section detection patterns
    SECTION_TITLES = [
        "Abstract",
        "Introduction",
        "Background",
        "Methods",
        "Materials",
        "Results",
        "Discussion",
        "Conclusion",
        "References",
        "Summary",
        "Acknowledgments",
        "Study Design",
        "Patient Cohort",
        "Preliminary Results",
        "Future Work",
        "Limitations",
        "Methodology",
        "Analysis",
        "Findings",
        "Related Work",
        "Literature Review",
        "Experimental Setup",
        "Data Collection",
        "Statistical Analysis",
        "Clinical Implications",
        "Ethics",
        "Funding",
        "Conflicts of Interest"
    ]
    
    # Additional patterns for section detection
    SECTION_PATTERNS = [
        r'^(\d+\.\s*)?({})\s*:?\s*$',  # Numbered sections
        r'^({})\s*:?\s*$',             # Plain sections
        r'^\s*({})\s*:?\s*$',          # Indented sections
    ]
    
    # Performance settings
    RESPECT_SENTENCE_BOUNDARIES = True
    MERGE_SHORT_CHUNKS = True
    PRESERVE_METADATA = True
    
    # Token counting models
    SUPPORTED_MODELS = [
        "gpt-4",
        "gpt-3.5-turbo",
        "text-davinci-003",
        "text-curie-001"
    ]
    
    DEFAULT_MODEL = "gpt-4"


# Report Generator Settings
class ReportGeneratorConfig:
    # Supported output formats
    SUPPORTED_FORMATS = [
        "txt",
        "md", 
        "html",
        "json",
        "pdf"
    ]
    
    DEFAULT_FORMAT = "md"
    
    # Report components
    INCLUDE_TOC = True
    INCLUDE_METADATA = True
    INCLUDE_STATISTICS = True
    INCLUDE_TIMESTAMPS = True
    
    # Content settings
    MAX_SUMMARY_LENGTH = 500
    PAPER_NUMBERING = True
    SECTION_NUMBERING = True
    
    # File settings
    OUTPUT_ENCODING = "utf-8"
    DEFAULT_OUTPUT_PATH = "litlens_summary_report"
    
    # Template settings
    TEMPLATE_DIR = "templates"
    CUSTOM_CSS_PATH = None
    
    # Metadata formatting
    MAX_AUTHORS_DISPLAY = 3
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # HTML specific settings
    HTML_TITLE = "LitLens Summary Report"
    HTML_VIEWPORT = "width=device-width, initial-scale=1.0"
    
    # JSON specific settings
    JSON_INDENT = 2
    JSON_ENSURE_ASCII = False
    
    # PDF specific settings (for future implementation)
    PDF_PAGE_SIZE = "A4"
    PDF_MARGINS = {
        "top": 2.5,
        "bottom": 2.5,
        "left": 2.5,
        "right": 2.5
    }
    
    # Performance settings
    ENABLE_CACHING = True
    CACHE_DURATION = 3600  # 1 hour
    
    # Validation settings
    VALIDATE_SUMMARIES = True
    STRICT_VALIDATION = False
    
    # Error handling
    CONTINUE_ON_ERROR = True
    LOG_ERRORS = True
    
    # Statistics settings
    CALCULATE_WORD_COUNT = True
    CALCULATE_READING_TIME = True
    AVERAGE_READING_SPEED = 200  # words per minute
    
    # Multi-format generation
    PARALLEL_GENERATION = False
    MAX_WORKERS = 4