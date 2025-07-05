"""
API Models for LitLens Backend

This module defines all Pydantic models used for request/response validation
and data serialization in the LitLens API.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import os


class StatusEnum(str, Enum):
    """Status enumeration for API responses."""
    SUCCESS = "success"
    ERROR = "error"
    PROCESSING = "processing"


class SummarizeRequest(BaseModel):
    """Request model for text summarization endpoints."""
    goal: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="Research goal to guide the summarization process"
    )
    content: str = Field(
        ..., 
        min_length=1, 
        max_length=50000,
        description="Text content to be summarized"
    )
    
    @validator('goal')
    def validate_goal(cls, v):
        """Validate research goal is not empty and contains meaningful content."""
        if not v.strip():
            raise ValueError('Research goal cannot be empty')
        return v.strip()
    
    @validator('content')
    def validate_content(cls, v):
        """Validate content is not empty and contains meaningful text."""
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()


class SummarizeResponse(BaseModel):
    """Response model for text summarization endpoints."""
    goal: str = Field(..., description="The research goal that guided summarization")
    summary: str = Field(..., description="The generated summary")
    status: StatusEnum = Field(default=StatusEnum.SUCCESS, description="Processing status")
    word_count: Optional[int] = Field(None, description="Word count of the summary")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


class PaperSummary(BaseModel):
    """Model for individual paper summary."""
    filename: str = Field(..., description="Original filename of the paper")
    title: str = Field(..., description="Title of the paper")
    summary: str = Field(..., description="Generated summary of the paper")
    relevance_score: Optional[float] = Field(None, description="Relevance score (0-1)")
    word_count: Optional[int] = Field(None, description="Word count of the summary")
    extraction_success: bool = Field(True, description="Whether text extraction was successful")
    
    @validator('filename')
    def validate_filename(cls, v):
        """Validate filename is safe and has appropriate extension."""
        if not v.strip():
            raise ValueError('Filename cannot be empty')
        # Check for path traversal attempts
        if '..' in v or '/' in v or '\\' in v:
            raise ValueError('Filename contains invalid characters')
        return v.strip()


class PDFSummarizeResponse(BaseModel):
    """Response model for PDF summarization endpoints."""
    goal: str = Field(..., description="The research goal that guided summarization")
    summaries: List[PaperSummary] = Field(..., description="List of paper summaries")
    output_path: str = Field(..., description="Path to the generated report file")
    status: StatusEnum = Field(default=StatusEnum.SUCCESS, description="Processing status")
    total_papers: int = Field(..., description="Total number of papers processed")
    relevant_papers: int = Field(..., description="Number of relevant papers found")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    @validator('summaries')
    def validate_summaries(cls, v):
        """Validate that summaries list is not empty."""
        if not v:
            raise ValueError('Summaries cannot be empty')
        return v


class ReportRequest(BaseModel):
    """Request model for report generation."""
    goal: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="Research goal for the report"
    )
    summaries: List[Dict[str, Any]] = Field(
        ..., 
        min_items=1,
        description="List of paper summaries to include in the report"
    )
    
    @validator('goal')
    def validate_goal(cls, v):
        """Validate research goal is not empty."""
        if not v.strip():
            raise ValueError('Research goal cannot be empty')
        return v.strip()
    
    @validator('summaries')
    def validate_summaries(cls, v):
        """Validate summaries list is not empty."""
        if not v:
            raise ValueError('Summaries cannot be empty')
        return v


class ErrorResponse(BaseModel):
    """Standard error response model."""
    status: StatusEnum = Field(default=StatusEnum.ERROR, description="Error status")
    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request identifier for debugging")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: StatusEnum = Field(default=StatusEnum.SUCCESS, description="Service status")
    message: str = Field(..., description="Status message")
    usage: Optional[str] = Field(None, description="Usage instructions")
    version: str = Field(default="0.1.0", description="API version")


class FileUploadMixin:
    """Mixin for file upload validation."""
    
    @staticmethod
    def validate_file_type(filename: str, allowed_extensions: List[str]) -> bool:
        """Validate file extension."""
        if not filename:
            return False
        ext = os.path.splitext(filename)[-1].lower()
        return ext in allowed_extensions
    
    @staticmethod
    def validate_file_size(file_size: int, max_size_mb: int = 50) -> bool:
        """Validate file size."""
        max_size_bytes = max_size_mb * 1024 * 1024
        return file_size <= max_size_bytes


class ProcessingConfig(BaseModel):
    """Configuration for processing parameters."""
    max_file_size_mb: int = Field(default=50, description="Maximum file size in MB")
    allowed_extensions: List[str] = Field(default=['.pdf'], description="Allowed file extensions")
    relevance_threshold: float = Field(default=0.4, description="Relevance threshold for filtering")
    max_content_length: int = Field(default=2048, description="Maximum content length for summarization")
    timeout_seconds: int = Field(default=300, description="Processing timeout in seconds")
