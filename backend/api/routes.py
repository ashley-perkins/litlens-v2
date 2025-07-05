"""
LitLens API Routes

This module contains all the API endpoints for the LitLens backend service.
It provides endpoints for text summarization, PDF processing, and report generation.
"""

import os
import tempfile
import logging
import time
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer

from backend.api.models import (
    SummarizeRequest,
    SummarizeResponse, 
    PDFSummarizeResponse,
    PaperSummary,
    ReportRequest,
    ErrorResponse,
    HealthResponse,
    ProcessingConfig,
    FileUploadMixin,
    StatusEnum
)
from backend.modules import summarizer, pdf_extractor, embedder, relevance_filter, report_generator
from backend.utils import output_writer, hf_utils, embedder_hf
from backend.utils.pdf_utils import extract_pdf_metadata
from backend.utils.output_writer import sanitize_filename
from backend.utils.hf_utils import summarize_text_with_hf_api

# === Configure logger ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Configuration ===
config = ProcessingConfig()
security = HTTPBearer(auto_error=False)

# === Router setup ===
router = APIRouter(
    prefix="",
    tags=["litlens"],
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse},
    }
)

# === Helper Functions ===

def generate_request_id() -> str:
    """Generate a unique request ID for tracking."""
    return str(uuid.uuid4())[:8]


def create_error_response(
    error_type: str, 
    message: str, 
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> ErrorResponse:
    """Create standardized error response."""
    return ErrorResponse(
        error_type=error_type,
        message=message,
        details=details,
        request_id=request_id
    )


def validate_file_uploads(files: List[UploadFile]) -> None:
    """
    Validate uploaded files for security and format requirements.
    
    Args:
        files: List of uploaded files to validate
        
    Raises:
        HTTPException: If validation fails
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded"
        )
    
    for file in files:
        # Validate file type
        if not FileUploadMixin.validate_file_type(file.filename, config.allowed_extensions):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {file.filename}. Allowed: {config.allowed_extensions}"
            )
        
        # Validate file size
        if hasattr(file, 'size') and file.size:
            if not FileUploadMixin.validate_file_size(file.size, config.max_file_size_mb):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File too large: {file.filename}. Max size: {config.max_file_size_mb}MB"
                )


def validate_research_goal(goal: str) -> str:
    """
    Validate and sanitize research goal.
    
    Args:
        goal: Research goal string
        
    Returns:
        Sanitized research goal
        
    Raises:
        HTTPException: If validation fails
    """
    if not goal or not goal.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Research goal cannot be empty"
        )
    
    goal = goal.strip()
    if len(goal) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Research goal too long (max 1000 characters)"
        )
    
    return goal


async def process_uploaded_files(files: List[UploadFile]) -> List[Dict[str, Any]]:
    """
    Process uploaded files and extract text content.
    
    Args:
        files: List of uploaded files
        
    Returns:
        List of extracted paper dictionaries
        
    Raises:
        HTTPException: If processing fails
    """
    extracted_papers = []
    
    for file in files:
        try:
            # Create temporary file
            suffix = os.path.splitext(file.filename)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            # Extract text from PDF
            paper = pdf_extractor.extract_text_from_pdf(tmp_path)
            if paper and paper.get('content'):
                extracted_papers.append(paper)
            else:
                logger.warning(f"Failed to extract content from {file.filename}")
            
            # Clean up temporary file
            try:
                os.remove(tmp_path)
            except OSError:
                logger.warning(f"Failed to remove temporary file: {tmp_path}")
                
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            # Continue processing other files
            continue
    
    if not extracted_papers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid content could be extracted from uploaded files"
        )
    
    return extracted_papers


def filter_relevant_papers(
    goal: str, 
    papers: List[Dict[str, Any]], 
    use_hf: bool = False
) -> tuple[List[Dict[str, Any]], List[int]]:
    """
    Filter papers based on relevance to research goal.
    
    Args:
        goal: Research goal
        papers: List of paper dictionaries
        use_hf: Whether to use Hugging Face embeddings
        
    Returns:
        Tuple of (relevant_papers, relevant_indexes)
        
    Raises:
        HTTPException: If filtering fails
    """
    try:
        embedder_module = embedder_hf if use_hf else embedder
        goal_embedding, paper_embeddings = embedder_module.embed_goal_and_papers(goal, papers)
        
        relevant_indexes = relevance_filter.filter_relevant_papers(
            goal_embedding, 
            paper_embeddings, 
            threshold=config.relevance_threshold
        )
        
        if not relevant_indexes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No papers matched the research goal"
            )
        
        relevant_papers = [papers[i] for i in relevant_indexes]
        return relevant_papers, relevant_indexes
        
    except Exception as e:
        logger.error(f"Error filtering papers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to filter papers by relevance"
        )


def validate_file_path(file_path: str) -> Path:
    """
    Validate file path for security (prevent path traversal).
    
    Args:
        file_path: Path to validate
        
    Returns:
        Validated Path object
        
    Raises:
        HTTPException: If path is invalid or unsafe
    """
    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File path cannot be empty"
        )
    
    # Check for path traversal attempts
    if '..' in file_path or file_path.startswith('/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path"
        )
    
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found"
        )
    
    return path


# === API Endpoints ===

# 🚧 Route temporarily disabled to prevent public OpenAI API Usage
# 💡Re-enable when usage controls are in place
# @router.post("/summarize", response_model=SummarizeResponse)
# async def summarize_text(request: SummarizeRequest) -> SummarizeResponse:
#     """
#     Summarize text content using OpenAI API.
#     
#     Args:
#         request: Summarization request with goal and content
#         
#     Returns:
#         SummarizeResponse with summary and metadata
#     """
#     request_id = generate_request_id()
#     start_time = time.time()
#     
#     try:
#         logger.info(f"[{request_id}] Starting inline summarization")
#         
#         summary = summarizer.summarize_inline_text(request.content, request.goal)
#         processing_time = time.time() - start_time
#         
#         logger.info(f"[{request_id}] Inline summarization completed in {processing_time:.2f}s")
#         
#         return SummarizeResponse(
#             goal=request.goal,
#             summary=summary,
#             word_count=len(summary.split()),
#             processing_time=processing_time
#         )
#         
#     except Exception as e:
#         logger.error(f"[{request_id}] Inline summarization failed: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Summarization failed: {str(e)}"
#         )

@router.post("/summarize-hf", response_model=SummarizeResponse)
async def summarize_with_huggingface(request: SummarizeRequest) -> SummarizeResponse:
    """
    Summarize text content using Hugging Face API.
    
    Args:
        request: Summarization request with goal and content
        
    Returns:
        SummarizeResponse with summary and metadata
    """
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        logger.info(f"[{request_id}] Starting Hugging Face summarization")
        
        # Validate and truncate content if needed
        content = request.content
        if len(content) > config.max_content_length:
            content = content[:config.max_content_length]
            logger.warning(f"[{request_id}] Content truncated to {config.max_content_length} characters")
        
        summary = await hf_utils.summarize_text_with_hf_api(content)
        processing_time = time.time() - start_time
        
        logger.info(f"[{request_id}] Hugging Face summarization completed in {processing_time:.2f}s")
        
        return SummarizeResponse(
            goal=request.goal,
            summary=summary,
            word_count=len(summary.split()),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Hugging Face summarization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hugging Face summarization failed: {str(e)}"
        )

@router.get("/", response_model=HealthResponse, include_in_schema=False)
def root() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with service status and usage information
    """
    logger.info("Root endpoint accessed")
    return HealthResponse(
        message="LitLens backend is running",
        usage="Use the /summarize-hf endpoint or visit /docs for API documentation"
    )

# 🚧 Route temporarily disabled to prevent public OpenAI API Usage
# 💡Re-enable when usage controls are in place
# @router.post("/summarize-pdfs", response_model=PDFSummarizeResponse)
# async def summarize_uploaded_pdfs(
#     files: List[UploadFile] = File(..., description="PDF files to summarize"),
#     goal: str = Form("", description="Research goal to guide summarization")
# ) -> PDFSummarizeResponse:
#     """
#     Summarize uploaded PDF files using OpenAI API.
    
#     Args:
#         files: List of PDF files to process
#         goal: Research goal to guide summarization
        
#     Returns:
#         PDFSummarizeResponse with summaries and metadata
#     """
#     request_id = generate_request_id()
#     start_time = time.time()
    
#     try:
#         logger.info(f"[{request_id}] Starting PDF summarization pipeline with goal: {goal}")
        
#         # Validate inputs
#         validate_file_uploads(files)
#         goal = validate_research_goal(goal)
        
#         # Process files
#         extracted_papers = await process_uploaded_files(files)
#         logger.info(f"[{request_id}] Extracted {len(extracted_papers)} papers")
        
#         # Filter relevant papers
#         relevant_papers, relevant_indexes = filter_relevant_papers(goal, extracted_papers)
#         logger.info(f"[{request_id}] Found {len(relevant_papers)} relevant papers")
        
#         # Summarize papers
#         summaries = summarizer.summarize_papers(relevant_papers, goal)
        
#         # Generate output
#         output_path = output_writer.save_summary_to_file(summaries, goal)
#         processing_time = time.time() - start_time
        
#         logger.info(f"[{request_id}] PDF summarization completed in {processing_time:.2f}s")
        
#         # Convert to PaperSummary objects
#         paper_summaries = []
#         for summary in summaries:
#             paper_summaries.append(PaperSummary(
#                 filename=summary.get("filename", "Unknown"),
#                 title=summary.get("title", "Untitled"),
#                 summary=summary.get("summary", ""),
#                 word_count=len(summary.get("summary", "").split())
#             ))
        
#         return PDFSummarizeResponse(
#             goal=goal,
#             summaries=paper_summaries,
#             output_path=output_path,
#             total_papers=len(extracted_papers),
#             relevant_papers=len(relevant_papers),
#             processing_time=processing_time
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"[{request_id}] PDF summarization pipeline failed: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"PDF summarization pipeline failed: {str(e)}"
#         )

@router.post("/report", response_class=FileResponse)
def generate_report(request: ReportRequest) -> FileResponse:
    """
    Generate a markdown report from paper summaries.
    
    Args:
        request: ReportRequest with goal and summaries
        
    Returns:
        FileResponse with the generated markdown report
    """
    request_id = generate_request_id()
    
    try:
        logger.info(f"[{request_id}] Starting report generation")
        
        # Validate request
        if not request.summaries:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No summaries provided"
            )
        
        # Generate report
        report = report_generator.generate_markdown_report(request.summaries, request.goal)
        
        # Save to temp file
        safe_goal = sanitize_filename(request.goal)
        filename = f"{safe_goal}_summary_report.md"
        output_path = os.path.join(tempfile.gettempdir(), filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info(f"[{request_id}] Report generated successfully")
        return FileResponse(
            output_path, 
            media_type="text/markdown", 
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Failed to generate report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate report: {str(e)}"
        )


@router.get("/download", response_class=FileResponse)
async def download_file(
    path: str = Query(..., description="Path to the file on server")
) -> FileResponse:
    """
    Download a file from the server.
    
    Args:
        path: Path to the file on the server
        
    Returns:
        FileResponse with the requested file
    """
    request_id = generate_request_id()
    
    try:
        logger.info(f"[{request_id}] Download request for path: {path}")
        
        # Validate file path for security
        validated_path = validate_file_path(path)
        
        return FileResponse(
            validated_path,
            media_type="application/octet-stream",
            filename=validated_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Download failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Download failed: {str(e)}"
        )

@router.post("/summarize-hf-pdfs", response_model=PDFSummarizeResponse)
async def summarize_with_huggingface_pdfs(
    files: List[UploadFile] = File(..., description="PDF files to summarize"),
    goal: str = Form("", description="Research goal to guide summarization")
) -> PDFSummarizeResponse:
    """
    Summarize uploaded PDF files using Hugging Face API.
    
    Args:
        files: List of PDF files to process
        goal: Research goal to guide summarization
        
    Returns:
        PDFSummarizeResponse with summaries and metadata
    """
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        logger.info(f"[{request_id}] Starting HF PDF summarization pipeline with goal: {goal}")
        
        # Validate inputs
        validate_file_uploads(files)
        goal = validate_research_goal(goal)
        
        # Process files
        extracted_papers = await process_uploaded_files(files)
        logger.info(f"[{request_id}] Extracted {len(extracted_papers)} papers")
        
        # Filter relevant papers using HF embeddings
        relevant_papers, relevant_indexes = filter_relevant_papers(goal, extracted_papers, use_hf=True)
        logger.info(f"[{request_id}] Found {len(relevant_papers)} relevant papers")
        
        # Summarize papers using Hugging Face API
        summaries = []
        for paper in relevant_papers:
            try:
                filename = paper.get("filename", "Unknown")
                title = paper.get("title", "Untitled")
                content = paper.get("content", "")
                
                # Truncate content to max length
                truncated_content = content[:config.max_content_length]
                logger.debug(f"[{request_id}] Processing {filename}, content length: {len(truncated_content)}")
                
                summary_text = await summarize_text_with_hf_api(
                    truncated_content,
                    model_name="philschmid/bart-large-cnn-samsum"
                )
                
                summaries.append(PaperSummary(
                    filename=filename,
                    title=title,
                    summary=summary_text,
                    word_count=len(summary_text.split()),
                    extraction_success=True
                ))
                
            except Exception as summarization_error:
                logger.warning(f"[{request_id}] Summarization failed for {filename}: {summarization_error}")
                summaries.append(PaperSummary(
                    filename=filename,
                    title=title,
                    summary=f"[Summarization failed: {summarization_error}]",
                    word_count=0,
                    extraction_success=False
                ))
        
        # Generate report
        summaries_dict = [summary.dict() for summary in summaries]
        output_path = output_writer.save_summary_to_file(summaries_dict, goal)
        processing_time = time.time() - start_time
        
        logger.info(f"[{request_id}] HF PDF summarization completed in {processing_time:.2f}s")
        
        return PDFSummarizeResponse(
            goal=goal,
            summaries=summaries,
            output_path=output_path,
            total_papers=len(extracted_papers),
            relevant_papers=len(relevant_papers),
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] HF PDF summarization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"HF PDF summarization failed: {str(e)}"
        )