"""
Minimal LitLens API Routes for basic functionality
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from fastapi.responses import JSONResponse

# === Configure logger ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Router setup ===
router = APIRouter(
    prefix="",
    tags=["litlens"],
    responses={404: {"description": "Not found"}},
)

@router.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "LitLens API is running",
        "version": "minimal"
    }

@router.post("/summarize-hf-pdfs")
async def summarize_hf_pdfs_minimal(
    files: List[UploadFile] = File(...),
    goal: str = Form("")
):
    """
    Minimal PDF summarization endpoint (placeholder for now)
    """
    try:
        logger.info(f"Received {len(files)} files with goal: '{goal}'")
        
        # Basic file validation
        summaries = []
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not a PDF"
                )
            
            # Placeholder summary (we'll improve this later)
            summaries.append({
                "filename": file.filename,
                "title": f"Analysis of {file.filename}",
                "summary": f"Minimal backend processing of {file.filename}. Goal: {goal}. This is a basic response while we build up the full functionality.",
                "metadata": {
                    "file_size": file.size,
                    "content_type": file.content_type
                }
            })
        
        return {
            "goal": goal,
            "summaries": summaries,
            "status": "success",
            "message": f"Processed {len(files)} files (minimal mode)",
            "backend": "minimal"
        }
        
    except Exception as e:
        logger.error(f"Error processing PDFs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )