"""
OpenAI-Only LitLens API Routes

Complete implementation using only OpenAI APIs for:
- PDF text extraction
- GPT-4 summarization  
- Ada embeddings for relevance filtering
- Report generation
"""

import os
import logging
import tempfile
import asyncio
from typing import List, Dict, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from fastapi.responses import JSONResponse
import openai
from openai import OpenAI
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Router setup
router = APIRouter(
    prefix="",
    tags=["litlens-openai"],
    responses={404: {"description": "Not found"}},
)

class OpenAIProcessor:
    """OpenAI-powered PDF processing and summarization"""
    
    def __init__(self):
        self.embeddings_model = "text-embedding-ada-002"
        self.chat_model = "gpt-4"
        self.max_tokens = 3000
        
    async def extract_pdf_text(self, file_content: bytes, filename: str) -> str:
        """Extract text from PDF using pdfminer.six"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(file_content)
                temp_file.flush()
                
                # Extract text
                text = extract_text(temp_file.name)
                
                # Clean up
                os.unlink(temp_file.name)
                
                if not text.strip():
                    raise ValueError("No text could be extracted from PDF")
                    
                return text.strip()
                
        except Exception as e:
            logger.error(f"PDF extraction failed for {filename}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract text from {filename}: {str(e)}"
            )
    
    def chunk_text(self, text: str, max_chunk_size: int = 3000) -> List[str]:
        """Split text into manageable chunks for processing"""
        if len(text) <= max_chunk_size:
            return [text]
        
        # Try to split on paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= max_chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from OpenAI"""
        try:
            response = client.embeddings.create(
                model=self.embeddings_model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embeddings: {str(e)}"
            )
    
    async def filter_relevant_chunks(self, chunks: List[str], goal: str, threshold: float = 0.7) -> List[str]:
        """Filter chunks based on relevance to research goal using embeddings"""
        if not goal.strip():
            return chunks[:5]  # Return first 5 chunks if no goal specified
        
        try:
            # Get embeddings for goal and chunks
            all_texts = [goal] + chunks
            embeddings = await self.get_embeddings(all_texts)
            
            goal_embedding = np.array(embeddings[0]).reshape(1, -1)
            chunk_embeddings = np.array(embeddings[1:])
            
            # Calculate similarities
            similarities = cosine_similarity(goal_embedding, chunk_embeddings)[0]
            
            # Filter chunks above threshold
            relevant_chunks = []
            for i, sim in enumerate(similarities):
                if sim >= threshold:
                    relevant_chunks.append(chunks[i])
            
            # If no chunks meet threshold, return top 3 most similar
            if not relevant_chunks:
                top_indices = np.argsort(similarities)[-3:][::-1]
                relevant_chunks = [chunks[i] for i in top_indices]
            
            logger.info(f"Filtered {len(relevant_chunks)} relevant chunks from {len(chunks)} total")
            return relevant_chunks
            
        except Exception as e:
            logger.warning(f"Relevance filtering failed: {e}, returning first 5 chunks")
            return chunks[:5]
    
    async def summarize_with_openai(self, text: str, goal: str, filename: str) -> str:
        """Generate summary using OpenAI GPT-4"""
        try:
            prompt = f"""Please analyze and summarize the following academic paper excerpt in relation to this research goal: "{goal}"

Paper: {filename}

Content:
{text}

Please provide a comprehensive summary that:
1. Identifies key findings relevant to the research goal
2. Highlights important methodologies or approaches
3. Notes any limitations or future research directions
4. Explains how this relates to the specified research goal

Summary:"""

            response = client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are an expert research assistant specializing in academic literature analysis. Provide clear, comprehensive summaries that highlight relevance to specified research goals."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI summarization failed for {filename}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Summarization failed for {filename}: {str(e)}"
            )

# Initialize processor
processor = OpenAIProcessor()

@router.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "LitLens OpenAI API is running",
        "version": "openai-only",
        "models": {
            "chat": processor.chat_model,
            "embeddings": processor.embeddings_model
        }
    }

@router.post("/summarize-pdfs")
async def summarize_pdfs_openai(
    files: List[UploadFile] = File(...),
    goal: str = Form("")
):
    """
    Complete PDF summarization using OpenAI APIs
    - Extracts text from PDFs
    - Filters content by relevance to research goal
    - Generates AI summaries using GPT-4
    """
    try:
        # Validate OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
            )
        
        logger.info(f"Processing {len(files)} files with goal: '{goal}'")
        
        summaries = []
        
        for file in files:
            # Validate file type
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not a PDF"
                )
            
            try:
                # Read file content
                file_content = await file.read()
                
                # Extract text from PDF
                logger.info(f"Extracting text from {file.filename}")
                text = await processor.extract_pdf_text(file_content, file.filename)
                
                # Split into chunks
                chunks = processor.chunk_text(text)
                logger.info(f"Split {file.filename} into {len(chunks)} chunks")
                
                # Filter relevant chunks based on goal
                if goal.strip():
                    relevant_chunks = await processor.filter_relevant_chunks(chunks, goal)
                else:
                    relevant_chunks = chunks[:3]  # Use first 3 chunks if no goal
                
                # Combine relevant chunks
                relevant_text = "\n\n".join(relevant_chunks)
                
                # Generate summary using OpenAI
                logger.info(f"Generating summary for {file.filename}")
                summary = await processor.summarize_with_openai(relevant_text, goal, file.filename)
                
                summaries.append({
                    "filename": file.filename,
                    "title": f"AI Analysis of {file.filename}",
                    "summary": summary,
                    "metadata": {
                        "file_size": len(file_content),
                        "chunks_processed": len(relevant_chunks),
                        "total_chunks": len(chunks),
                        "model_used": processor.chat_model,
                        "embeddings_model": processor.embeddings_model
                    }
                })
                
                logger.info(f"Successfully processed {file.filename}")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                summaries.append({
                    "filename": file.filename,
                    "title": f"Processing Error - {file.filename}",
                    "summary": f"Failed to process {file.filename}: {str(e)}",
                    "metadata": {
                        "error": str(e),
                        "status": "failed"
                    }
                })
        
        return {
            "goal": goal,
            "summaries": summaries,
            "status": "success",
            "message": f"Processed {len(files)} files using OpenAI",
            "processing_info": {
                "model": processor.chat_model,
                "embeddings_model": processor.embeddings_model,
                "backend": "openai-only"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in PDF processing: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@router.get("/models")
async def get_models():
    """Get information about available OpenAI models"""
    return {
        "chat_model": processor.chat_model,
        "embeddings_model": processor.embeddings_model,
        "provider": "openai",
        "features": [
            "pdf_text_extraction",
            "semantic_similarity",
            "relevance_filtering",
            "ai_summarization"
        ]
    }