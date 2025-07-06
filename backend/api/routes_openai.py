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
import tiktoken

load_dotenv()
load_dotenv('.env.local')  # Also load .env.local explicitly

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with better error handling
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY not found in environment variables!")
    logger.error("Please set OPENAI_API_KEY in .env.local file")
    client = None
else:
    client = OpenAI(api_key=api_key)
    logger.info("OpenAI client initialized successfully")

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
        self.max_context_tokens = 7500  # Leave buffer for response
        self.encoding = tiktoken.encoding_for_model("gpt-4")
    
    def clean_pdf_text(self, text: str) -> str:
        """Clean up common PDF text extraction formatting issues"""
        import re
        
        # Fix concatenated words by adding spaces before capital letters
        # that follow lowercase letters (but preserve acronyms)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'\s+', ' ', text)  # Multiple whitespace to single space
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Ensure space after sentences
        
        # Remove common PDF artifacts
        text = re.sub(r'□|▢|■|●|◦|►|▲|▼|◄|▶', ' ', text)  # Box/bullet characters
        text = re.sub(r'\(cid:\d+\)', '', text)  # CID characters
        
        return text.strip()
        
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
                
                # Clean up text formatting issues common in academic PDFs
                cleaned_text = self.clean_pdf_text(text.strip())
                return cleaned_text
                
        except Exception as e:
            logger.error(f"PDF extraction failed for {filename}: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract text from {filename}: {str(e)}"
            )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, max_tokens_per_chunk: int = 1000) -> List[str]:
        """Split text into chunks based on token count, not character count"""
        if self.count_tokens(text) <= max_tokens_per_chunk:
            return [text]
        
        # Try to split on paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            test_chunk = current_chunk + paragraph + "\n\n"
            if self.count_tokens(test_chunk) <= max_tokens_per_chunk:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If single paragraph is too long, split by sentences
                if self.count_tokens(paragraph) > max_tokens_per_chunk:
                    sentences = paragraph.split('. ')
                    sentence_chunk = ""
                    for sentence in sentences:
                        # Skip sentences that are too long by themselves
                        if self.count_tokens(sentence + ". ") > max_tokens_per_chunk:
                            continue
                            
                        test_sentence_chunk = sentence_chunk + sentence + ". "
                        if self.count_tokens(test_sentence_chunk) <= max_tokens_per_chunk:
                            sentence_chunk = test_sentence_chunk
                        else:
                            if sentence_chunk:
                                chunks.append(sentence_chunk.strip())
                            sentence_chunk = sentence + ". "
                    if sentence_chunk:
                        current_chunk = sentence_chunk
                    else:
                        current_chunk = ""
                else:
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
            return chunks[:3]  # Return first 3 chunks if no goal specified
        
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
            
            # Limit chunks to stay within context window
            filtered_chunks = []
            total_tokens = 0
            for chunk in relevant_chunks:
                chunk_tokens = self.count_tokens(chunk)
                if total_tokens + chunk_tokens < self.max_context_tokens:
                    filtered_chunks.append(chunk)
                    total_tokens += chunk_tokens
                else:
                    break
            relevant_chunks = filtered_chunks
            
            logger.info(f"Filtered {len(relevant_chunks)} relevant chunks from {len(chunks)} total")
            return relevant_chunks
            
        except Exception as e:
            logger.warning(f"Relevance filtering failed: {e}, returning first few chunks within token limit")
            # Fallback: return chunks that fit within token limit
            filtered_chunks = []
            total_tokens = 0
            for chunk in chunks:
                chunk_tokens = self.count_tokens(chunk)
                if total_tokens + chunk_tokens < self.max_context_tokens:
                    filtered_chunks.append(chunk)
                    total_tokens += chunk_tokens
                else:
                    break
            return filtered_chunks
    
    async def summarize_with_openai(self, text: str, goal: str, filename: str) -> str:
        """Generate summary using OpenAI GPT-4 with token limit verification"""
        try:
            system_message = "You are an expert research assistant specializing in academic literature analysis. Provide clear, comprehensive summaries that highlight relevance to specified research goals."
            
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

            # Count tokens to ensure we're within limits
            system_tokens = self.count_tokens(system_message)
            prompt_tokens = self.count_tokens(prompt)
            total_input_tokens = system_tokens + prompt_tokens
            
            logger.info(f"Token count for {filename}: {total_input_tokens} input + {self.max_tokens} max output = {total_input_tokens + self.max_tokens} total")
            logger.info(f"Content length being sent to OpenAI for {filename}: {len(text)} characters")
            
            # GPT-4 has 8192 token limit
            if total_input_tokens + self.max_tokens > 8000:
                # Reduce text size if still too large
                max_text_tokens = 8000 - system_tokens - self.max_tokens - 200  # 200 token buffer for prompt structure
                
                # Truncate text to fit
                text_tokens = self.encoding.encode(text)
                if len(text_tokens) > max_text_tokens:
                    truncated_tokens = text_tokens[:max_text_tokens]
                    text = self.encoding.decode(truncated_tokens)
                    logger.warning(f"Truncated text for {filename} to {max_text_tokens} tokens")
                
                # Rebuild prompt with truncated text
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
                
                final_tokens = self.count_tokens(system_message) + self.count_tokens(prompt)
                logger.info(f"Final token count for {filename}: {final_tokens} input tokens")

            response = client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_message},
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
        # Validate OpenAI API key and client
        if not client:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Please set OPENAI_API_KEY in .env.local file."
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
                logger.info(f"Extracting text from {file.filename} (size: {len(file_content)} bytes)")
                text = await processor.extract_pdf_text(file_content, file.filename)
                logger.info(f"Extracted {len(text)} characters from {file.filename}")
                
                # Split into chunks
                chunks = processor.chunk_text(text)
                logger.info(f"Split {file.filename} into {len(chunks)} chunks")
                
                # Log first chunk preview for debugging
                if chunks:
                    first_chunk_preview = chunks[0][:200] + "..." if len(chunks[0]) > 200 else chunks[0]
                    logger.info(f"First chunk preview for {file.filename}: {first_chunk_preview}")
                
                # Filter relevant chunks based on goal
                if goal.strip():
                    relevant_chunks = await processor.filter_relevant_chunks(chunks, goal)
                else:
                    relevant_chunks = chunks[:3]  # Use first 3 chunks if no goal
                
                # Combine relevant chunks
                relevant_text = "\n\n".join(relevant_chunks)
                logger.info(f"Combined {len(relevant_chunks)} chunks into {len(relevant_text)} characters for {file.filename}")
                
                # Check if we have any text to summarize
                if not relevant_text.strip():
                    logger.warning(f"No text found in {file.filename} - may be image-based PDF")
                    summary = f"Unable to extract text from {file.filename}. This may be an image-based PDF that requires OCR processing."
                else:
                    # Log sample of text being sent to AI
                    text_preview = relevant_text[:500] + "..." if len(relevant_text) > 500 else relevant_text
                    logger.info(f"Text preview for {file.filename}: {text_preview}")
                    
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