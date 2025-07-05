from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tempfile
import os
import sys

# Add backend modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from backend.modules import pdf_extractor, summarizer
from backend.modules.relevance_filter import filter_relevant_papers_advanced, FilterConfig
from backend.utils import embedder_hf, output_writer

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def summarize_uploaded_pdfs(
    files: List[UploadFile] = File(..., description="PDF files to summarize"),
    goal: str = Form("", description="Research goal to guide summarization")
):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")

        # Extract text from PDFs
        extracted_papers = []
        for file in files:
            suffix = os.path.splitext(file.filename)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
                
            paper = pdf_extractor.extract_text_from_pdf(tmp_path)
            extracted_papers.append(paper)
            os.remove(tmp_path)

        if not extracted_papers:
            raise HTTPException(status_code=400, detail="No valid PDFs extracted.")

        # Embed and filter papers
        goal_embedding, paper_embeddings = embedder_hf.embed_goal_and_papers(goal, extracted_papers)
        relevant_indexes = filter_relevant_papers_advanced(goal_embedding, paper_embeddings)
        relevant_papers = [extracted_papers[i] for i in relevant_indexes]

        if not relevant_papers:
            raise HTTPException(status_code=404, detail="No papers matched the research goal.")

        # Summarize using HuggingFace (simpler than OpenAI for serverless)
        summaries = []
        for paper in relevant_papers:
            filename = paper.get("filename", "Unknown")
            title = paper.get("title", "Untitled")
            content = paper.get("content", "")
            
            # Truncate for HF API limits
            max_chars = 2048
            truncated_content = content[:max_chars]
            
            # Simple summarization for demo
            summary_text = f"Summary of {title}: [Content analysis would go here for {len(truncated_content)} characters]"
            
            summaries.append({
                "filename": filename,
                "title": title,
                "summary": summary_text
            })

        return {
            "goal": goal,
            "summaries": summaries,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


# For Vercel serverless functions
def handler(request):
    return app(request)