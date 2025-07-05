import os
import re
import fitz  # PyMuPDF

# --- Lightweight PDF Utilities ---

def extract_title_from_text(text):
    lines = text.strip().split('\n')
    for line in lines:
        clean_line = line.strip()
        if 10 < len(clean_line) < 200 and re.search(r'[A-Za-z]', clean_line):
            if not re.match(r'^\d+\.\s', clean_line):
                return clean_line
    return "Untitled Paper"

def extract_section_titles(text):
    section_titles = []
    section_pattern = re.compile(r'^(\d+)\.\s+(Abstract|Introduction|Methods|Materials|Results|Discussion|Conclusion|References|Acknowledgments)', re.IGNORECASE | re.MULTILINE)
    for match in section_pattern.finditer(text):
        section_num = match.group(1).strip()
        section_title = match.group(2).strip()
        section_titles.append((section_num, section_title))
    return section_titles

def extract_pdf_metadata(text):
    metadata = {}
    year_match = re.search(r'\b(19|20)\d{2}\b', text)
    if year_match:
        metadata['year'] = year_match.group(0)

    journal_match = re.search(r'(Journal|Proceedings|Conference)[^\n]{0,100}', text, re.IGNORECASE)
    if journal_match:
        metadata['journal'] = journal_match.group(0).strip()

    author_match = re.search(r'(?i)by\s+([A-Za-z,\s]+)', text[:1000])
    if author_match:
        metadata['authors'] = author_match.group(1).strip()

    return metadata

# --- PDF Extraction Helpers ---

def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        print(f"❌ Failed to extract text from {pdf_path}: {str(e)}")
        return ""

def extract_text_from_folder(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError(f"Invalid folder path: {folder_path}")

    papers = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(full_path)
            if text:
                papers.append({
                    "filename": filename,
                    "content": text
                })

    return papers
