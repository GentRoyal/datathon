import os
import logging
from datetime import datetime
from pathlib import Path
from io import BytesIO

import PyPDF2
import docx
import textract
from fastapi import HTTPException

logger = logging.getLogger(__name__)

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract text from various file formats"""
    file_ext = Path(filename).suffix.lower()
        
    try:
        if file_ext == '.pdf':
            return _extract_from_pdf(file_content)
        elif file_ext in ['.docx']:
            return _extract_from_docx(file_content)
        elif file_ext == '.txt':
            return file_content.decode('utf-8')
        else:
                # Fallback to textract for other formats
            return _extract_with_textract(file_content)
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {e}")
        raise HTTPException(status_code = 422, detail = f"Could not extract text from file: {str(e)}")
    
    
def _extract_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF"""
    pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text
    
    
def _extract_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX"""
    doc = docx.Document(BytesIO(file_content))
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text
    
    
def _extract_with_textract(file_content: bytes) -> str:
    """Fallback extraction using textract"""
    # Save temporarily and extract
    temp_path = f"/tmp/temp_doc_{datetime.now().timestamp()}"
    with open(temp_path, 'wb') as f:
        f.write(file_content)
        
    try:
        text = textract.process(temp_path).decode('utf-8')
        os.remove(temp_path)
        return text
        
    except:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise