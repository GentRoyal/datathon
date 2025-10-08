from fastapi import APIRouter, File, UploadFile, HTTPException

from pathlib import Path
import uuid
from datetime import datetime
from typing import Dict

from scripts.core.document_processor import extract_text_from_file
from scripts.services.summary_assistant import KnowledgeCompressionSystem
from scripts.core.model_schema import ChatRequest, ChatResponse

import logging
logger = logging.getLogger(__name__)

from config import Config
config = Config()

router = APIRouter()

MAX_FILE_SIZE = config.MAX_FILE_SIZE
SUPPORTED_FORMATS = {'.txt', '.pdf', '.docx'}

# In-memory session storage
active_sessions: Dict[str, dict] = {}

@router.post("/api/ingest/document")
async def lesson_copilot(
    file: UploadFile = File(...)
):
    """Upload and process educational document"""
    
    content = await file.read()
    file_size = len(content)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 15MB)")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=415, 
            detail=f"Unsupported file format. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    try:
        extracted_text = extract_text_from_file(content, file.filename)
        
        if not extracted_text or len(extracted_text.strip()) < 50:
            raise HTTPException(
                status_code=400, 
                detail="Document appears to be empty or text extraction failed"
            )
        
        session_id = str(uuid.uuid4())
        
        kcs = KnowledgeCompressionSystem()
        
        compressed_content = kcs.process_material(
            material=extracted_text,
            region="Nigeria"
        )
        
        # Store session data
        active_sessions[session_id] = {
            "kcs": kcs,
            "filename": file.filename,
            "file_size": file_size,
            "upload_time": datetime.now().isoformat(),
            "original_text": extracted_text,
            "compressed_content": compressed_content,
            "created_at": datetime.now()
        }
        
        logger.info(f"Document processed successfully. Session: {session_id}")
        
        return {"session_id" : session_id}
        
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Upload processing failed: {str(e)}"
        )
    
    
@router.post("/api/document/chat", response_model=ChatResponse)
async def chat_with_document(request: ChatRequest):
    """Chat with AI about the uploaded document"""
    
    session_data = active_sessions.get(request.session_id)
    if not session_data:
        raise HTTPException(
            status_code=404, 
            detail="Session not found or expired. Please upload a document first."
        )
    
    try:
        kcs: KnowledgeCompressionSystem = session_data["kcs"]
        
        response = kcs.ask_question(request.message)
        
        logger.info(f"Chat query processed for session: {request.session_id}")
        
        return ChatResponse(
            response=response,
            session_id=request.session_id
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Chat processing failed: {str(e)}"
        )