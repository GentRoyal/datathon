import uuid
import os
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, HTTPException
from fastapi import Request

from scripts.services.chat_assistant import ConversationalTeacherAssistant
from scripts.core.model_schema import ConversationPhase, StartConversationResponse, TeacherMessageRequest, TeacherMessageResponse

import logging
logger = logging.getLogger(__name__)

from config import Config
config = Config()

router = APIRouter()

# In-memory storage for conversation sessions
active_conversations: Dict[str, ConversationalTeacherAssistant] = {}
session_timestamps: Dict[str, datetime] = {}


def update_session_timestamp(session_id: str):
    """Update the timestamp for a session to track activity"""
    session_timestamps[session_id] = datetime.now()


# Teacher Assistant Endpoints
@router.post("/api/teacher-assistant/start", response_model = StartConversationResponse)
async def start_teacher_conversation(request: Request):
    """
    Start a new conversation session with the AI Teacher Assistant
    
    Returns:
        - session_id: Unique identifier for this conversation session
        - message: Initial greeting from the assistant
        - phase: Current conversation phase
    """
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())

        data = await request.json()
        grade = data.get("grade")
        subject = data.get("subject")
        
        # Initialize the conversational assistant
        groq_api_key = config.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        assistant = ConversationalTeacherAssistant(groq_api_key=groq_api_key)
        
        # Start the conversation
        initial_message = assistant.start_conversation(grade, subject)
        
        # Store in active conversations
        active_conversations[session_id] = assistant
        update_session_timestamp(session_id)
        
        return StartConversationResponse(
            session_id=session_id,
            message=initial_message,
            phase=assistant.current_phase.value
        )
        
    except Exception as e:
        logger.error(f"Failed to start conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start conversation: {str(e)}")


@router.post("/api/teacher-assistant/message", response_model=TeacherMessageResponse)
async def send_teacher_message(request: TeacherMessageRequest):
    """
    Send a message in an ongoing conversation with the AI Teacher Assistant
    
    Args:
        - session_id: The conversation session identifier
        - message: Teacher's response/message
        
    Returns:
        - assistant_message: AI's response
        - current_phase: Current conversation phase
        - questions_asked: Number of questions asked in current phase
        - is_complete: Whether the conversation has reached final assessment
    """
    try:
        # Retrieve the conversation session
        if request.session_id not in active_conversations:
            raise HTTPException(status_code=404, detail="Conversation session not found. Please start a new conversation.")
        
        assistant = active_conversations[request.session_id]
        
        # Process the teacher's message
        response = assistant.process_teacher_response(request.message)
        
        # Update session timestamp
        update_session_timestamp(request.session_id)
        
        # Check if conversation is complete
        is_complete = assistant.current_phase == ConversationPhase.FINAL_ASSESSMENT
        
        return TeacherMessageResponse(
            session_id=request.session_id,
            assistant_message=response,
            current_phase=assistant.current_phase.value,
            questions_asked=assistant.questions_asked,
            is_complete=is_complete
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")
