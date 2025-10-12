import uuid
import os
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, HTTPException
from fastapi import Request

from scripts.services.chat_assistant import ConversationalTeacherAssistant
from scripts.core.model_schema import StartConversationResponse, TeacherMessageRequest, TeacherMessageResponse

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
@router.post("/api/teacher-assistant/start", response_model=StartConversationResponse)
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
    Send a message with enhanced quality metrics in response
    
    Args:
        - session_id: The conversation session identifier
        - message: Teacher's response/message
        
    Returns:
        Enhanced response with quality metrics and progress tracking
    """
    try:
        # Retrieve the conversation session
        if request.session_id not in active_conversations:
            raise HTTPException(status_code=404, detail="Conversation session not found.")
        
        assistant = active_conversations[request.session_id]
        
        # Process the teacher's message
        response = assistant.process_teacher_response(request.message)
        
        # Update session timestamp
        update_session_timestamp(request.session_id)
        
        # Get full conversation state
        state = assistant.get_conversation_state()
        
        # Check if conversation is complete
        is_complete = assistant.is_assessment_complete()
        
        # Calculate progress percentage
        # Rough estimate based on phases completed
        phase_weights = {
            "INITIAL_ASSESSMENT": 10,
            "CONCEPT_EXPLORATION": 20,
            "PEDAGOGY_DEVELOPMENT": 30,
            "CULTURAL_INTEGRATION": 40,
            "READINESS_CHECK": 45,
            "STUDENT_ROLEPLAY": 60,
            "PARENT_ROLEPLAY": 75,
            "ADMINISTRATOR_ROLEPLAY": 90,
            "FINAL_ASSESSMENT": 100
        }
        progress = phase_weights.get(state["current_phase"], 0)
        
        return TeacherMessageResponse(
            session_id=request.session_id,
            assistant_message=response,
            current_phase=state["current_phase"],
            questions_asked=state["questions_in_phase"],
            is_complete=is_complete,
            substantive_responses=state["substantive_responses"],
            weak_response_count=state["weak_response_count"],
            last_question_asked=state.get("last_question"),
            progress_percentage=progress
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")


@router.get("/api/teacher-assistant/session/{session_id}/state")
async def get_session_state(session_id: str):
    """
    Get detailed state information for a conversation session
    
    Args:
        - session_id: The conversation session identifier
        
    Returns:
        Detailed conversation state including quality metrics
    """
    try:
        if session_id not in active_conversations:
            raise HTTPException(status_code=404, detail="Conversation session not found.")
        
        assistant = active_conversations[session_id]
        state = assistant.get_conversation_state()
        
        return {
            "session_id": session_id,
            "state": state,
            "timestamp": session_timestamps.get(session_id).isoformat() if session_id in session_timestamps else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session state: {str(e)}")


@router.delete("/api/teacher-assistant/session/{session_id}")
async def end_session(session_id: str):
    """
    End a conversation session and clean up resources
    
    Args:
        - session_id: The conversation session identifier
        
    Returns:
        Confirmation message
    """
    try:
        if session_id not in active_conversations:
            raise HTTPException(status_code=404, detail="Conversation session not found.")
        
        # Remove from active conversations
        del active_conversations[session_id]
        if session_id in session_timestamps:
            del session_timestamps[session_id]
        
        return {"message": "Session ended successfully", "session_id": session_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to end session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")


@router.get("/api/teacher-assistant/sessions")
async def list_active_sessions():
    """
    List all active conversation sessions
    
    Returns:
        List of active session IDs with timestamps and phases
    """
    try:
        sessions = []
        for session_id, assistant in active_conversations.items():
            sessions.append({
                "session_id": session_id,
                "current_phase": assistant.current_phase.value,
                "questions_asked": assistant.questions_asked,
                "is_complete": assistant.is_assessment_complete(),
                "last_activity": session_timestamps.get(session_id).isoformat() if session_id in session_timestamps else None,
                "extracted_subject": assistant.extracted_subject,
                "extracted_grade": assistant.extracted_grade
            })
        
        return {
            "active_sessions": len(sessions),
            "sessions": sessions
        }
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")
