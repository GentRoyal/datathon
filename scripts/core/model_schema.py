from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    session_id: str

class StartConversationResponse(BaseModel):
    session_id: str
    message: str
    phase: str
    
class TeacherMessageRequest(BaseModel):
    session_id: str
    message: str
    
class TeacherMessageResponse(BaseModel):
    session_id: str
    assistant_message: str
    current_phase: str
    questions_asked: int
    is_complete: bool
    
    
class ConversationMessage(BaseModel):
    role: str
    content: str


class ConversationHistoryResponse(BaseModel):
    session_id: str
    history: List[ConversationMessage]
    current_phase: str
    total_messages: int

class EndConversationResponse(BaseModel):
    message: str
    session_id: str
    
class SessionInfo(BaseModel):
    session_id: str
    current_phase: str
    questions_asked: int
    conversation_length: int


class ActiveSessionsResponse(BaseModel):
    count: int
    sessions: List[SessionInfo]
    
class AssessmentReportResponse(BaseModel):
    session_id: str
    topic: str
    grade: str
    assessment_text: str
    conversation_length: int
    generated_at: str
    

class DocAnalysisRequest(BaseModel):
    query_text: str
    top_k: Optional[int] = 5

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    extracted_text_preview: str

class ConversationPhase(Enum):
    INITIAL_ASSESSMENT = "initial_assessment"
    CONCEPT_EXPLORATION = "concept_exploration"
    PEDAGOGY_DEVELOPMENT = "pedagogy_development"
    CULTURAL_INTEGRATION = "cultural_integration"
    READINESS_CHECK = "readiness_check"
    STUDENT_ROLEPLAY = "student_roleplay"
    PARENT_ROLEPLAY = "parent_roleplay"
    ADMINISTRATOR_ROLEPLAY = "administrator_roleplay"
    FINAL_ASSESSMENT = "final_assessment"

class CulturalContext(Enum):
    NIGERIAN = "Nigerian"
    
@dataclass
class CulturalElements:
    """Cultural context for lesson planning"""
    proverbs: List[str]
    historical_examples: List[str]
    local_names: List[str]
    cultural_practices: List[str]
    regional_context: str

@dataclass
class TeacherProfile:
    """Accumulated knowledge about the teacher"""
    topic: Optional[str] = None
    grade_level: Optional[str] = None
    pedagogical_approach: Optional[str] = None
    cultural_context: Optional[CulturalContext] = None
    content_understanding: Dict[str, str] = None
    teaching_strategies: List[str] = None
    cultural_integration_ideas: List[str] = None
    
    def __post_init__(self):
        if self.content_understanding is None:
            self.content_understanding = {}
        if self.teaching_strategies is None:
            self.teaching_strategies = []
        if self.cultural_integration_ideas is None:
            self.cultural_integration_ideas = []

class ReadinessAssessment(BaseModel):
    """Assessment of teacher's readiness"""
    content_mastery: int = Field(description="Content mastery score 1-10")
    pedagogical_preparedness: int = Field(description="Pedagogical readiness 1-10")
    cultural_relevance: int = Field(description="Cultural integration score 1-10")
    student_interaction_readiness: int = Field(description="Student roleplay score 1-10")
    parent_communication_readiness: int = Field(description="Parent roleplay score 1-10")
    administrator_readiness: int = Field(description="Administrator roleplay score 1-10")
    overall_readiness: str = Field(description="Overall readiness: Not Ready, Needs Improvement, Ready, Highly Ready")
    strengths: List[str] = Field(description="Key strengths demonstrated")
    areas_for_improvement: List[str] = Field(description="Areas needing work")
    recommendations: List[str] = Field(description="Specific recommendations")
