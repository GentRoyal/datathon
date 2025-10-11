from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from scripts.services.teacher_needs_engine import (
    TeacherNeedsEngine,
    TeacherNeedsAnalysis,
    ContentRecommendationEngine,
    api_key,
    model,
    logger
)

router = APIRouter()

try:
    analysis_engine = TeacherNeedsEngine(
        api_key=api_key,
        model=model,
        enable_caching=True,
        temperature=0.01  # Lower temp for stable JSON output
    )
    recommender_engine = ContentRecommendationEngine(engine=analysis_engine)
    logger.info("FastAPI: TeacherNeedsEngine initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize TeacherNeedsEngine: {e}")
    # Set to None so endpoints can check for failed dependency
    analysis_engine = None 
    recommender_engine = None

class AnalysisRequest(BaseModel):
    teacher_input: str
    curriculum_topics: List[str]
    teacher_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    analysis_depth: str = "comprehensive"
    use_cache: bool = True
    content_catalog: Dict[str, List[Dict[str, Any]]]

class ContentCatalogItem(BaseModel):
    """Schema for a single content item in the catalog."""
    content_id: str
    title: str
    type: str
    url: str
    metadata: Dict[str, Any]

class RecommendationRequest(BaseModel):
    """Input model for the advanced recommendation endpoint."""
    analysis: TeacherNeedsAnalysis = Field(..., description="The full analysis object from the /analyze-needs endpoint.")
    content_catalog: Dict[str, List[ContentCatalogItem]] = Field(..., description="A mapping of topic names to a list of available content items for that topic.")
    personalization_factors: Optional[Dict[str, Any]] = Field(None, description="Factors like 'time_available_hours' for fine-tuning recommendations.")


def get_example_content_catalog_for_docs() -> Dict[str, List[Dict[str, Any]]]:
    """Provides a sample content catalog structure."""
    return {
        "Artificial Intelligence Fundamentals": [
            {
                "content_id": "AI_101_video", "title": "AI: Visual Intro (Video)", "type": "video_series",
                "url": "/content/ai-fundamentals-v", "metadata": {"duration": "1 hour", "difficulty": "beginner", "format": "video"}
            },
            {
                "content_id": "AI_101_hands", "title": "AI: Hands-On Workshop", "type": "interactive_module",
                "url": "/content/ai-fundamentals-h", "metadata": {"duration": "3 hours", "difficulty": "intermediate", "format": "hands-on"}
            }
        ],
        "Robotics and Automation": [
            {
                "content_id": "ROBOT_101_quick", "title": "Robotics: 30 Min Quick Start", "type": "video_series",
                "url": "/content/robotics-quick", "metadata": {"duration": "30 min", "difficulty": "beginner", "format": "video"}
            },
        ]
    }

@router.post(
    "/api/analyze-needs",
    response_model=TeacherNeedsAnalysis,
    summary="Generate Teacher Readiness Analysis"
)
async def analyze_teacher_needs_endpoint(request: AnalysisRequest):
    """
    Analyzes unstructured teacher input (text) against curriculum topics 
    to generate a comprehensive readiness assessment using the Groq LLM.
    """
    if not analysis_engine:
        raise HTTPException(status_code=503, detail="Analysis engine is not available.")
        
    try:
        analysis = analysis_engine.analyze_teacher_needs(
            teacher_input=request.teacher_input,
            curriculum_topics=request.curriculum_topics,
            teacher_id=request.teacher_id,
            context=request.context,
            use_cache=request.use_cache,
            analysis_depth=request.analysis_depth
        )
        return analysis
    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM analysis or parsing failed: {e}")


@router.post(
    "/api/generate-dashboard",
    summary="Generate Full Dashboard Payload"
)
async def generate_dashboard_data_endpoint(request: AnalysisRequest):
    if not analysis_engine:
        raise HTTPException(status_code=503, detail="Analysis engine is not available.")
    
    try:
        
        analysis = analysis_engine.analyze_teacher_needs(
            teacher_input=request.teacher_input,
            curriculum_topics=request.curriculum_topics,
            teacher_id=request.teacher_id,
            context=request.context,
            use_cache=request.use_cache,
            analysis_depth=request.analysis_depth
        )

        dashboard_data = analysis_engine.generate_dashboard_data(
            analysis=analysis,
            include_recommendations=True
        )

        return dashboard_data

    except Exception as e:
        logger.error(f"Dashboard generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard data: {e}")

@router.post(
    "/api/recommend-content",
    summary="Personalized Content Recommendation"
)
async def recommend_content_endpoint(request: RecommendationRequest):
    """
    Takes a completed analysis object and a full content catalog to provide a 
    personalized, ranked list of content based on learning style hints and emotional indicators.
    """
    if not recommender_engine:
        raise HTTPException(status_code=503, detail="Recommendation engine is not available.")

    try:
        # Convert the Pydantic-validated content catalog back to the required structure
        content_catalog_dict = {
            topic: [item.model_dump() for item in content_list] 
            for topic, content_list in request.content_catalog.items()
        }
        
        recommendations = recommender_engine.recommend_content(
            analysis=request.analysis,
            content_catalog=content_catalog_dict,
            personalization_factors=request.personalization_factors
        )
        
        return recommendations
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Content recommendation failed: {e}")