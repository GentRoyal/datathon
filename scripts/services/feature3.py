from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from config import Config

config = Config()
model = config.GROQ_MODEL
api_key = config.GROQ_API_KEY


class TopicReadinessScore(BaseModel):
    """Model for individual topic readiness assessment"""
    topic: str = Field(description="The curriculum topic being assessed")
    readiness_score: float = Field(
        description="Readiness score from 0.0 to 1.0",
        ge=0.0,
        le=1.0
    )
    confidence: float = Field(
        description="Confidence in the assessment from 0.0 to 1.0",
        ge=0.0,
        le=1.0
    )
    identified_gaps: List[str] = Field(
        description="Specific knowledge/skill gaps identified"
    )
    evidence_quotes: List[str] = Field(
        description="Direct quotes from teacher input supporting the assessment"
    )
    priority: str = Field(
        description="Priority level: critical, high, medium, or low"
    )


class TeacherNeedsAnalysis(BaseModel):
    """Complete analysis of teacher needs across all topics"""
    teacher_id: Optional[str] = Field(
        default=None,
        description="Identifier for the teacher"
    )
    overall_readiness: float = Field(
        description="Overall readiness score across all topics",
        ge=0.0,
        le=1.0
    )
    topic_scores: List[TopicReadinessScore] = Field(
        description="Readiness scores for each curriculum topic"
    )
    recommended_pathways: List[str] = Field(
        description="Ordered list of topics to address based on priority"
    )
    summary: str = Field(
        description="Brief summary of the teacher's overall readiness and needs"
    )


class TeacherNeedsEngine:
    """
    Core engine for analyzing teacher needs and generating readiness scores.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = model,
        temperature: float = 0.1
    ):
        """
        Initialize the needs analysis engine.
        
        Args:
            api_key: API key for Groq
            model: Name of the Groq model to use
            temperature: Temperature for LLM generation
        """
        self.llm = ChatGroq(
            api_key=api_key,
            model=model,
            temperature=temperature
        )
        self.parser = PydanticOutputParser(pydantic_object=TeacherNeedsAnalysis)
    
    def analyze_teacher_needs(
        self,
        teacher_input: str,
        curriculum_topics: List[str],
        teacher_id: Optional[str] = None,
        context: Optional[Dict[str, any]] = None
    ) -> TeacherNeedsAnalysis:
        """
        Analyze unstructured teacher input to generate topic-specific readiness scores.
        
        Args:
            teacher_input: Unstructured text from teacher (survey, transcript, notes)
            curriculum_topics: List of curriculum topics to assess
            teacher_id: Optional identifier for the teacher
            context: Optional additional context (previous assessments, role, etc.)
        
        Returns:
            TeacherNeedsAnalysis object with readiness scores and recommendations
        """
        
        # Build context string if provided
        context_str = ""
        if context:
            context_str = "\n\nAdditional Context:\n"
            for key, value in context.items():
                context_str += f"- {key}: {value}\n"
        
        # Create the analysis prompt
        prompt = ChatPromptTemplate.from_template(
            """You are an expert educational analyst specializing in teacher professional development and curriculum readiness assessment.

Your task is to analyze the following unstructured input from a teacher and assess their readiness across specific curriculum topics. Extract nuanced insights from their own words to identify knowledge gaps, concerns, and areas of strength.

Curriculum Topics to Assess:
{topics}

Teacher Input:
{teacher_input}
{context}

Analysis Instructions:
1. For each curriculum topic, assign a readiness score from 0.0 (no readiness) to 1.0 (fully ready)
2. Identify specific knowledge or skill gaps mentioned or implied
3. Extract direct quotes that support your assessment
4. Assign priority levels based on:
   - critical: Fundamental gaps that block teaching ability
   - high: Significant gaps that would impair teaching quality
   - medium: Moderate gaps that could be addressed over time
   - low: Minor gaps or areas of relative strength
5. Calculate an overall readiness score (average weighted by priority)
6. Recommend a learning pathway (ordered list of topics to address)

Be nuanced and thorough. Look for:
- Direct statements of uncertainty or lack of knowledge
- Questions that reveal gaps
- Concerns about teaching specific topics
- Language that suggests discomfort or unfamiliarity
- Absence of discussion about certain topics
- Misconceptions or incorrect understanding

{format_instructions}
"""
        )
        
        # Format the prompt
        formatted_prompt = prompt.format(
            topics="\n".join([f"- {topic}" for topic in curriculum_topics]),
            teacher_input=teacher_input,
            context=context_str,
            format_instructions=self.parser.get_format_instructions()
        )
        
        # Generate analysis
        response = self.llm.invoke(formatted_prompt)
        
        # Parse the response
        analysis = self.parser.parse(response.content)
        
        # Add teacher_id if provided
        if teacher_id:
            analysis.teacher_id = teacher_id
        
        return analysis
    
    def get_triggered_content(
        self,
        analysis: TeacherNeedsAnalysis,
        content_mapping: Dict[str, Dict[str, any]],
        threshold: float = 0.7
    ) -> List[Dict[str, any]]:
        """
        Trigger appropriate learning content based on readiness scores.
        
        Args:
            analysis: TeacherNeedsAnalysis object
            content_mapping: Dictionary mapping topics to content resources
                Format: {topic: {
                    "content_id": str,
                    "title": str,
                    "type": str,
                    "url": str,
                    "metadata": dict
                }}
            threshold: Score below which content is triggered (default 0.7)
        
        Returns:
            List of triggered content resources in priority order
        """
        triggered_content = []
        
        # Priority mapping for sorting
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        
        # Sort topics by priority and readiness score
        sorted_scores = sorted(
            analysis.topic_scores,
            key=lambda x: (priority_order.get(x.priority, 4), x.readiness_score)
        )
        
        for topic_score in sorted_scores:
            # Trigger content if below threshold
            if topic_score.readiness_score < threshold:
                topic = topic_score.topic
                
                if topic in content_mapping:
                    content = content_mapping[topic].copy()
                    content["triggered_by"] = {
                        "readiness_score": topic_score.readiness_score,
                        "priority": topic_score.priority,
                        "identified_gaps": topic_score.identified_gaps,
                        "evidence": topic_score.evidence_quotes
                    }
                    triggered_content.append(content)
        
        return triggered_content
    
    def generate_dashboard_data(
        self,
        analysis: TeacherNeedsAnalysis,
        triggered_content: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """
        Generate structured data for teacher dashboard display.
        
        Args:
            analysis: TeacherNeedsAnalysis object
            triggered_content: List of triggered content from get_triggered_content
        
        Returns:
            Dictionary with dashboard display data
        """
        return {
            "teacher_id": analysis.teacher_id,
            "timestamp": None,  # Should be added by calling code
            "overall_readiness": {
                "score": analysis.overall_readiness,
                "percentage": round(analysis.overall_readiness * 100, 1),
                "status": self._get_readiness_status(analysis.overall_readiness)
            },
            "summary": analysis.summary,
            "topic_breakdown": [
                {
                    "topic": score.topic,
                    "score": score.readiness_score,
                    "percentage": round(score.readiness_score * 100, 1),
                    "priority": score.priority,
                    "confidence": score.confidence,
                    "gaps": score.identified_gaps,
                    "status": self._get_readiness_status(score.readiness_score)
                }
                for score in analysis.topic_scores
            ],
            "learning_pathway": [
                {
                    "step": idx + 1,
                    "topic": topic,
                    "status": "not_started"
                }
                for idx, topic in enumerate(analysis.recommended_pathways)
            ],
            "triggered_content": triggered_content,
            "action_required": len(triggered_content) > 0
        }
    
    @staticmethod
    def _get_readiness_status(score: float) -> str:
        """Convert numeric score to status label"""
        if score >= 0.8:
            return "ready"
        elif score >= 0.6:
            return "developing"
        elif score >= 0.4:
            return "needs_support"
        else:
            return "critical_need"


# Example usage function
def example_usage():
    """
    Example demonstrating how to use the TeacherNeedsEngine
    """
    
    # Initialize the engine
    engine = TeacherNeedsEngine(
        api_key=api_key
    )
    
    # Define curriculum topics
    curriculum_topics = [
        "Artificial Intelligence Fundamentals",
        "Robotics and Automation",
        "Solar PV Technology",
        "Machine Learning Basics",
        "Programming for AI"
    ]
    
    # Sample teacher input (could be from survey, interview, etc.)
    teacher_input = """
    I'm really excited about teaching the new curriculum, but I have to admit
    I'm feeling quite overwhelmed. I've been teaching traditional science for 
    15 years, but AI and robotics are completely new to me.
    
    I understand the basics of how solar panels work from our current physics
    curriculum, so I think I can handle that part. But when it comes to AI,
    I honestly don't even know where to start. What's the difference between
    machine learning and AI? How do I explain neural networks to students when
    I barely understand them myself?
    
    The robotics component also worries me. I've never programmed anything
    beyond a basic calculator in Excel. How am I supposed to teach students
    to program robots?
    
    I really want to do well with this, but I need help getting up to speed.
    """
    
    # Analyze teacher needs
    analysis = engine.analyze_teacher_needs(
        teacher_input=teacher_input,
        curriculum_topics=curriculum_topics,
        teacher_id="TEACHER_001",
        context={
            "years_experience": 15,
            "subject_background": "Traditional Science",
            "previous_training": "None in AI/Robotics"
        }
    )
    
    # Define content mapping
    content_mapping = {
        "Artificial Intelligence Fundamentals": {
            "content_id": "AI_101",
            "title": "AI Fundamentals: Compressed Teaching Brief",
            "type": "interactive_module",
            "url": "/content/ai-fundamentals",
            "metadata": {"duration": "2 hours", "difficulty": "beginner"}
        },
        "Machine Learning Basics": {
            "content_id": "ML_101",
            "title": "Machine Learning for Teachers",
            "type": "video_series",
            "url": "/content/ml-basics",
            "metadata": {"duration": "3 hours", "difficulty": "beginner"}
        },
        "Programming for AI": {
            "content_id": "PROG_101",
            "title": "Python Programming for AI Education",
            "type": "hands_on_course",
            "url": "/content/programming-ai",
            "metadata": {"duration": "5 hours", "difficulty": "beginner"}
        },
        "Robotics and Automation": {
            "content_id": "ROBOT_101",
            "title": "Robotics Teaching Essentials",
            "type": "interactive_module",
            "url": "/content/robotics",
            "metadata": {"duration": "4 hours", "difficulty": "beginner"}
        },
        "Solar PV Technology": {
            "content_id": "SOLAR_101",
            "title": "Solar PV: From Science to Practice",
            "type": "reading_module",
            "url": "/content/solar-pv",
            "metadata": {"duration": "1.5 hours", "difficulty": "intermediate"}
        }
    }
    
    # Get triggered content
    triggered_content = engine.get_triggered_content(
        analysis=analysis,
        content_mapping=content_mapping,
        threshold=0.7
    )
    
    # Generate dashboard data
    dashboard_data = engine.generate_dashboard_data(
        analysis=analysis,
        triggered_content=triggered_content
    )
    
    # Print results
    print("=" * 80)
    print("TEACHER NEEDS ANALYSIS RESULTS")
    print("=" * 80)
    print(f"\nTeacher ID: {analysis.teacher_id}")
    print(f"Overall Readiness: {analysis.overall_readiness:.2%}")
    print(f"\nSummary: {analysis.summary}")
    
    print("\n" + "=" * 80)
    print("TOPIC READINESS SCORES")
    print("=" * 80)
    for score in analysis.topic_scores:
        print(f"\n{score.topic}")
        print(f"  Score: {score.readiness_score:.2%} (Priority: {score.priority})")
        print(f"  Gaps: {', '.join(score.identified_gaps)}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED LEARNING PATHWAY")
    print("=" * 80)
    for idx, topic in enumerate(analysis.recommended_pathways, 1):
        print(f"{idx}. {topic}")
    
    print("\n" + "=" * 80)
    print("TRIGGERED CONTENT")
    print("=" * 80)
    for content in triggered_content:
        print(f"\n{content['title']}")
        print(f"  Type: {content['type']}")
        print(f"  Triggered by: {content['triggered_by']['priority']} priority")
        print(f"  Readiness Score: {content['triggered_by']['readiness_score']:.2%}")
    
    return analysis, dashboard_data, triggered_content


if __name__ == "__main__":
    # Run example
    example_usage()