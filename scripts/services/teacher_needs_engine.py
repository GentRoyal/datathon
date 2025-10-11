from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

import logging
from config import Config
config = Config()
model = config.GROQ_MODEL
api_key = config.GROQ_API_KEY


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PriorityLevel(str, Enum):
    """Priority levels for learning needs"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ReadinessStatus(str, Enum):
    """Readiness status categories"""
    READY = "ready"
    DEVELOPING = "developing"
    NEEDS_SUPPORT = "needs_support"
    CRITICAL_NEED = "critical_need"


class TopicReadinessScore(BaseModel):
    topic: str
    readiness_score: float
    confidence: float
    identified_gaps: List[str]
    identified_strengths: List[str]
    evidence_quotes: List[str]
    priority: str
    suggested_interventions: List[str]

class TeacherNeedsAnalysis(BaseModel):
    teacher_id: Optional[str] = None
    analysis_timestamp: datetime = datetime.now()
    overall_readiness: float
    topic_scores: List[TopicReadinessScore]
    recommended_pathways: List[str]
    summary: str
    emotional_indicators: Dict[str, str]
    learning_style_hints: List[str]

class AnalysisCache:
    """Simple in-memory cache for analysis results"""
    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, TeacherNeedsAnalysis] = {}
        self.max_size = max_size
        self.access_times: Dict[str, datetime] = {}
    
    def get(self, key: str) -> Optional[TeacherNeedsAnalysis]:
        """Retrieve cached analysis"""
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: TeacherNeedsAnalysis):
        """Store analysis in cache"""
        if len(self.cache) >= self.max_size:
            # Remove oldest accessed item
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = datetime.now()
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.access_times.clear()


class TeacherNeedsEngine:
    """
    Core engine for analyzing teacher needs and generating readiness scores.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = model,
        temperature: float = 0.1,
        enable_streaming: bool = False,
        enable_caching: bool = True,
        cache_size: int = 100,
        custom_prompt_template: Optional[str] = None,
        max_retries: int = 3
    ):
        """
        Initialize the needs analysis engine.
        """
        callbacks = None
        if enable_streaming:
            callbacks = CallbackManager([StreamingStdOutCallbackHandler()])
        
        self.llm = ChatGroq(
            api_key=api_key,
            model=model,
            temperature=temperature,
            callback_manager=callbacks,
            max_retries=max_retries
        )
        
        self.parser = PydanticOutputParser(pydantic_object=TeacherNeedsAnalysis)
        self.custom_prompt_template = custom_prompt_template
        self.cache = AnalysisCache(max_size=cache_size) if enable_caching else None
        
        logger.info(f"TeacherNeedsEngine initialized with model: {model}")
    
    def _generate_cache_key(
        self,
        teacher_input: str,
        curriculum_topics: List[str],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate a cache key for the analysis"""
        import hashlib
        key_data = f"{teacher_input}_{sorted(curriculum_topics)}_{context}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def analyze_teacher_needs(
        self,
        teacher_input: str,
        curriculum_topics: List[str],
        teacher_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        analysis_depth: str = "comprehensive"
    ) -> TeacherNeedsAnalysis:
        """
        Analyze unstructured teacher input to generate topic-specific readiness scores
        """
        
        # Check cache first
        if use_cache and self.cache:
            cache_key = self._generate_cache_key(teacher_input, curriculum_topics, context)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached analysis")
                return cached_result
        
        # Build context string if provided
        context_str = ""
        if context:
            context_str = "\n\nAdditional Context:\n"
            for key, value in context.items():
                context_str += f"- {key}: {value}\n"
        
        # Adjust analysis instructions based on depth
        depth_instructions = self._get_depth_instructions(analysis_depth)
        format_instructions = self.parser.get_format_instructions()

        # Use custom template if provided, otherwise use default
        template = """
        You are an expert educational analyst specializing in teacher professional development and curriculum readiness assessment.

        Your task is to analyze the following unstructured input from a teacher and assess their readiness across specific curriculum topics. Extract nuanced insights from their own words to identify knowledge gaps, concerns, strengths, and areas for development.

        Curriculum Topics to Assess:
        {topics}

        Teacher Input:
        {teacher_input}
        {context}

        {depth_instructions}

        Analysis Instructions:
        1. For each curriculum topic, assign a readiness score from 0.0 (no readiness) to 1.0 (fully ready)
        2. Identify specific knowledge or skill gaps mentioned or implied
        3. Identify existing strengths and relevant prior knowledge
        4. Extract direct quotes that support your assessment
        5. Suggest specific interventions tailored to the identified gaps
        6. Assign priority levels based on:
        - critical: Fundamental gaps that block teaching ability
        - high: Significant gaps that would impair teaching quality
        - medium: Moderate gaps that could be addressed over time
        - low: Minor gaps or areas of relative strength
        7. Calculate an overall readiness score (weighted by priority)
        8. Recommend a learning pathway (ordered list of topics to address)
        9. Detect emotional indicators (anxiety, confidence, enthusiasm, overwhelm, etc.)
        10. Infer learning style hints from how they describe their needs

        Analyze the teacher input and return a JSON that EXACTLY matches this schema.
        Do not change any field names or add extras.

        {format_instructions}
        """
        
        # Create the analysis prompt
        prompt = ChatPromptTemplate.from_template(template)
        # format_instructions="Return your answer strictly as a valid JSON object that fits the TeacherNeedsAnalysis model — include actual values, not the schema or examples."
        format_instructions = """
Return ONLY a valid JSON object with this EXACT structure (replace examples with actual analysis):

{
    "teacher_id": "string or null",
    "analysis_timestamp": "ISO datetime string",
    "overall_readiness": 0.75,
    "topic_scores": [
        {
            "topic": "Topic Name",
            "readiness_score": 0.8,
            "confidence": 0.7,
            "identified_gaps": ["specific gap 1", "specific gap 2"],
            "identified_strengths": ["strength 1", "strength 2"],
            "evidence_quotes": ["direct quote from teacher input"],
            "priority": "high",
            "suggested_interventions": ["intervention 1", "intervention 2"]
        }
    ],
    "recommended_pathways": ["topic to learn first", "topic to learn second"],
    "summary": "Brief summary of overall readiness",
    "emotional_indicators": {
        "anxiety": "description if present",
        "confidence": "description if present"
    },
    "learning_style_hints": ["hint 1", "hint 2"]
}

CRITICAL: 
- Return ONLY the JSON object, no markdown, no explanations, no code blocks
- All fields are required
- Use actual analyzed values, not placeholders
- priority must be one of: "critical", "high", "medium", "low"
"""
        
        # Format the prompt
        formatted_prompt = prompt.format(
            topics="\n".join([f"- {topic}" for topic in curriculum_topics]),
            teacher_input=teacher_input,
            context=context_str,
            depth_instructions=depth_instructions,
            format_instructions = format_instructions
        )
        
        try:
            # Generate analysis
            logger.info("Generating teacher needs analysis")
            response = self.llm.invoke(formatted_prompt)
            
            # Parse the response
            analysis = self.parser.parse(response.content)
            
            # Add teacher_id and timestamp
            if teacher_id:
                analysis.teacher_id = teacher_id
            analysis.analysis_timestamp = datetime.now()
            
            # Cache the result
            if use_cache and self.cache:
                self.cache.set(cache_key, analysis)
            
            logger.info("Analysis completed successfully")
            return analysis
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise

    
    def _get_depth_instructions(self, depth: str) -> str:
        """Get analysis depth-specific instructions"""
        depth_map = {
            "quick": "Focus on high-level assessment. Provide 1-2 gaps per topic and brief evidence.",
            "standard": "Provide moderate detail. Include 2-4 gaps per topic with supporting evidence.",
            "comprehensive": "Provide deep analysis. Include detailed gaps, strengths, emotional indicators, and learning style hints. Extract multiple evidence quotes."
        }
        return depth_map.get(depth, depth_map["standard"])
    
    def get_triggered_content(
        self,
        analysis: TeacherNeedsAnalysis,
        content_mapping: Dict[str, Dict[str, Any]],
        threshold: float = 0.7,
        max_items: Optional[int] = None,
        priority_filter: Optional[List[PriorityLevel]] = None
    ) -> List[Dict[str, Any]]:
        """
        Trigger appropriate learning content based on readiness scores.
        
        Args:
            analysis: TeacherNeedsAnalysis object
            content_mapping: Dictionary mapping topics to content resources
            threshold: Score below which content is triggered (default 0.7)
            max_items: Maximum number of content items to return
            priority_filter: Only include specific priority levels
        
        Returns:
            List of triggered content resources in priority order
        """
        triggered_content = []
        
        # Priority mapping for sorting
        priority_order = {
            PriorityLevel.CRITICAL: 0,
            PriorityLevel.HIGH: 1,
            PriorityLevel.MEDIUM: 2,
            PriorityLevel.LOW: 3
        }
        
        # Sort topics by priority and readiness score
        sorted_scores = sorted(
            analysis.topic_scores,
            key=lambda x: (priority_order.get(x.priority, 4), x.readiness_score)
        )
        
        for topic_score in sorted_scores:
            # Apply priority filter if specified
            if priority_filter and topic_score.priority not in priority_filter:
                continue
            
            # Trigger content if below threshold
            if topic_score.readiness_score < threshold:
                topic = topic_score.topic
                
                if topic in content_mapping:
                    content = content_mapping[topic].copy()
                    content["triggered_by"] = {
                        "readiness_score": topic_score.readiness_score,
                        "priority": topic_score.priority, #here
                        "identified_gaps": topic_score.identified_gaps,
                        "identified_strengths": topic_score.identified_strengths,
                        "evidence": topic_score.evidence_quotes,
                        "suggested_interventions": topic_score.suggested_interventions
                    }
                    triggered_content.append(content)
                    
                    # Stop if max_items reached
                    if max_items and len(triggered_content) >= max_items:
                        break
        
        logger.info(f"Triggered {len(triggered_content)} content items")
        return triggered_content
    
    def generate_dashboard_data(
        self,
        analysis,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate structured data for teacher dashboard display.

        """
        dashboard = {
            "teacher_id": analysis.teacher_id,
            "timestamp": analysis.analysis_timestamp.isoformat(),
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
                    "strengths": score.identified_strengths,
                    "status": self._get_readiness_status(score.readiness_score),
                    "interventions": score.suggested_interventions
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
            # "triggered_content": triggered_content,
            # "action_required": len(triggered_content) > 0,
            "emotional_indicators": analysis.emotional_indicators,
            "learning_style_hints": analysis.learning_style_hints
        }
        
        if include_recommendations:
            dashboard["recommendations"] = self._generate_recommendations(analysis)
        
        return dashboard
    
    def _generate_recommendations(
        self,
        analysis: TeacherNeedsAnalysis,
    ) -> Dict[str, Any]:
        """Generate actionable recommendations"""
        recommendations = {
            "immediate_actions": [],
            "short_term_goals": [],
            "long_term_goals": [],
            "support_needed": []
        }
        
        # Immediate actions for critical items
        critical_topics = [
            score for score in analysis.topic_scores
            if score.priority == PriorityLevel.CRITICAL and score.readiness_score < 0.5
        ]
        
        if critical_topics:
            recommendations["immediate_actions"] = [
                f"Begin {topic.topic} training immediately - foundational for teaching readiness"
                for topic in critical_topics[:3]
            ]
        
        # Short-term goals for high priority
        high_priority = [
            score for score in analysis.topic_scores
            if score.priority == PriorityLevel.HIGH and score.readiness_score < 0.7
        ]
        
        if high_priority:
            recommendations["short_term_goals"] = [
                f"Complete {topic.topic} module within 2 weeks"
                for topic in high_priority[:3]
            ]
        
        # Long-term for medium/low
        medium_low = [
            score for score in analysis.topic_scores
            if score.priority in [PriorityLevel.MEDIUM, PriorityLevel.LOW]
            and score.readiness_score < 0.8
        ]
        
        if medium_low:
            recommendations["long_term_goals"] = [
                f"Enhance {topic.topic} skills through ongoing practice"
                for topic in medium_low[:3]
            ]
        
        # Detect need for support
        if analysis.emotional_indicators:
            if any(emotion in analysis.emotional_indicators for emotion in ["anxiety", "overwhelm", "stress"]):
                recommendations["support_needed"].append("Consider peer mentoring or coaching support")
        
        return recommendations
    
    @staticmethod
    def _get_readiness_status(score: float) -> ReadinessStatus:
        """Convert numeric score to status label"""
        if score >= 0.8:
            return ReadinessStatus.READY
        elif score >= 0.6:
            return ReadinessStatus.DEVELOPING
        elif score >= 0.4:
            return ReadinessStatus.NEEDS_SUPPORT
        else:
            return ReadinessStatus.CRITICAL_NEED

class ContentRecommendationEngine:
    """
    Advanced content recommendation based on multiple signals.
    """
    
    def __init__(self, engine: TeacherNeedsEngine):
        self.engine = engine
    
    def recommend_content(
        self,
        analysis: TeacherNeedsAnalysis,
        content_catalog: Dict[str, List[Dict[str, Any]]],
        personalization_factors: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Recommend content using advanced personalization.
        
        Args:
            analysis: Teacher needs analysis
            content_catalog: Full catalog with multiple content options per topic
            personalization_factors: Learning style, time availability, etc.
        
        Returns:
            Personalized, ranked list of content recommendations
        """
        recommendations = []
        
        # Extract personalization signals
        learning_styles = analysis.learning_style_hints or []
        emotional_state = analysis.emotional_indicators
        
        for topic_score in analysis.topic_scores:
            if topic_score.readiness_score < 0.7:
                topic = topic_score.topic
                
                if topic in content_catalog:
                    # Get all content for this topic
                    topic_content = content_catalog[topic]
                    
                    # Score each content item
                    scored_content = []
                    for content in topic_content:
                        score = self._score_content(
                            content,
                            topic_score,
                            learning_styles,
                            emotional_state,
                            personalization_factors
                        )
                        scored_content.append((score, content))
                    
                    # Sort by score and take best match
                    scored_content.sort(reverse=True, key=lambda x: x[0])
                    
                    if scored_content:
                        best_content = scored_content[0][1].copy()
                        best_content["match_score"] = scored_content[0][0]
                        best_content["topic_priority"] = topic_score.priority
                        best_content["readiness_gap"] = 1.0 - topic_score.readiness_score
                        recommendations.append(best_content)
        
        # Sort recommendations by priority and match score
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(
            key=lambda x: (
                priority_order.get(x["topic_priority"], 4),
                -x["match_score"],
                -x["readiness_gap"]
            )
        )
        
        return recommendations
    
    def _score_content(
        self,
        content: Dict[str, Any],
        topic_score: TopicReadinessScore,
        learning_styles: List[str],
        emotional_state: Dict[str, str],
        personalization_factors: Optional[Dict[str, Any]]
    ) -> float:
        """Score content based on multiple factors"""
        score = 0.5  # Base score
        
        # Match learning style
        content_format = content.get("metadata", {}).get("format", "")
        
        for style in learning_styles:
            if "hands-on" in style.lower() and content_format in ["hands-on", "interactive"]:
                score += 0.2
            elif "visual" in style.lower() and content_format in ["video", "visual"]:
                score += 0.2
            elif "reading" in style.lower() and content_format in ["text", "reading"]:
                score += 0.15
        
        # Adjust for emotional state
        if emotional_state:
            if any(emotion in emotional_state for emotion in ["anxiety", "overwhelm"]):
                # Prefer shorter, easier content
                duration = content.get("metadata", {}).get("duration", "")
                if "1" in duration or "30 min" in duration:
                    score += 0.15
                
                difficulty = content.get("metadata", {}).get("difficulty", "")
                if difficulty == "beginner":
                    score += 0.1
        
        # Match difficulty to readiness
        difficulty = content.get("metadata", {}).get("difficulty", "")
        if topic_score.readiness_score < 0.3 and difficulty == "beginner":
            score += 0.15
        elif 0.3 <= topic_score.readiness_score < 0.6 and difficulty == "intermediate":
            score += 0.15
        elif topic_score.readiness_score >= 0.6 and difficulty == "advanced":
            score += 0.15
        
        # Apply personalization factors
        if personalization_factors:
            time_available = personalization_factors.get("time_available_hours")
            if time_available:
                content_duration = self._parse_duration(
                    content.get("metadata", {}).get("duration", "")
                )
                if content_duration and content_duration <= time_available:
                    score += 0.1
        
        return min(score, 1.0)
    
    @staticmethod
    def _parse_duration(duration_str: str) -> Optional[float]:
        """Parse duration string to hours"""
        try:
            if "hour" in duration_str:
                return float(duration_str.split()[0])
            elif "min" in duration_str:
                return float(duration_str.split()[0]) / 60
        except:
            return None
        return None