import random
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser

from config import Config
config = Config()

from scripts.core.templates import greeting_templates
from scripts.core.model_schema import (
    ConversationPhase,
    CulturalContext,
    CulturalElements,
    TeacherProfile,
    ReadinessAssessment
)

CULTURAL_CONTEXTS = {
    CulturalContext.NIGERIAN: CulturalElements(
        proverbs = [
            "A child who says his mother will not sleep, he too will not sleep",
            "The lizard that jumped from the high Iroko tree said he would praise himself if no one else did",
            "No matter how long a log stays in the river, it does not become a crocodile",
            "When the music changes, so does the dance",
            "A single hand cannot tie a bundle",
            "The eagle does not chase flies"
        ],
        historical_examples=[
            "Queen Amina of Zazzau's leadership and military strategy",
            "The Benin Kingdom's bronze casting technology",
            "Chinua Achebe's literary contributions",
            "Funmilayo Ransome-Kuti's activism",
            "Wole Soyinka's Nobel Prize achievements",
            "Nnamdi Azikiwe's role in independence"
        ],
        local_names=[
            "Adebayo", "Chioma", "Ngozi", "Emeka", "Fatima", "Aisha",
            "Tunde", "Oluwatobi", "Chiamaka", "Ifeanyi", "Zainab", "Yusuf",
            "Folake", "Chukwudi", "Blessing", "Ibrahim"
        ],
        cultural_practices=[
            "Community decision-making through town hall meetings",
            "Oral tradition storytelling",
            "Extended family support systems",
            "Respect for elders and community leaders",
            "Communal learning and peer teaching",
            "Use of local languages in explanation"
        ],
        regional_context="Nigeria - diverse, multilingual, with strong community values and rich oral traditions"
    )
}


@dataclass
class ResponseQuality:
    """Structure for response quality assessment"""
    quality_level: str  # "weak", "superficial", "substantive"
    is_empty: bool
    is_evasive: bool
    word_count: int
    has_specifics: bool
    addresses_question: bool
    reasoning: str


@dataclass
class ConversationState:
    """Enhanced conversation state tracking"""
    last_question_asked: Optional[str] = None
    last_assistant_message: Optional[str] = None
    questions_in_current_phase: int = 0
    substantive_responses_in_phase: int = 0
    weak_response_count: int = 0
    phase_start_index: int = 0


@dataclass
class AssistantConfig:
    """Configuration for customizing assistant behavior"""
    min_questions_per_phase: int = 1
    max_questions_per_phase: int = 1
    min_substantive_responses: int = 1  # Required before phase transition
    enable_dynamic_scenarios: bool = True
    temperature: float = 0.7
    model_name: str = config.GROQ_MODEL
    cultural_context: CulturalContext = CulturalContext.NIGERIAN


class ConversationalTeacherAssistant:
    """
    Enhanced interactive AI assistant with robust context awareness and response validation
    """

    def __init__(
        self,
        groq_api_key: str,
        config: Optional[AssistantConfig] = None
    ):
        """Initialize the conversational AI Teacher's Assistant"""
        self.config = config or AssistantConfig()

        self.llm = ChatGroq(
            api_key=groq_api_key,
            model=self.config.model_name,
            temperature=self.config.temperature
        )

        self.conversation_history: List[Dict[str, str]] = []
        self.teacher_profile = TeacherProfile()
        self.teacher_profile.cultural_context = self.config.cultural_context

        self.current_phase = ConversationPhase.INITIAL_ASSESSMENT
        self.conversation_state = ConversationState()
        self.roleplay_scenarios_completed = []

        # Context extracted from conversation OR provided upfront
        self.extracted_subject: Optional[str] = None
        self.extracted_grade: Optional[str] = None

        # Roleplay character names
        self.student_name: Optional[str] = None

    # ============================================================================
    # HELPER METHODS - Core Utilities
    # ============================================================================

    def _get_last_assistant_message(self) -> Optional[str]:
        """Get the last message sent by the assistant"""
        for msg in reversed(self.conversation_history):
            if msg["role"] == "assistant":
                return msg["content"]
        return None

    def _get_last_question_asked(self) -> Optional[str]:
        """Extract the question from the last assistant message"""
        last_message = self._get_last_assistant_message()
        if not last_message:
            return None
        
        # Split by common question markers
        sentences = last_message.replace('?', '?\n').split('\n')
        questions = [s.strip() for s in sentences if '?' in s]
        
        return questions[-1] if questions else last_message

    def _add_to_history(self, role: str, content: str):
        """Add message to history and update state"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        if role == "assistant":
            self.conversation_state.last_assistant_message = content
            self.conversation_state.last_question_asked = self._get_last_question_asked()

    def _format_history(self, num_messages: Optional[int] = None) -> str:
        """Format conversation history"""
        messages = self.conversation_history[-num_messages:] if num_messages else self.conversation_history
        formatted = []
        for entry in messages:
            role = entry["role"].upper()
            content = entry["content"]
            formatted.append(f"{role}: {content}\n")
        return "\n".join(formatted)

    # ============================================================================
    # VALIDATION METHODS - Response Quality Assessment
    # ============================================================================

    def _check_response_quality(self, response: str, context: Optional[str] = None) -> ResponseQuality:
        """
        Comprehensive response quality check using LLM analysis
        """
        last_question = context or self.conversation_state.last_question_asked or "the previous question"
        
        prompt = ChatPromptTemplate.from_template(
            """You are an expert evaluator assessing a teacher's response quality.

            QUESTION ASKED: {question}

            TEACHER'S RESPONSE: "{response}"

            Evaluate this response comprehensively:

            1. ADDRESSES QUESTION: Does it actually answer what was asked? (yes/no)
            2. SPECIFICITY: Does it contain concrete details, examples, or plans? (yes/no)
            3. SUBSTANCE: Is it more than just acknowledgment or deflection? (yes/no)
            4. EFFORT: Does it show genuine engagement with the topic? (yes/no)

            QUALITY CLASSIFICATION:
            - "weak": Empty, "I don't know", one-word answers, complete deflection
            - "superficial": On-topic but vague, lacks specifics, generic statements
            - "substantive": Detailed, specific, shows understanding, addresses the question

            Return ONLY a JSON object:
            {{
            "quality_level": "weak|superficial|substantive",
            "addresses_question": true/false,
            "has_specifics": true/false,
            "is_evasive": true/false,
            "reasoning": "brief explanation of why you classified it this way"
            }}

            JSON:"""
        )

        try:
            chain = prompt | self.llm
            result = chain.invoke({
                "question": last_question,
                "response": response
            })

            content = result.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            evaluation = json.loads(content)
            
            return ResponseQuality(
                quality_level=evaluation.get("quality_level", "superficial"),
                is_empty=len(response.strip()) < 10,
                is_evasive=evaluation.get("is_evasive", False),
                word_count=len(response.split()),
                has_specifics=evaluation.get("has_specifics", False),
                addresses_question=evaluation.get("addresses_question", False),
                reasoning=evaluation.get("reasoning", "")
            )

        except Exception:
            # Fallback to simple heuristic
            return self._simple_quality_check(response)

    def _simple_quality_check(self, response: str) -> ResponseQuality:
        """Fallback quality check using heuristics"""
        response_lower = response.lower().strip()
        
        weak_patterns = ["i don't know", "not sure", "maybe", "idk", "dunno", 
                        "no idea", "unsure", "can't", "cannot"]
        generic_responses = ["yes", "no", "ok", "fine", "sure", "okay"]
        
        is_weak = any(pattern in response_lower for pattern in weak_patterns)
        is_generic = response_lower in generic_responses
        is_empty = len(response.strip()) < 10
        word_count = len(response.split())
        
        has_specifics = any(indicator in response_lower for indicator in 
                          ["will", "would", "example", "such as", "by", "using", 
                           "through", "first", "then", "specifically"])
        
        if is_empty or is_weak or is_generic:
            quality_level = "weak"
        elif word_count < 20 or not has_specifics:
            quality_level = "superficial"
        else:
            quality_level = "substantive"
        
        return ResponseQuality(
            quality_level=quality_level,
            is_empty=is_empty,
            is_evasive=is_weak,
            word_count=word_count,
            has_specifics=has_specifics,
            addresses_question=not (is_weak or is_generic),
            reasoning=f"Heuristic classification: {quality_level}"
        )

    def _should_transition_phase(self) -> bool:
        """
        Determine if sufficient quality responses have been received to progress
        """
        return (
            self.conversation_state.questions_in_current_phase >= self.config.max_questions_per_phase
            and self.conversation_state.substantive_responses_in_phase >= self.config.min_substantive_responses
        )

    # ============================================================================
    # PUBLIC INTERFACE METHODS
    # ============================================================================

    def start_conversation(self, grade: Optional[str] = None, subject: Optional[str] = None) -> str:
        """Start the conversation with a dynamic, personalized greeting"""
        if grade:
            self.extracted_grade = grade
        if subject:
            self.extracted_subject = subject

        greetings = greeting_templates(self.extracted_subject, self.extracted_grade)
        initial_message = random.choice(greetings)

        self._add_to_history("assistant", initial_message)
        return initial_message

    def process_teacher_response(self, teacher_response: str) -> str:
        """Process teacher's response with quality validation"""
        
        # Add teacher response to history
        self._add_to_history("teacher", teacher_response)

        # Extract context if needed
        if not self.extracted_subject or not self.extracted_grade:
            self._extract_teaching_context(teacher_response)

        # Check response quality
        quality = self._check_response_quality(teacher_response)
        
        # Update conversation state
        if quality.quality_level == "substantive":
            self.conversation_state.substantive_responses_in_phase += 1
        elif quality.quality_level == "weak":
            self.conversation_state.weak_response_count += 1

        try:
            # Route to appropriate phase handler with quality context
            phase_handlers = {
                ConversationPhase.INITIAL_ASSESSMENT: self._handle_initial_assessment,
                ConversationPhase.CONCEPT_EXPLORATION: self._handle_concept_exploration,
                ConversationPhase.PEDAGOGY_DEVELOPMENT: self._handle_pedagogy_development,
                ConversationPhase.CULTURAL_INTEGRATION: self._handle_cultural_integration,
                ConversationPhase.READINESS_CHECK: self._handle_readiness_check,
                ConversationPhase.STUDENT_ROLEPLAY: self._handle_student_roleplay,
                ConversationPhase.PARENT_ROLEPLAY: self._handle_parent_roleplay,
                ConversationPhase.ADMINISTRATOR_ROLEPLAY: self._handle_administrator_roleplay,
                ConversationPhase.FINAL_ASSESSMENT: self._generate_final_assessment
            }

            handler = phase_handlers.get(self.current_phase)
            if handler:
                return handler(teacher_response, quality)
            else:
                return self._handle_error(Exception(f"Unknown phase: {self.current_phase}"))

        except Exception as e:
            return self._handle_error(e)

    def reset_conversation(self):
        """Reset conversation to start fresh"""
        self.conversation_history = []
        self.current_phase = ConversationPhase.INITIAL_ASSESSMENT
        self.conversation_state = ConversationState()
        self.roleplay_scenarios_completed = []
        self.extracted_subject = None
        self.extracted_grade = None
        self.student_name = None

    # ============================================================================
    # PHASE HANDLERS - With Quality-Aware Prompts
    # ============================================================================

    def _handle_initial_assessment(self, response: str, quality: ResponseQuality) -> str:
        """Handle initial assessment with quality-aware prompting"""
        
        if quality.quality_level == "weak":
            prompt = self._get_weak_response_prompt(
                phase="initial_assessment",
                response=response,
                quality=quality
            )
        elif quality.quality_level == "superficial":
            prompt = self._get_superficial_response_prompt(
                phase="initial_assessment",
                response=response,
                quality=quality
            )
        else:
            prompt = self._get_substantive_response_prompt(
                phase="initial_assessment",
                response=response,
                quality=quality
            )

        return self._execute_prompt_and_transition(
            prompt, 
            response, 
            next_phase=ConversationPhase.CONCEPT_EXPLORATION
        )

    def _handle_concept_exploration(self, response: str, quality: ResponseQuality) -> str:
        """Handle concept exploration with quality-aware prompting"""
        
        if quality.quality_level == "weak":
            prompt = self._get_weak_response_prompt(
                phase="concept_exploration",
                response=response,
                quality=quality
            )
        elif quality.quality_level == "superficial":
            prompt = self._get_superficial_response_prompt(
                phase="concept_exploration",
                response=response,
                quality=quality
            )
        else:
            prompt = self._get_substantive_response_prompt(
                phase="concept_exploration",
                response=response,
                quality=quality
            )

        return self._execute_prompt_and_transition(
            prompt,
            response,
            next_phase=ConversationPhase.PEDAGOGY_DEVELOPMENT
        )

    def _handle_pedagogy_development(self, response: str, quality: ResponseQuality) -> str:
        """Handle pedagogy development with quality-aware prompting"""
        
        if quality.quality_level == "weak":
            prompt = self._get_weak_response_prompt(
                phase="pedagogy_development",
                response=response,
                quality=quality
            )
        elif quality.quality_level == "superficial":
            prompt = self._get_superficial_response_prompt(
                phase="pedagogy_development",
                response=response,
                quality=quality
            )
        else:
            prompt = self._get_substantive_response_prompt(
                phase="pedagogy_development",
                response=response,
                quality=quality
            )

        return self._execute_prompt_and_transition(
            prompt,
            response,
            next_phase=ConversationPhase.CULTURAL_INTEGRATION
        )

    def _handle_cultural_integration(self, response: str, quality: ResponseQuality) -> str:
        """Handle cultural integration with quality-aware prompting"""
        
        cultural_elements = CULTURAL_CONTEXTS.get(
            self.teacher_profile.cultural_context or CulturalContext.NIGERIAN,
            CULTURAL_CONTEXTS[CulturalContext.NIGERIAN]
        )

        selected_proverbs = random.sample(
            cultural_elements.proverbs,
            min(2, len(cultural_elements.proverbs))
        )
        selected_examples = random.sample(
            cultural_elements.historical_examples,
            min(2, len(cultural_elements.historical_examples))
        )

        if quality.quality_level == "weak":
            prompt = self._get_weak_response_prompt(
                phase="cultural_integration",
                response=response,
                quality=quality,
                cultural_context=cultural_elements.regional_context,
                proverbs=", ".join(selected_proverbs),
                historical_examples=", ".join(selected_examples)
            )
        elif quality.quality_level == "superficial":
            prompt = self._get_superficial_response_prompt(
                phase="cultural_integration",
                response=response,
                quality=quality,
                cultural_context=cultural_elements.regional_context,
                proverbs=", ".join(selected_proverbs),
                historical_examples=", ".join(selected_examples)
            )
        else:
            prompt = self._get_substantive_response_prompt(
                phase="cultural_integration",
                response=response,
                quality=quality,
                cultural_context=cultural_elements.regional_context,
                proverbs=", ".join(selected_proverbs),
                historical_examples=", ".join(selected_examples)
            )

        return self._execute_prompt_and_transition(
            prompt,
            response,
            next_phase=ConversationPhase.READINESS_CHECK
        )

    def _execute_prompt_and_transition(
        self, 
        prompt: ChatPromptTemplate, 
        response: str,
        next_phase: ConversationPhase
    ) -> str:
        """Execute prompt and handle phase transition logic"""
        
        chain = prompt | self.llm
        result = chain.invoke({
            "response": response,
            "subject": self.extracted_subject or "the subject",
            "grade": self.extracted_grade or "your students",
            "last_question": self.conversation_state.last_question_asked or "the previous question",
            "conversation_history": self._format_history(5)
        })

        assistant_message = result.content.strip()
        self._add_to_history("assistant", assistant_message)
        
        self.conversation_state.questions_in_current_phase += 1

        # Check for phase transition
        if self._should_transition_phase():
            self.current_phase = next_phase
            self.conversation_state.questions_in_current_phase = 0
            self.conversation_state.substantive_responses_in_phase = 0
            self.conversation_state.phase_start_index = len(self.conversation_history)

        return assistant_message

    # ============================================================================
    # PROMPT TEMPLATES - Quality-Specific
    # ============================================================================

    def _get_weak_response_prompt(self, phase: str, response: str, quality: ResponseQuality, **kwargs) -> ChatPromptTemplate:
        """Generate prompt for weak responses"""
        
        phase_specific_context = {
            "initial_assessment": "understanding their lesson preparation",
            "concept_exploration": "understanding core concepts in {subject}",
            "pedagogy_development": "developing specific teaching strategies",
            "cultural_integration": "connecting {subject} to Nigerian contexts"
        }
        
        context = phase_specific_context.get(phase, "this aspect of teaching")
        
        return ChatPromptTemplate.from_template(
            f"""You are an AI teaching coach. The teacher gave an inadequate response.

PREVIOUS QUESTION: {{last_question}}

TEACHER'S RESPONSE: "{{response}}"

ISSUE IDENTIFIED: This response is insufficient because it shows uncertainty or lack of preparation regarding {context}.

Your task:
1. Express concern directly but supportively
2. Provide brief scaffolding or an example to guide them
3. Ask a MORE SPECIFIC, concrete question that helps them engage
   - Break down the question into smaller parts
   - Ask about something they CAN answer
   - Give them a framework to work with

DO NOT:
- Accept this response as adequate
- Move on without addressing the gap
- Be overly critical - stay supportive but firm

Your conversational response (natural tone, no bullet points):"""
        )

    def _get_superficial_response_prompt(self, phase: str, response: str, quality: ResponseQuality, **kwargs) -> ChatPromptTemplate:
        """Generate prompt for superficial responses"""
        
        phase_probes = {
            "initial_assessment": "What SPECIFICALLY will you do? Walk me through the first 5 minutes.",
            "concept_exploration": "Why does this matter for {grade} students? What if they ask 'why do we need to learn this?'",
            "pedagogy_development": "How will you know if students actually understand, not just seem to follow along?",
            "cultural_integration": "What SPECIFIC Nigerian examples, names, or contexts will you use?"
        }
        
        probe = phase_probes.get(phase, "Can you give me more specific details?")
        
        return ChatPromptTemplate.from_template(
            f"""You are an AI teaching coach. The teacher gave a surface-level response.

PREVIOUS QUESTION: {{last_question}}

TEACHER'S RESPONSE: "{{response}}"

ANALYSIS: The teacher touched on the topic but lacks depth and specifics.

Your task:
1. Push for depth with ONE of these approaches:
   - Ask "WHY"
   - Demand specifics
   - Challenge with scenarios

2. Do not let them stay at surface level. Require concrete details.

Your conversational response (challenging but supportive):"""
        )

    def _get_substantive_response_prompt(self, phase: str, response: str, quality: ResponseQuality, **kwargs) -> ChatPromptTemplate:
        """Generate prompt for substantive responses"""
        
        phase_extensions = {
            "initial_assessment": "Now let's dig deeper into the core concepts. What's the most challenging part of {subject} for {grade} students to grasp?",
            "concept_exploration": "That's solid understanding. What common misconceptions do {grade} students typically have about this?",
            "pedagogy_development": "Good thinking. How will you differentiate this for students who are struggling versus those who need extension?",
            "cultural_integration": "I like how you're thinking about local context. How will this help Nigerian students see themselves in {subject}?"
        }
        
        extension = phase_extensions.get(phase, "How will you implement this in practice?")
        
        return ChatPromptTemplate.from_template(
            f"""You are an AI teaching coach. The teacher gave a thoughtful, substantive response.

PREVIOUS QUESTION: {{last_question}}

TEACHER'S RESPONSE: "{{response}}"

ANALYSIS: This shows good engagement and understanding.

Your task:
1. Acknowledge SPECIFIC points they made (be genuine, not generic)

2. Extend their thinking with ONE focused follow-up: "{extension}"

3. Keep the momentum going - they're ready for deeper exploration

Your conversational response (affirming and extending):"""
        )

    # ============================================================================
    # ROLEPLAY HANDLERS - With Quality Checks
    # ============================================================================

    def _handle_readiness_check(self, response: str, quality: ResponseQuality) -> str:
        """Transition to roleplay scenarios"""
        cultural_elements = CULTURAL_CONTEXTS.get(
            self.teacher_profile.cultural_context or CulturalContext.NIGERIAN,
            CULTURAL_CONTEXTS[CulturalContext.NIGERIAN]
        )

        self.student_name = random.choice(cultural_elements.local_names)

        scenario_prompt = ChatPromptTemplate.from_template(
            """Generate a challenging student question for a roleplay scenario.

Context: Teaching {subject} to {grade} students. Student name: {student_name}.

Create a realistic, challenging question that tests the teacher's ability to:
- Handle unexpected questions
- Explain concepts clearly under pressure
- Stay patient and supportive

Generate ONLY the student's question (no quotes, no extra text):"""
        )

        challenge_chain = scenario_prompt | self.llm
        challenge_result = challenge_chain.invoke({
            "student_name": self.student_name,
            "grade": self.extracted_grade or "students",
            "subject": self.extracted_subject or "the subject"
        })

        challenge_text = challenge_result.content.strip()

        message = f"""
I'm {self.student_name}, a student in your {self.extracted_grade} class. During your lesson on {self.extracted_subject}, I raise my hand and say:

"{challenge_text}"

How do you respond?
"""

        self._add_to_history("assistant", message)
        self.current_phase = ConversationPhase.STUDENT_ROLEPLAY
        return message

    def _handle_student_roleplay(self, response: str, quality: ResponseQuality) -> str:
        """Handle student roleplay with quality validation"""
        
        student_roleplay_count = sum(
            1 for msg in self.conversation_history
            if msg.get("role") == "assistant" and 
            "I'm" in msg["content"] and 
            self.student_name in msg["content"]
        )

        # If response is weak, stay in roleplay longer
        should_transition = student_roleplay_count >= 1 and quality.quality_level == "substantive"

        if should_transition:
            self.current_phase = ConversationPhase.PARENT_ROLEPLAY
            self.roleplay_scenarios_completed.append("student")
            return self._generate_parent_scenario()
        else:
            # Generate student follow-up based on quality
            if quality.quality_level == "weak":
                prompt = ChatPromptTemplate.from_template(
                    """You are {student_name}, a {grade} student. The teacher gave a vague response: "{teacher_response}"

You're confused because they didn't really answer your question clearly.

Express confusion (stay respectful): "I still don't understand..." or "But what about..."

Your response as {student_name} (1-2 sentences):"""
                )
            else:
                prompt = ChatPromptTemplate.from_template(
                    """You are {student_name}, a {grade} student learning {subject}.

Teacher's response: "{teacher_response}"

Ask a thoughtful follow-up question that tests their deeper understanding.

Your response as {student_name} (1-2 sentences):"""
                )

            chain = prompt | self.llm
            result = chain.invoke({
                "student_name": self.student_name,
                "grade": self.extracted_grade or "student",
                "subject": self.extracted_subject or "this subject",
                "teacher_response": response
            })

            assistant_message = result.content.strip()
            self._add_to_history("assistant", assistant_message)
            return assistant_message

    def _generate_parent_scenario(self) -> str:
        """Generate parent roleplay scenario"""
        cultural_elements = CULTURAL_CONTEXTS.get(
            self.teacher_profile.cultural_context or CulturalContext.NIGERIAN,
            CULTURAL_CONTEXTS[CulturalContext.NIGERIAN]
        )

        parent_title = random.choice(["Mr.", "Mrs.", "Dr.", "Prof."])
        parent_name = f"{parent_title} {random.choice(cultural_elements.local_names)}"
        student_name = random.choice(cultural_elements.local_names)

        scenario_prompt = ChatPromptTemplate.from_template(
            """Generate a concerned parent's question about their child's learning.

Context: Parent {parent_name}, child {student_name}, subject {subject}, grade {grade}.

Create a realistic concern about difficulty, relevance, or homework.

Generate ONLY the parent's statement (no quotes):"""
        )

        concern_chain = scenario_prompt | self.llm
        concern_result = concern_chain.invoke({
            "parent_name": parent_name,
            "student_name": student_name,
            "grade": self.extracted_grade or "students",
            "subject": self.extracted_subject or "the subject"
        })

        parent_concern_quote = concern_result.content.strip()

        message = f"""
I'm {parent_name}, {student_name}'s parent. I've requested a meeting after school.

"Good afternoon, Teacher. {parent_concern_quote} What are you going to do about this?"

How do you respond?
"""

        self._add_to_history("assistant", message)
        return message

    def _handle_parent_roleplay(self, response: str, quality: ResponseQuality) -> str:
        """Handle parent roleplay with quality validation"""
        
        parent_roleplay_count = sum(
            1 for msg in self.conversation_history
            if msg.get("role") == "assistant" and "parent" in msg["content"].lower()
        )

        should_transition = parent_roleplay_count >= 1 and quality.quality_level == "substantive"

        if should_transition:
            self.current_phase = ConversationPhase.ADMINISTRATOR_ROLEPLAY
            self.roleplay_scenarios_completed.append("parent")
            return self._generate_administrator_scenario()
        else:
            if quality.quality_level == "weak":
                prompt = ChatPromptTemplate.from_template(
                    """You are a concerned parent. The teacher's response was vague: "{teacher_response}"

You're not satisfied. Push back professionally: "That doesn't really answer my question..."

Your response (1-2 sentences):"""
                )
            else:
                prompt = ChatPromptTemplate.from_template(
                    """You are a concerned parent discussing {subject} for {grade}.

Teacher's response: "{teacher_response}"

Ask a thoughtful follow-up question.

Your response (1-2 sentences):"""
                )

            chain = prompt | self.llm
            result = chain.invoke({
                "grade": self.extracted_grade or "the class",
                "subject": self.extracted_subject or "the subject",
                "teacher_response": response
            })

            assistant_message = result.content.strip()
            self._add_to_history("assistant", assistant_message)
            return assistant_message

    def _generate_administrator_scenario(self) -> str:
        """Generate administrator roleplay scenario"""
        admin_titles = ["Principal", "Vice Principal", "Head of Department", "Academic Coordinator"]
        admin_names = ["Okafor", "Adeyemi", "Ibrahim", "Nwosu", "Bello", "Chukwuma"]

        admin_title = random.choice(admin_titles)
        admin_prefix = random.choice(["Mrs.", "Mr.", "Dr.", "Prof."])
        admin_name = f"{admin_prefix} {random.choice(admin_names)}"

        scenario_prompt = ChatPromptTemplate.from_template(
            """Generate an administrator's evaluative question about teaching.

Context: {admin_title} reviewing {subject} lesson for {grade}.

Create a professional concern about standards, pedagogy, or management.

Generate ONLY the administrator's statement (no quotes):"""
        )

        concern_chain = scenario_prompt | self.llm
        concern_result = concern_chain.invoke({
            "admin_title": admin_title,
            "grade": self.extracted_grade or "students",
            "subject": self.extracted_subject or "the subject"
        })

        administrator_concern_quote = concern_result.content.strip()

        message = f"""
I'm {admin_name}, the {admin_title}. I've called you to my office.

"{administrator_concern_quote}"

How do you respond?
"""

        self._add_to_history("assistant", message)
        return message

    def _handle_administrator_roleplay(self, response: str, quality: ResponseQuality) -> str:
        """Handle administrator roleplay with quality validation"""
        
        admin_roleplay_count = sum(
            1 for msg in self.conversation_history
            if msg.get("role") == "assistant" and 
            any(title in msg["content"] for title in ["Principal", "Vice Principal", "Head of Department", "Academic Coordinator"])
        )

        should_transition = admin_roleplay_count >= 1 and quality.quality_level == "substantive"

        if should_transition:
            self.current_phase = ConversationPhase.FINAL_ASSESSMENT
            self.roleplay_scenarios_completed.append("administrator")
            return self._generate_final_assessment(response)
        else:
            if quality.quality_level == "weak":
                prompt = ChatPromptTemplate.from_template(
                    """You are a school administrator. The teacher's response was inadequate: "{teacher_response}"

Express professional concern: "I need more specifics about your approach..."

Your response (1-2 sentences):"""
                )
            else:
                prompt = ChatPromptTemplate.from_template(
                    """You are a school administrator evaluating teaching of {subject} for {grade}.

Teacher's response: "{teacher_response}"

Ask a probing follow-up question about their approach.

Your response (1-2 sentences):"""
                )

            chain = prompt | self.llm
            result = chain.invoke({
                "subject": self.extracted_subject or "the subject",
                "grade": self.extracted_grade or "the class",
                "teacher_response": response
            })

            assistant_message = result.content.strip()
            self._add_to_history("assistant", assistant_message)
            return assistant_message

    # ============================================================================
    # FINAL ASSESSMENT
    # ============================================================================

    def _generate_final_assessment(self, response: str = "") -> str:
        """Generate comprehensive assessment with content analysis"""
        
        # Generate detailed content analysis
        content_analysis = self._generate_content_analysis()

        # Generate overall readiness assessment
        parser = PydanticOutputParser(pydantic_object=ReadinessAssessment)

        prompt = PromptTemplate(
            template="""You are an expert pedagogical evaluator. Review this teacher's preparation for teaching {subject} to {grade} students.

FULL CONVERSATION:
{conversation_history}

Evaluate across these dimensions (1-10 scale):
1. Content Mastery - Deep understanding of {subject}
2. Pedagogical Preparedness - Teaching strategies and methods for {grade} level
3. Cultural Relevance - Integration of culturally relevant pedagogy
4. Student Interaction - Performance in student roleplay
5. Parent Communication - Performance in parent roleplay
6. Administrator Interaction - Performance in administrator roleplay

Provide:
- Scores for each dimension
- Overall readiness level (Not Ready/Needs Improvement/Ready/Highly Ready)
- 3-5 specific strengths demonstrated
- 2-4 areas for improvement
- 3-5 actionable recommendations

{format_instructions}

Assessment:""",
            input_variables=["subject", "grade", "conversation_history"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            result = chain.run(
                subject=self.extracted_subject or "this subject",
                grade=self.extracted_grade or "the specified grade level",
                conversation_history=self._format_history()
            )

            assessment = parser.parse(result)
            formatted_assessment = self._format_complete_assessment(assessment, content_analysis)

            self._add_to_history("assistant", formatted_assessment)
            return formatted_assessment

        except Exception:
            return self._generate_fallback_assessment()

    def _generate_content_analysis(self) -> str:
        """Generate detailed content analysis section"""
        
        prompt = ChatPromptTemplate.from_template(
            """You are an expert subject matter specialist analyzing a teacher's content knowledge for teaching {subject} to {grade} students.

FULL CONVERSATION:
{conversation_history}

Analyze the teacher's content understanding and provide:

1. CONTENT STRENGTHS (2-4 points):
- What concepts do they understand well?
- What explanations were accurate and clear?

2. CONTENT GAPS (2-4 points):
- What important concepts are missing from their lesson plan?
- What misconceptions did they demonstrate?
- What foundational knowledge seems weak?

3. WHAT TO ADD (3-5 specific recommendations):
- Key concepts they should include
- Important examples or applications they should incorporate
- Essential prerequisite knowledge to review

4. WHAT TO REMOVE/CHANGE (2-4 specific recommendations):
- Incorrect information or misconceptions to correct
- Overly complex explanations to simplify
- Unnecessary tangents to eliminate
- Better alternatives to current approaches

5. CONTENT DEPTH ASSESSMENT:
- Is their understanding sufficient for {grade} level?
- Are they teaching at the right depth and complexity?
- What advanced connections could they make?

Provide specific, actionable feedback based on the conversation. Be direct and constructive.

Your detailed content analysis:"""
        )

        chain = prompt | self.llm
        result = chain.invoke({
            "subject": self.extracted_subject or "this subject",
            "grade": self.extracted_grade or "the specified grade level",
            "conversation_history": self._format_history()
        })

        return result.content.strip()

    def _format_complete_assessment(self, assessment: ReadinessAssessment, content_analysis: str) -> str:
        """Format the complete assessment with content analysis"""
        
        formatted = f"""
{'='*80}
COMPREHENSIVE TEACHER READINESS ASSESSMENT
Subject: {self.extracted_subject or 'Your Lesson'}
Grade: {self.extracted_grade or 'N/A'}
{'='*80}

SECTION 1: CONTENT ANALYSIS
{'-'*80}

{content_analysis}

{'='*80}

SECTION 2: OVERALL READINESS EVALUATION
{'-'*80}

OVERALL READINESS: {assessment.overall_readiness}

DETAILED SCORES:
├─ Content Mastery: {assessment.content_mastery}/10
├─ Pedagogical Preparedness: {assessment.pedagogical_preparedness}/10
├─ Cultural Relevance: {assessment.cultural_relevance}/10
├─ Student Interaction: {assessment.student_interaction_readiness}/10
├─ Parent Communication: {assessment.parent_communication_readiness}/10
└─ Administrator Interaction: {assessment.administrator_readiness}/10

STRENGTHS DEMONSTRATED:
"""
        for i, strength in enumerate(assessment.strengths, 1):
            formatted += f"  {i}. {strength}\n"

        formatted += "\nAREAS FOR IMPROVEMENT:\n"
        for i, area in enumerate(assessment.areas_for_improvement, 1):
            formatted += f"  {i}. {area}\n"

        formatted += "\nACTIONABLE RECOMMENDATIONS:\n"
        for i, rec in enumerate(assessment.recommendations, 1):
            formatted += f"  {i}. {rec}\n"

        formatted += f"""
{'='*80}

FINAL NOTES:

Review both the Content Analysis and Overall Readiness sections carefully.
Focus especially on the "What to Add" and "What to Remove/Change" sections
to refine your lesson before teaching.

Good luck with your lesson!
{'='*80}
"""
        return formatted

    def _generate_fallback_assessment(self) -> str:
        """Generate fallback assessment if main generation fails"""
        
        try:
            content_analysis = self._generate_content_analysis()
            
            return f"""
{'='*80}
COMPREHENSIVE TEACHER READINESS ASSESSMENT
Subject: {self.extracted_subject or 'Your Lesson'}
Grade: {self.extracted_grade or 'N/A'}
{'='*80}

SECTION 1: CONTENT ANALYSIS
{'-'*80}

{content_analysis}

{'='*80}

SECTION 2: OVERALL READINESS EVALUATION
{'-'*80}

Based on our conversation, you've demonstrated preparation for teaching
{self.extracted_subject or 'this subject'} to {self.extracted_grade or 'your students'}.

NEXT STEPS:
• Review the Content Analysis section above carefully
• Address the gaps and implement the recommended additions
• Remove or change any misconceptions identified
• Practice your explanations to ensure clarity

Continue refining your approach based on this feedback!
{'='*80}
"""
        except Exception:
            return f"""
{'='*80}
TEACHER READINESS ASSESSMENT
Subject: {self.extracted_subject or 'Your Lesson'}
Grade: {self.extracted_grade or 'N/A'}
{'='*80}

Based on our conversation, you've demonstrated preparation for teaching
{self.extracted_subject or 'this subject'} to {self.extracted_grade or 'your students'}.

Continue refining your approach and you'll be well-prepared for your lesson!
{'='*80}
"""

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def _extract_teaching_context(self, response: str):
        """Extract topic and grade level from teacher's response"""
        try:
            prompt = ChatPromptTemplate.from_template(
                """Extract the teaching topic and grade level from this teacher's response.

Teacher's Response: {response}

Return ONLY a JSON object with this exact format:
{{"topic": "extracted topic or null", "grade": "extracted grade level or null"}}

If information is not mentioned, use null. Be concise.

JSON:"""
            )

            chain = prompt | self.llm
            result = chain.invoke({"response": response})

            content = result.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            extracted = json.loads(content)

            if not self.extracted_subject:
                self.extracted_subject = extracted.get("topic")
            if not self.extracted_grade:
                self.extracted_grade = extracted.get("grade")

        except Exception:
            pass

    def _handle_error(self, error: Exception) -> str:
        """Handle errors gracefully"""
        error_message = """I apologize, but I encountered an issue processing your response.

Let's continue our conversation. Could you please rephrase your last response or provide more details?"""

        self._add_to_history("assistant", error_message)
        return error_message

    # ============================================================================
    # PUBLIC STATE ACCESS METHODS
    # ============================================================================

    def get_conversation_state(self) -> Dict:
        """Get current conversation state for debugging/monitoring"""
        return {
            "current_phase": self.current_phase.value,
            "questions_in_phase": self.conversation_state.questions_in_current_phase,
            "substantive_responses": self.conversation_state.substantive_responses_in_phase,
            "weak_response_count": self.conversation_state.weak_response_count,
            "extracted_subject": self.extracted_subject,
            "extracted_grade": self.extracted_grade,
            "total_messages": len(self.conversation_history),
            "scenarios_completed": self.roleplay_scenarios_completed,
            "last_question": self.conversation_state.last_question_asked
        }

    def is_assessment_complete(self) -> bool:
        """Check if the conversation has reached final assessment"""
        return self.current_phase == ConversationPhase.FINAL_ASSESSMENT

    def get_final_report_data(self) -> Optional[Dict]:
        """Get structured report data for completed assessments"""
        if not self.is_assessment_complete():
            return None

        for msg in reversed(self.conversation_history):
            if msg["role"] == "assistant" and "TEACHER READINESS ASSESSMENT" in msg["content"]:
                return {
                    "topic": self.extracted_subject,
                    "grade": self.extracted_grade,
                    "assessment_text": msg["content"],
                    "conversation_length": len(self.conversation_history),
                    "weak_responses": self.conversation_state.weak_response_count
                }

        return None


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_custom_assistant(
    api_key: str,
    cultural_context: CulturalContext = CulturalContext.NIGERIAN,
    min_questions: int = 5,
    max_questions: int = 10,
    min_substantive: int = 7,
    temperature: float = 0.7
) -> ConversationalTeacherAssistant:
    """
    Helper function to create a customized assistant

    Args:
        api_key: Groq API key
        cultural_context: Cultural context for the assistant
        min_questions: Minimum questions per phase
        max_questions: Maximum questions per phase
        min_substantive: Minimum substantive responses required before transition
        temperature: LLM temperature (0.0-1.0)

    Returns:
        Configured ConversationalTeacherAssistant instance
    """
    config = AssistantConfig(
        min_questions_per_phase=min_questions,
        max_questions_per_phase=max_questions,
        min_substantive_responses=min_substantive,
        enable_dynamic_scenarios=True,
        temperature=temperature,
        cultural_context=cultural_context
    )

    return ConversationalTeacherAssistant(api_key, config)