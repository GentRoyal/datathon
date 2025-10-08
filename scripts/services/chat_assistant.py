import random
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser

from scripts.services.templates import greeting_templates
from scripts.core.model_schema import (
    ConversationPhase,
    CulturalContext,
    CulturalElements,
    TeacherProfile,
    ReadinessAssessment
)


CULTURAL_CONTEXTS = {
    CulturalContext.NIGERIAN: CulturalElements(
        proverbs=[
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
class AssistantConfig:
    """Configuration for customizing assistant behavior"""
    min_questions_per_phase: int = 2
    max_questions_per_phase: int = 2
    enable_dynamic_scenarios: bool = True
    temperature: float = 0.7
    model_name: str = "llama-3.1-8b-instant"
    cultural_context: CulturalContext = CulturalContext.NIGERIAN


# ============================================================================
# Conversational AI Teacher's Assistant
# ============================================================================

class ConversationalTeacherAssistant:
    """
    Interactive AI assistant that guides teachers through lesson preparation
    via conversational dialogue, progressing through multiple phases and
    culminating in roleplay scenarios and comprehensive assessment.
    """

    def __init__(
        self,
        groq_api_key: str,
        config: Optional[AssistantConfig] = None
    ):
        """Initialize the conversational AI Teacher's Assistant"""
        self.config = config or AssistantConfig()

        self.llm = ChatGroq(
            api_key = groq_api_key,
            model=self.config.model_name,
            temperature=self.config.temperature
        )

        self.conversation_history: List[Dict[str, str]] = []
        self.teacher_profile = TeacherProfile()
        self.teacher_profile.cultural_context = self.config.cultural_context

        self.current_phase = ConversationPhase.INITIAL_ASSESSMENT
        self.questions_asked = 0
        self.roleplay_scenarios_completed = []

        # Context extracted from conversation OR provided upfront
        self.extracted_subject: Optional[str] = None
        self.extracted_grade: Optional[str] = None

        # Roleplay character names
        self.student_name: Optional[str] = None

    # ========================================================================
    # Public Interface Methods
    # ========================================================================

    def start_conversation(self, grade: Optional[str] = None, subject: Optional[str] = None) -> str:
        """
        Start the conversation with a dynamic, personalized greeting. If grade and subject are provided, they are used as context.
        """
        if grade:
            self.extracted_grade = grade

        if subject:
            self.extracted_subject = subject

        
        greetings = greeting_templates(self.extracted_subject, self.extracted_grade)
                
        initial_message = random.choice(greetings)

        self.conversation_history.append({
            "role": "assistant",
            "content": initial_message
        })

        return initial_message

    def process_teacher_response(self, teacher_response: str) -> str:
        """Process teacher's response and generate next question"""

        # Add teacher response to history
        self.conversation_history.append({
            "role": "teacher",
            "content": teacher_response
        })

        # Extract additional context if not fully specified
        if not self.extracted_subject or not self.extracted_grade:
            self._extract_teaching_context(teacher_response)

        try:
            # Route to appropriate phase handler
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
                return handler(teacher_response)
            else:
                return self._handle_error(Exception(f"Unknown phase: {self.current_phase}"))

        except Exception as e:
            return self._handle_error(e)

    def get_conversation_state(self) -> Dict:
        """Get current conversation state for debugging/monitoring"""
        return {
            "current_phase": self.current_phase.value,
            "questions_asked": self.questions_asked,
            "extracted_subject": self.extracted_subject,
            "extracted_grade": self.extracted_grade,
            "total_messages": len(self.conversation_history),
            "scenarios_completed": self.roleplay_scenarios_completed
        }

    def is_assessment_complete(self) -> bool:
        """Check if the conversation has reached final assessment"""
        return self.current_phase == ConversationPhase.FINAL_ASSESSMENT

    def get_final_report_data(self) -> Optional[Dict]:
        """Get structured report data for completed assessments"""
        if not self.is_assessment_complete():
            return None

        # Find the final assessment message in history
        for msg in reversed(self.conversation_history):
            if msg["role"] == "assistant" and "TEACHER READINESS ASSESSMENT" in msg["content"]:
                return {
                    "topic": self.extracted_subject,
                    "grade": self.extracted_grade,
                    "assessment_text": msg["content"],
                    "conversation_length": len(self.conversation_history)
                }

        return None

    def reset_conversation(self):
        """Reset conversation to start fresh"""
        self.conversation_history = []
        self.current_phase = ConversationPhase.INITIAL_ASSESSMENT
        self.questions_asked = 0
        self.roleplay_scenarios_completed = []
        self.extracted_subject = None
        self.extracted_grade = None
        self.student_name = None

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

            # Parse JSON response
            content = result.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            extracted = json.loads(content)

            # Only update if not already set
            if not self.extracted_subject:
                self.extracted_subject = extracted.get("topic")
            if not self.extracted_grade:
                self.extracted_grade = extracted.get("grade")

        except Exception:
            # If extraction fails, continue without it
            pass

    def _format_history(self) -> str:
        """Format full conversation history"""
        formatted = []
        for entry in self.conversation_history:
            role = entry["role"].upper()
            content = entry["content"]
            formatted.append(f"{role}: {content}\n")
        return "\n".join(formatted)

    def _format_recent_history(self, num_messages: int = 5) -> str:
        """Format recent conversation history"""
        recent = self.conversation_history[-num_messages:]
        formatted = []
        for entry in recent:
            role = entry["role"].upper()
            content = entry["content"]
            formatted.append(f"{role}: {content}\n")
        return "\n".join(formatted)

    def _handle_error(self, error: Exception) -> str:
        """Handle errors gracefully"""
        error_message = """I apologize, but I encountered an issue processing your response.

        Let's continue our conversation. Could you please rephrase your last response or provide more details?"""

        self.conversation_history.append({
            "role": "assistant",
            "content": error_message
        })

        return error_message

    # ========================================================================
    # Phase Handlers
    # ========================================================================

    def _handle_initial_assessment(self, response: str) -> str:
        """Extract basic information and move to concept exploration"""

        prompt = ChatPromptTemplate.from_template(
            """You are an AI teaching coach having a conversation with a teacher preparing to teach {subject} to {grade} students.

            Teacher's response: {response}

            Your task: Write ONLY your next conversational response to the teacher.
            - Briefly acknowledge what they shared (1 sentence)
            - Ask ONE focused follow-up question about their core concepts understanding for {subject}
            - Keep it natural and conversational
            - DO NOT include any internal reasoning, analysis, or task descriptions

            Your response:"""
        )

        chain = prompt | self.llm
        result = chain.invoke({
            "response": response,
            "subject": self.extracted_subject or "the subject",
            "grade": self.extracted_grade or "your students"
        })

        assistant_message = result.content.strip()
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        self.questions_asked += 1

        if self.questions_asked >= self.config.min_questions_per_phase:
            self.current_phase = ConversationPhase.CONCEPT_EXPLORATION
            self.questions_asked = 0

        return assistant_message

    def _handle_concept_exploration(self, response: str) -> str:
        """Deep dive into content understanding"""

        should_transition = self.questions_asked >= self.config.max_questions_per_phase - 1

        if should_transition:
            prompt = ChatPromptTemplate.from_template(
                """You are an AI teaching coach having a conversation with a teacher about {subject} for {grade} students.

            Teacher's latest response: {response}

            Recent conversation:
            {conversation_history}

            Your task: Write ONLY your next conversational response.
            - Acknowledge their understanding briefly
            - Transition smoothly by saying something like "Great! Now let's explore how you'll teach this..."
            - Ask your first question about their teaching approach
            - Keep it natural and conversational
            - DO NOT include any internal reasoning or task descriptions

            Your response:"""
            )
        else:
            prompt = ChatPromptTemplate.from_template(
                """You are an AI teaching coach having a conversation with a teacher about {subject} for {grade} students.

            Teacher's latest response: {response}

            Recent conversation:
            {conversation_history}

            Your task: Write ONLY your next conversational response.
            - Ask ONE probing follow-up question about:
            * Core principles they'll teach in {subject}
            * Common misconceptions {grade} students might have
            * Real-world applications they'll use
            * Connections to related concepts
            - Keep it natural and conversational
            - DO NOT include any internal reasoning or task descriptions

            Your response:"""
            )

        chain = prompt | self.llm
        result = chain.invoke({
            "response": response,
            "subject": self.extracted_subject or "this subject",
            "grade": self.extracted_grade or "your grade level",
            "conversation_history": self._format_recent_history(5)
        })

        assistant_message = result.content.strip()
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        self.questions_asked += 1

        if should_transition:
            self.current_phase = ConversationPhase.PEDAGOGY_DEVELOPMENT
            self.questions_asked = 0

        return assistant_message

    def _handle_pedagogy_development(self, response: str) -> str:
        """Explore teaching strategies and methods"""

        should_transition = self.questions_asked >= self.config.max_questions_per_phase - 1

        if should_transition:
            prompt = ChatPromptTemplate.from_template(
                """You are an AI teaching coach discussing pedagogy with a teacher.

            Teacher's latest response: {response}

            Subject: {subject}
            Grade: {grade}

            Your task: Write ONLY your next conversational response.
            - Acknowledge their teaching approach
            - Transition to cultural relevance by saying something like "Now let's think about how to make this culturally relevant for your {grade} students in Nigeria..."
            - Ask how they'll connect {subject} to local Nigerian context
            - Keep it natural and conversational

            Your response:"""
            )
        else:
            prompt = ChatPromptTemplate.from_template(
                """You are an AI teaching coach discussing pedagogy with a teacher.

            Teacher's latest response: {response}

            Your task: Write ONLY your next conversational response.
            - Ask ONE focused question about:
            * Their student engagement strategies for {grade} learners
            * Hands-on activities they'll use for {subject}
            * How they'll differentiate for different learners
            * Their assessment approach
            - Challenge them to think deeper with "why" or "how" questions
            - Keep it natural and conversational

            Your response:"""
            )

        chain = prompt | self.llm
        result = chain.invoke({
            "response": response,
            "subject": self.extracted_subject or "this subject",
            "grade": self.extracted_grade or "students"
        })

        assistant_message = result.content.strip()
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        self.questions_asked += 1

        if should_transition:
            self.current_phase = ConversationPhase.CULTURAL_INTEGRATION
            self.questions_asked = 0

        return assistant_message

    def _handle_cultural_integration(self, response: str) -> str:
        """Explore culturally relevant pedagogy"""

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

        should_transition = self.questions_asked >= 2

        if should_transition:
            prompt = ChatPromptTemplate.from_template(
            """You are an AI teaching coach having a conversation with a teacher about cultural relevance in their lesson.

            Teacher's latest response: {response}

            Your task: Write ONLY your next conversational response. Follow these guidelines:

            - Acknowledge and affirm the teacher's ideas for integrating cultural relevance.
            - Transition naturally to the next step, indicating that you will test their readiness through roleplay scenarios.
            - Keep the tone natural, encouraging, and conversational.
            - Generate the content of each scenario dynamically—do not reuse any fixed text or example phrases.
            - Begin the first scenario immediately after your transition, creating realistic dialogue for the roleplay.

            Roleplay three realistic scenarios sequentially:  
                1. A student asking a challenging question during the lesson.  
                2. A parent raising a concern or question about the lesson.  
                3. A school administrator asking about teaching strategy, classroom management, or alignment with school standards.  
            - Generate all dialogue dynamically, do not reuse any fixed example text.  
            - Keep the tone natural, encouraging, and conversational.  
            - Begin the first scenario immediately after your transition, creating realistic dialogue for the roleplay.

            How do you respond?"

            Your response:"""
            )
        else:
            prompt = ChatPromptTemplate.from_template(
                """You are an AI teaching coach discussing cultural relevance for teaching {subject} to {grade} students.

                Teacher's latest response: {response}

                Cultural context: {cultural_context}
                Available cultural elements:
                - Proverbs: {proverbs}
                - Historical examples: {historical_examples}

                Your task: Write ONLY your next conversational response.
                - Ask how they'll make {subject} relatable to {grade} Nigerian students
                - Suggest using local examples or cultural elements if they haven't mentioned any
                - Keep it natural and conversational

                Your response:"""
            )

        chain = prompt | self.llm

        if should_transition:
            result = chain.invoke({"response": response})
        else:
            result = chain.invoke({
                "response": response,
                "subject": self.extracted_subject or "this subject",
                "grade": self.extracted_grade or "your",
                "cultural_context": cultural_elements.regional_context,
                "proverbs": ", ".join(selected_proverbs),
                "historical_examples": ", ".join(selected_examples)
            })

        assistant_message = result.content.strip()
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        self.questions_asked += 1

        if should_transition:
            self.current_phase = ConversationPhase.READINESS_CHECK
            self.questions_asked = 0

        return assistant_message

    def _handle_readiness_check(self, response: str) -> str:
        """Transition to roleplay scenarios with dynamic content"""

        cultural_elements = CULTURAL_CONTEXTS.get(
            self.teacher_profile.cultural_context or CulturalContext.NIGERIAN,
            CULTURAL_CONTEXTS[CulturalContext.NIGERIAN]
        )

        # Generate dynamic student scenario
        self.student_name = random.choice(cultural_elements.local_names)

        scenario_prompt = ChatPromptTemplate.from_template(
        """You are a creative AI generator. Your task is to generate the challenging quote for a student roleplay.
        
        Context: The teacher is teaching {subject} to {grade} students. The student's name is {student_name}.
        
        Task: Write a single, realistic, and challenging question or statement from a student during the lesson that tests the teacher's ability to handle classroom dynamics or material relevance.
        
        Generate ONLY the student's quote, without quotation marks, internal reasoning, or any other text.
        
        Student's Challenge:"""
        )
        
        # Run the chain to get the dynamic challenge text
        challenge_chain = scenario_prompt | self.llm
        challenge_result = challenge_chain.invoke({
            "student_name": self.student_name or "the student",
            "grade": self.extracted_grade or "students",
            "subject": self.extracted_subject or "the subject"
        })
        
        challenge_text = challenge_result.content.strip()

        # **NEW MESSAGE CONSTRUCTION**
        message = f"""
    I'm {self.student_name}, a student in your {self.extracted_grade} class. During your lesson on {self.extracted_subject}, I raise my hand and say:

    "{challenge_text}"

    How do you respond?
    """

        self.conversation_history.append({
            "role": "assistant",
            "content": message
        })

        self.current_phase = ConversationPhase.STUDENT_ROLEPLAY
        return message

    def _handle_student_roleplay(self, response: str) -> str:
        """Dynamic student roleplay"""

        # Count exchanges in this roleplay phase
        student_roleplay_count = sum(
            1 for msg in self.conversation_history
            if msg.get("role") == "assistant" and
            "SCENARIO:" not in msg["content"] and
            self.current_phase == ConversationPhase.STUDENT_ROLEPLAY
        )

        should_transition = student_roleplay_count >= 1

        if should_transition:
            # Move to parent roleplay
            self.current_phase = ConversationPhase.PARENT_ROLEPLAY
            self.roleplay_scenarios_completed.append("student")
            return self._generate_parent_scenario()
        else:
            # Continue student roleplay
            prompt = ChatPromptTemplate.from_template(
                """You are roleplaying as {student_name}, a {grade} student learning about {subject}.

                    Teacher's response to you: {teacher_response}

                    Your task: Write ONLY your next response as the student.
                    - Be authentic and challenging but not rude
                    - Ask a follow-up question or express a concern
                    - Keep it brief (1-2 sentences)
                    - Stay in character

                    Your response as {student_name}:"""
            )

            chain = prompt | self.llm
            result = chain.invoke({
                "student_name": self.student_name or "the student",
                "grade": self.extracted_grade or "student",
                "subject": self.extracted_subject or "this subject",
                "teacher_response": response
            })

            assistant_message = result.content.strip()
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            return assistant_message

    def _generate_parent_scenario(self) -> str:
        """Generate dynamic parent scenario"""

        cultural_elements = CULTURAL_CONTEXTS.get(
            self.teacher_profile.cultural_context or CulturalContext.NIGERIAN,
            CULTURAL_CONTEXTS[CulturalContext.NIGERIAN]
        )

        parent_title = random.choice(["Mr.", "Mrs.", "Dr.", "Prof."])
        parent_name = f"{parent_title} {random.choice(cultural_elements.local_names)}"
        student_name = random.choice(cultural_elements.local_names)

        scenario_prompt = ChatPromptTemplate.from_template(
        """You are a creative AI generator. Your task is to generate a realistic parent concern quote.
        
        Context: The teacher is teaching {subject} to {grade} students. The parent is {parent_name}, whose child, {student_name}, is struggling or confused.
        
        Task: Write a single, professional, yet concerned quote from the parent about the lesson's difficulty, relevance, or homework load.
        
        Generate ONLY the parent's quote, without quotation marks, internal reasoning, or any other text.
        
        Parent's Quote:"""
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
    I'm {parent_name}, {student_name}'s parent. I've requested a meeting with you after school.

    "Good afternoon, Teacher. {parent_concern_quote} What are you going to do about this?"

    How do you respond?
    """

        self.conversation_history.append({
            "role": "assistant",
            "content": message
        })

        return message

    def _handle_parent_roleplay(self, response: str) -> str:
        """Dynamic parent roleplay"""

        parent_roleplay_count = sum(
            1 for msg in self.conversation_history
            if msg.get("role") == "assistant" and
            "SCENARIO:" not in msg["content"] and
            self.current_phase == ConversationPhase.PARENT_ROLEPLAY
        )

        should_transition = parent_roleplay_count >= 1

        if should_transition:
            self.current_phase = ConversationPhase.ADMINISTRATOR_ROLEPLAY
            self.roleplay_scenarios_completed.append("parent")
            return self._generate_administrator_scenario()
        else:
            prompt = ChatPromptTemplate.from_template(
                """You are roleplaying as a concerned parent whose child is in {grade} studying {subject}.

            Teacher's response: {teacher_response}

            Your task: Write ONLY your next response as the parent.
            - Be concerned but willing to collaborate
            - Ask a follow-up question or express a worry
            - Keep it brief (1-2 sentences)
            - Stay in character

            Your response:"""
            )

            chain = prompt | self.llm
            result = chain.invoke({
                "grade": self.extracted_grade or "the class",
                "subject": self.extracted_subject or "the subject",
                "teacher_response": response
            })

            assistant_message = result.content.strip()
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            return assistant_message

    def _generate_administrator_scenario(self) -> str:
        """Generate dynamic administrator scenario"""

        admin_titles = ["Principal", "Vice Principal", "Head of Department", "Academic Coordinator"]
        admin_names = ["Okafor", "Adeyemi", "Ibrahim", "Nwosu", "Bello", "Chukwuma"]

        admin_title = random.choice(admin_titles)
        admin_prefix = random.choice(["Mrs.", "Mr.", "Dr.", "Prof."])
        admin_name = f"{admin_prefix} {random.choice(admin_names)}"

        scenario_prompt = ChatPromptTemplate.from_template(
            """You are a creative AI generator. Your task is to generate a professional administrator concern quote.
            
            Context: The administrator is the {admin_title}, focusing on the teacher's lesson on {subject} for {grade}.
            
            Task: Write a single, professional, and evaluative quote from the administrator about alignment with standards, classroom management, or pedagogical choice.
            
            Generate ONLY the administrator's quote, without quotation marks, internal reasoning, or any other text.
            
            Administrator's Quote:"""
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

        self.conversation_history.append({
            "role": "assistant",
            "content": message
        })

        return message

    def _handle_administrator_roleplay(self, response: str) -> str:
        """Dynamic administrator roleplay"""

        admin_roleplay_count = sum(
            1 for msg in self.conversation_history
            if msg.get("role") == "assistant" and
            "SCENARIO" not in msg["content"] and
            self.current_phase == ConversationPhase.ADMINISTRATOR_ROLEPLAY
        )

        should_transition = admin_roleplay_count >= 1

        if should_transition:
            self.current_phase = ConversationPhase.FINAL_ASSESSMENT
            self.roleplay_scenarios_completed.append("administrator")

            final_report_text = self._generate_final_assessment()

            return final_report_text
        else:
            prompt = ChatPromptTemplate.from_template(
                """You are roleplaying as a school administrator evaluating a teacher's approach to teaching {subject} to {grade} students.

                Teacher's response: {teacher_response}

                Your task: Write ONLY your next response as the administrator.
                - Be professional and evaluative
                - Ask a follow-up question or raise a concern
                - Keep it brief (1-2 sentences)
                - Stay in character

                Your response:"""
                            )

            chain = prompt | self.llm
            result = chain.invoke({
                "subject": self.extracted_subject or "the subject",
                "grade": self.extracted_grade or "the class",
                "teacher_response": response
            })

            assistant_message = result.content.strip()
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            return assistant_message


    def _generate_final_assessment(self) -> str:
        """Generate comprehensive assessment with content analysis"""

        # Step 1: Generate detailed content analysis
        content_analysis = self._generate_content_analysis()

        # Step 2: Generate overall readiness assessment
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

            # Format complete assessment with content analysis
            formatted_assessment = self._format_complete_assessment(assessment, content_analysis)

            self.conversation_history.append({
                "role": "assistant",
                "content": formatted_assessment
            })

            return formatted_assessment

        except Exception:
            # Fallback if parsing fails
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
        """Generate a comprehensive fallback assessment if main generation fails"""
        
        # Try to generate at least the content analysis
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
            # Ultimate fallback
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


def create_custom_assistant(
    api_key: str,
    cultural_context: CulturalContext = CulturalContext.NIGERIAN,
    min_questions: int = 1,
    max_questions: int = 1,
    temperature: float = 0.7
) -> ConversationalTeacherAssistant:
    """
    Helper function to create a customized assistant

    Args:
        api_key: Groq API key
        cultural_context: Cultural context for the assistant
        min_questions: Minimum questions per phase
        max_questions: Maximum questions per phase
        temperature: LLM temperature (0.0-1.0)

    Returns:
        Configured ConversationalTeacherAssistant instance
    """
    config = AssistantConfig(
        min_questions_per_phase = min_questions,
        max_questions_per_phase = max_questions,
        enable_dynamic_scenarios = True,
        temperature = temperature,
        cultural_context = cultural_context
    )

    return ConversationalTeacherAssistant(api_key, config)