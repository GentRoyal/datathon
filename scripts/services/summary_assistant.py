from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from typing import List, Dict, Optional
import os


class KnowledgeCompressor:
    """Handles compression of educational material into print-optimized teaching briefs"""
    
    COMPRESSION_PROMPT = """You are an expert educational content compressor specializing in creating low-bandwidth, print-optimized teaching materials for resource-constrained environments.

    TASK: Transform the following educational material into a Single-Page, Print-Optimized Teaching Brief.

    REQUIREMENTS:
    1. Content Completeness: Preserve all key concepts, definitions, and learning objectives
    2. Minimalist Design: Text-only, no images, high-contrast formatting suitable for basic printers
    3. Cultural Relevance: Replace complex vocabulary with local/regional analogies from {region}
    4. Offline-First: Optimize for printing or low-bandwidth transfer
    5. Length: Must fit on ONE printed page (approximately 500-700 words)
    6. Structure: Use clear headings, bullet points, and concise paragraphs

    LOCALIZATION INSTRUCTIONS:
    - Substitute technical jargon with relatable terms familiar to students in {region}
    - Include culturally relevant success stories or analogies where appropriate
    - Maintain academic integrity while making content accessible

    ORIGINAL MATERIAL:
    {content}

    Generate the compressed teaching brief now:"""
        
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-8b-instant"):
        """
        Initialize the Knowledge Compressor
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env variable)
            model: Groq model to use
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided or set as environment variable")
        
        self.llm = ChatGroq(
            model=model,
            temperature=0.3,
            api_key = self.api_key
        )
        
        self.prompt_template = ChatPromptTemplate.from_template(self.COMPRESSION_PROMPT)
    
    def compress(
        self, 
        content: str, 
        region: str = "Nigeria"
    ) -> str:
        """
        Compress educational material into a single-page teaching brief
        
        Args:
            content: Original educational material to compress
            region: Geographic region for cultural context
            
        Returns:
            Compressed teaching brief as string
        """
        chain = self.prompt_template | self.llm
        
        response = chain.invoke({
            "content": content,
            "region": region
        })
        
        print(response.content)
        return response.content


class ContextualChatAssistant:
    """Handles chat interactions with context from teaching briefs"""
    
    CHAT_PROMPT = """You are an expert AI Teaching Support Specialist helping teachers plan lessons and understand material. You have access to the original educational content and a compressed, print-optimized teaching brief.

    CONTEXT - COMPRESSED TEACHING BRIEF:
    {compressed_content}

    ORIGINAL MATERIAL (for reference):
    {original_material}

    CHAT HISTORY:
    {chat_history}

    TEACHER'S QUESTION: {question}

    ADAPTIVE RESPONSE GUIDELINES:
    Analyze the question type and respond appropriately:

    SHORT RESPONSES (2-4 sentences, 50-100 words) for:
    - Simple factual questions ("What is...?", "When should...?", "Which topic...?")
    - Clarifications or confirmations
    - Quick definitions or explanations
    - Status checks or summaries

    MEDIUM RESPONSES (1-2 paragraphs, 100-200 words) for:
    - "How to teach..." questions
    - Explaining concepts or strategies
    - Suggestions or recommendations
    - Comparative questions

    LONG RESPONSES (3-5 paragraphs, structured format) for:
    - Lesson plan requests (include: objectives, activities, materials, timing, assessment)
    - Curriculum design (include: scope, sequence, milestones)
    - Comprehensive guides or frameworks
    - Detailed activity designs
    - Assessment strategies with rubrics

    RESPONSE PRINCIPLES:
    1. Be direct and actionable - start with the answer immediately
    2. Match the depth of your response to what's being asked
    3. For structured requests (lesson plans, curricula), use clear formatting with headings/bullets
    4. Reference the teaching brief naturally when relevant
    5. Don't apologize for length - just deliver what's needed

    EXAMPLES:
    Q: "What's the main topic here?"
    A: "The teaching brief covers entrepreneurship fundamentals, with emphasis on business planning, market analysis, and financial literacy for small enterprises."

    Q: "How should I teach market research?"
    A: "Start with observation exercises at local markets where students identify best-selling products and customer behaviors. Then introduce the 'Understanding Your Customer' section from the brief, having students interview 3-5 community members about their needs. Conclude with a simple market analysis framework where they match products to customer problems."

    Q: "Create a lesson plan for this topic"
    A: [Provides full structured lesson plan with objectives, materials, activities, timing, and assessment]

    Your response:"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-8b-instant"):
        """
        Initialize the Contextual Chat Assistant
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env variable)
            model: Groq model to use
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be provided or set as environment variable")
        
        self.llm = ChatGroq(
            model=model,
            temperature=0.5,
            api_key=self.api_key,
            max_tokens=800  # Allows for longer structured responses when needed
        )
        
        self.prompt_template = ChatPromptTemplate.from_template(self.CHAT_PROMPT)
        self.chat_history: List[HumanMessage | AIMessage] = []
    
    def chat(
        self,
        question: str,
        compressed_content: str,
        original_material: str,
        use_history: bool = True
    ) -> str:
        """
        Chat with AI using teaching brief as context
        
        Args:
            question: Teacher's question
            compressed_content: The compressed teaching brief
            original_material: Original educational material
            use_history: Whether to include chat history in context
            
        Returns:
            AI assistant's response as string
        """
        # Format chat history (last 4 messages = 2 exchanges)
        history_text = ""
        if use_history and self.chat_history:
            history_text = "\n".join([
                f"{'Teacher' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
                for msg in self.chat_history[-4:]
            ])
        
        # Truncate original material to avoid token limits
        truncated_original = original_material[:1500] if len(original_material) > 1500 else original_material
        
        chain = self.prompt_template | self.llm
        
        response = chain.invoke({
            "compressed_content": compressed_content,
            "original_material": truncated_original,
            "chat_history": history_text if history_text else "No previous conversation",
            "question": question
        })
        
        # Store in chat history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response.content))
        
        return response.content
    
    def clear_history(self):
        """Clear the chat history"""
        self.chat_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get chat history in a readable format
        
        Returns:
            List of dicts with 'role' and 'content' keys
        """
        return [
            {
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            }
            for msg in self.chat_history
        ]


# Convenience class to manage both components together
class KnowledgeCompressionSystem:
    """Main system that combines compression and chat functionality"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-8b-instant"):
        """
        Initialize the complete Knowledge Compression System
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env variable)
            model: Groq model to use
        """
        self.compressor = KnowledgeCompressor(api_key=api_key, model=model)
        self.chat_assistant = ContextualChatAssistant(api_key=api_key, model=model)
        
        self.compressed_content: str = ""
        self.original_material: str = ""
        self.region: str = "Nigeria"
    
    def process_material(
        self,
        material: str,
        region: str = "Nigeria",
    ) -> str:
        """
        Process uploaded material and generate compressed teaching brief
        
        Args:
            material: Original educational material
            region: Geographic region
            
        Returns:
            Compressed teaching brief
        """
        self.original_material = material
        self.region = region
        
        self.compressed_content = self.compressor.compress(
            content=material,
            region=region
        )
        
        return self.compressed_content
    
    def ask_question(self, question: str) -> str:
        """
        Ask a question about the material
        
        Args:
            question: Teacher's question
            
        Returns:
            AI assistant's response
        """
        if not self.compressed_content:
            raise ValueError("No material has been processed yet. Call process_material() first.")
        
        return self.chat_assistant.chat(
            question=question,
            compressed_content=self.compressed_content,
            original_material=self.original_material
        )
    
    def update_compressed_content(self, edited_content: str):
        """
        Update the compressed content (e.g., after user edits)
        
        Args:
            edited_content: User-edited teaching brief
        """
        self.compressed_content = edited_content
    
    def get_compressed_content(self) -> str:
        """Get the current compressed teaching brief"""
        return self.compressed_content
    
    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_assistant.clear_history()
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the chat history"""
        return self.chat_assistant.get_history()