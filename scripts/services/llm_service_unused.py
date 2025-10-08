import logging
from typing import List, Dict

from fastapi import HTTPException

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from config import Config

logger = logging.getLogger(__name__)
config = Config()

model = config.GROQ_MODEL
api_key = config.GROQ_API_KEY

class LLMService:
    """Large Language Model interaction service for document Q&A using Groq"""
    
    @staticmethod
    async def answer_document_query(
        query_text: str, 
        similar_chunks: List[Dict]
    ):
        """
        Generate an LLM-based answer for a user query grounded in document chunks.

        Args:
            query_text: The user's question.
            similar_chunks: List of retrieved chunks, each with 'chunk' and 'table'.

        Returns:
            JSON-compatible dict with structured answer.
        """
        # Prepare context from top 5 relevant chunks
        context = "Relevant document excerpts:\n"
        

        for i, chunk in enumerate(similar_chunks):
            # Handle nested dicts or lists gracefully
            if isinstance(chunk, dict):
                # If 'chunk' itself contains nested dicts (like {'curriculum': {...}})
                # extract the inner text values recursively
                values = []
                for val in chunk.values():
                    if isinstance(val, dict):
                        values.extend(val.values())
                    elif isinstance(val, list):
                        values.extend(v['chunk'] if isinstance(v, dict) and 'chunk' in v else v for v in val)
                    else:
                        values.append(val)
                
                # Flatten values that are strings
                flat_texts = [v for v in values if isinstance(v, str)]
                text = "\n".join(flat_texts)
            else:
                # In normal case, chunk is already a dict with 'chunk' key
                text = chunk.get("chunk", "")

            context += f"Chunk {i+1} (from {chunk.get('table', 'unknown')}):\n{text}\n\n"

        print(context)
        system_prompt = f"""
        You are an AI assistant specialized in providing accurate answers based on document content.
        
        Follow these strict instructions:
        - Your entire response MUST be a valid JSON object.
        - Do NOT include any text before or after the JSON.
        - Do not use markdown backticks.

        USER QUERY:
        {{user_message}}

        {{context}}

        The JSON object you return must have the following keys:
        1. "answer": A concise and clear answer to the user's query.
        2. "supporting_chunks": List of chunk indices (from context) used to generate the answer.
        3. "uncertainty": A score from 0-100 indicating how confident the answer is based on the provided chunks.
        4. "recommendations": Optional suggestions for further reading or clarification from the documents.
        """

        try:
            model = ChatGroq(
                model = model,  # type: ignore
                temperature = 0,
                api_key = config.GROQ_API_KEY) # type: ignore
            
            memory_buffer = ChatMessageHistory()

            prompt_buffer = ChatPromptTemplate(
                                [("system", system_prompt),
                                MessagesPlaceholder(variable_name = "history"),
                                ("human", "{{user_message}}")
                                ])

            chain_buffer = prompt_buffer | model

            conversation_buffer = RunnableWithMessageHistory(chain_buffer, 
                           lambda : memory_buffer,
                           input_messages_key = "user_message",
                           history_messages_key = "history")

            response = conversation_buffer.invoke({"user_message" : query_text, "context": context})

            return memory_buffer


        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            raise HTTPException(status_code=500, detail=f"LLM query failed: {str(e)}")