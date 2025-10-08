import asyncio
import logging
import json
from typing import Dict, Optional, List
from collections import defaultdict

from fastapi import HTTPException

from langchain_community.embeddings import HuggingFaceEmbeddings

from scripts.services.database import execute_query

from config import Config
config = Config()

chunk_size = config.CHUNK_SIZE
chunk_overlap = config.CHUNK_OVERLAP
model_name = config.HUGGINGFACE_MODEL

logger = logging.getLogger(__name__)

class RAGService:
    """RAG service using LangChain + Groq embeddings + Postgres storage"""
    
    def __init__(self):
        self.embeddings_model = HuggingFaceEmbeddings(model_name = model_name)
    
    async def get_embeddings(self, text: str):
        """Generate embeddings via LangChain Groq wrapper"""
        try:
            embedding = await asyncio.to_thread(self.embeddings_model.embed_query, text)
            if not embedding:
                raise ValueError("Empty embedding returned")
            return embedding
        
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        chunk_size: number of characters per chunk
        overlap: number of overlapping characters
        """
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunks.append(text[start:end])
            start += chunk_size - chunk_overlap

        return chunks
    
    async def store_document(
        self,
        table: str,
        chunk: str,
        chunk_index: int,
        metadata: Optional[Dict] = None,
        **kwargs  # Additional table-specific fields
    ):
        """
        Store a chunk in the specified table with embedding.
        
        table: one of ['curriculum', 'lesson_plan', 'material', 'knowledge_entry']
        kwargs: table-specific fields
            - curriculum: subject, grade, description
            - lesson_plan: title, objectives, duration
            - material: title, resource_type
            - knowledge_entry: source
        """
        try:
            embedding = await self.get_embeddings(chunk)
            
            # Prepare columns and values
            data = {
                **kwargs,
                "chunk": chunk,
                "chunk_index": chunk_index,
                "embedding": embedding,
                "metadata": json.dumps(metadata or {})  # <-- convert dict to JSON string
            }
            columns = ", ".join(data.keys())
            values = ", ".join(f"%({k})s" for k in data.keys())

            new_table = table.lower().replace(" ", "_")
            
            query = f"INSERT INTO eduai.{new_table} ({columns}) VALUES ({values});"
            execute_query(query, data, True)
            
            logger.info(f"Chunk {chunk_index} stored in table {table}.")
        
        except Exception as e:
            logger.error(f"Failed to store chunk in {table}: {e}")
            raise HTTPException(status_code = 500, detail = f"Failed to store chunk: {str(e)}")
        
    async def search_similar_chunk(self, table: str, query_text: str, top_k: int = 5):
        """
        Search for top-k similar chunks in the given table using pgvector.
        """

        try:
            embedding = await self.get_embeddings(query_text)
            query = f"""
            SELECT id, chunk, chunk_index, metadata,
                1 - (embedding <=> %(embedding)s::vector) AS similarity
            FROM eduai.{table}
            ORDER BY embedding <=> %(embedding)s::vector
            LIMIT %(top_k)s
            """
            
            params = {
                "embedding": embedding, 
                "top_k": top_k
            }

            params = {
                "embedding": embedding,
                "top_k": top_k
            }
            
            results = execute_query(query, params, is_update = False)

            return results
        except Exception as e:
            logger.error(f"Similar chunk search failed: {e}")
            return []

    def organize_chunks_for_llm(self, results):
        """
        Groups chunks by table and document, sorts by chunk_index,
        and concatenates into a single string per document.
        """
        grouped = defaultdict(lambda: defaultdict(list))

        # Group by table and document ID
        for res in results:
            table = res['table']
            doc_id = res['id']
            grouped[table][doc_id].append(res)

        # Sort chunks by chunk_index and concatenate
        organized = {}
        for table, docs in grouped.items():
            organized[table] = {}
            for doc_id, chunks in docs.items():
                sorted_chunks = sorted(chunks, key=lambda x: x['chunk_index'])
                full_text = "\n".join(chunk['chunk'] for chunk in sorted_chunks)
                organized[table][doc_id] = full_text

        return organized