import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Configuration
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://ire:password@172.17.10.201:5432/datathon")
    
    # Document Processing
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # Embeddings
    HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "sentence-transformers/all-mpnet-base-v2")
    
    # File Upload Limits
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 15 * 1024 * 1024))  # 15MB default
    SUPPORTED_FORMATS = os.getenv("SUPPORTED_FORMATS", ".pdf,.docx,.txt,.doc").split(",")
    
    # Validation
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required but not set in environment variables")
        if not cls.DATABASE_URL:
            raise ValueError("DATABASE_URL is required but not set in environment variables")
        return True

# Validate configuration on import
Config.validate()