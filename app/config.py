import os
from typing import Optional

class Config:
    # API Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-3-small"
    
    # Vector DB Configuration
    VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "faiss")  # faiss, chromadb, redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./chromadb_data")
    
    # Search Configuration
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    MAX_RESULTS = int(os.getenv("MAX_RESULTS", "10"))
    
    # LLM Configuration
    LLM_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0.1

config = Config()