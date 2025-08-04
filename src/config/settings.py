import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    OPENWEATHER_API_KEY: str 
    GROQ_API_KEY: str 
    OPENAI_API_KEY: str 
    
    # LLM Configuration
    LLM_MODEL: str 
    LLM_TEMPERATURE: float 
    LLM_MAX_TOKENS: int 
    
    # Vector DB Configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_PREFER_GRPC: bool = os.getenv("QDRANT_PREFER_GRPC", "False").lower() == "true"
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "resume_collection")
    
    # Embedding Model Configuration
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en")
    EMBEDDING_MODEL_DEVICE: str = os.getenv("EMBEDDING_MODEL_DEVICE", "cpu")
    EMBEDDING_NORMALIZE: bool = os.getenv("EMBEDDING_NORMALIZE", "False").lower() == "true"
    
    # Document Processing
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # Retrieval Configuration
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "3"))
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
