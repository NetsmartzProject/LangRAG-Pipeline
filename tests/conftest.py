import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the FastAPI app
from src.main import app


@pytest.fixture
def client():
    """
    Create a test client for the FastAPI app.
    
    Returns:
        TestClient: A test client for the FastAPI app.
    """
    return TestClient(app)


@pytest.fixture
def mock_llm_manager():
    """
    Create a mock LLM manager for testing.
    
    Returns:
        Mock: A mock LLM manager.
    """
    from unittest.mock import MagicMock
    from src.utills.llm import LLMManager
    
    mock_manager = MagicMock(spec=LLMManager)
    mock_llm = MagicMock()
    mock_manager.get_llm.return_value = mock_llm
    mock_manager.create_rag_chain.return_value = lambda query: f"answer='Mocked response for {query}'"
    
    return mock_manager


@pytest.fixture
def mock_embeddings_manager():
    """
    Create a mock embeddings manager for testing.
    
    Returns:
        Mock: A mock embeddings manager.
    """
    from unittest.mock import MagicMock
    from src.utills.embeddings import EmbeddingsManager
    
    mock_manager = MagicMock(spec=EmbeddingsManager)
    mock_retriever = MagicMock()
    mock_manager.get_retriever.return_value = mock_retriever
    mock_manager.check_collection_exists.return_value = True
    
    return mock_manager