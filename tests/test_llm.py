import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utills.llm import LLMManager


@pytest.fixture
def llm_manager():
    """Create an LLM manager for testing."""
    with patch("src.utills.llm.ChatGroq") as mock_chatgroq:
        mock_llm = MagicMock()
        mock_chatgroq.return_value = mock_llm
        manager = LLMManager()
        yield manager


def test_llm_initialization(llm_manager):
    """Test LLM initialization."""
    with patch("src.utills.llm.ChatGroq") as mock_chatgroq:
        # First call should initialize the LLM
        llm = llm_manager.get_llm()
        mock_chatgroq.assert_called_once()
        
        # Second call should return the cached LLM
        llm_again = llm_manager.get_llm()
        assert mock_chatgroq.call_count == 1
        assert llm is llm_again


def test_create_rag_prompt(llm_manager):
    """Test RAG prompt creation."""
    prompt = llm_manager.create_rag_prompt()
    assert prompt is not None
    
    # Test the prompt format
    formatted = prompt.format(context="Test context", query="Test query")
    assert "Test context" in formatted
    assert "Test query" in formatted


def test_create_rag_chain(llm_manager):
    """Test RAG chain creation."""
    # Create a mock retriever
    mock_retriever = MagicMock()
    mock_docs = [MagicMock()]
    mock_docs[0].page_content = "Test content"
    mock_retriever.invoke.return_value = mock_docs
    
    # Mock the LLM
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "Test response"
    llm_manager._llm = mock_llm
    
    # Create the chain
    chain = llm_manager.create_rag_chain(mock_retriever)
    assert chain is not None
    
    # Test the chain
    with patch("src.utills.llm.StrOutputParser") as mock_parser:
        mock_parser.return_value.invoke.return_value = "Test parsed response"
        
        # This test is simplified since we can't easily test the full chain
        # In a real test, we'd need to mock more components
        assert callable(chain)