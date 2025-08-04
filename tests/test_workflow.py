import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utills.workflow import AgentWorkflow


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for testing."""
    with patch("src.utills.workflow.EmbeddingsManager") as mock_embeddings_manager_class, \
         patch("src.utills.workflow.LLMManager") as mock_llm_manager_class, \
         patch("src.utills.workflow.WeatherTools") as mock_weather_tools_class, \
         patch("src.utills.workflow.DocumentTools") as mock_document_tools_class, \
         patch("src.utills.workflow.AgentNode") as mock_agent_node_class:
        
        # Setup mock embeddings manager
        mock_embeddings_manager = MagicMock()
        mock_retriever = MagicMock()
        mock_embeddings_manager.get_retriever.return_value = mock_retriever
        mock_embeddings_manager_class.return_value = mock_embeddings_manager
        
        # Setup mock LLM manager
        mock_llm_manager = MagicMock()
        mock_llm = MagicMock()
        mock_llm_manager.get_llm.return_value = mock_llm
        mock_llm_manager_class.return_value = mock_llm_manager
        
        # Setup mock weather tools
        mock_weather_tools = MagicMock()
        mock_weather_tool = MagicMock()
        mock_weather_tools.get_weather_tool = mock_weather_tool
        mock_weather_tools_class.return_value = mock_weather_tools
        
        # Setup mock document tools
        mock_document_tools = MagicMock()
        mock_document_tool = MagicMock()
        mock_document_tools.query_document_tool = mock_document_tool
        mock_document_tools_class.return_value = mock_document_tools
        
        # Setup mock agent node
        mock_agent_node = MagicMock()
        mock_weather_agent = MagicMock()
        mock_rag_agent = MagicMock()
        mock_supervisor = MagicMock()
        mock_agent_node.create_weather_agent.return_value = mock_weather_agent
        mock_agent_node.create_rag_agent.return_value = mock_rag_agent
        mock_agent_node.create_supervisor_agent.return_value = mock_supervisor
        mock_agent_node_class.return_value = mock_agent_node
        
        yield {
            "embeddings_manager": mock_embeddings_manager,
            "llm_manager": mock_llm_manager,
            "weather_tools": mock_weather_tools,
            "document_tools": mock_document_tools,
            "agent_node": mock_agent_node,
            "retriever": mock_retriever,
            "weather_tool": mock_weather_tool,
            "document_tool": mock_document_tool,
            "weather_agent": mock_weather_agent,
            "rag_agent": mock_rag_agent,
            "supervisor": mock_supervisor
        }


def test_workflow_initialization():
    """Test workflow initialization."""
    workflow = AgentWorkflow()
    assert workflow.collection_name == "resume_collection"
    assert workflow.embeddings_manager is None
    assert workflow.llm_manager is None
    assert workflow.weather_tools is None
    assert workflow.document_tools is None
    assert workflow.agent_node is None
    assert workflow.retriever is None
    
    # Test with custom collection name
    custom_workflow = AgentWorkflow("custom_collection")
    assert custom_workflow.collection_name == "custom_collection"


def test_workflow_setup(mock_dependencies):
    """Test workflow setup."""
    workflow = AgentWorkflow()
    workflow.setup()
    
    # Verify that all dependencies were initialized
    assert workflow.embeddings_manager is not None
    assert workflow.llm_manager is not None
    assert workflow.weather_tools is not None
    assert workflow.document_tools is not None
    assert workflow.agent_node is not None
    assert workflow.retriever is not None
    
    # Verify that agents were created
    mock_dependencies["agent_node"].create_weather_agent.assert_called_once()
    mock_dependencies["agent_node"].create_rag_agent.assert_called_once()
    mock_dependencies["agent_node"].create_supervisor_agent.assert_called_once()


def test_process_query(mock_dependencies):
    """Test query processing."""
    workflow = AgentWorkflow()
    workflow.setup()
    
    # Configure mock result
    mock_result = {
        "messages": [
            {"type": "human", "content": "Test query"},
            {"type": "ai", "content": "Test response"}
        ]
    }
    mock_dependencies["agent_node"].process_query.return_value = mock_result
    
    # Process a query
    result = workflow.process_query("Test query")
    
    # Verify the result
    assert result == mock_result
    mock_dependencies["agent_node"].process_query.assert_called_once_with("Test query")


def test_extract_conversation():
    """Test conversation extraction."""
    workflow = AgentWorkflow()
    
    # Create a mock result
    mock_message1 = MagicMock()
    mock_message1.type = "human"
    mock_message1.content = "Test query"
    
    mock_message2 = MagicMock()
    mock_message2.type = "tool"
    mock_message2.content = "Successfully transferred to rag_agent"
    
    mock_message3 = MagicMock()
    mock_message3.type = "ai"
    mock_message3.content = "Test response"
    
    mock_result = {
        "messages": [mock_message1, mock_message2, mock_message3]
    }
    
    # Extract the conversation
    conversation = workflow.extract_conversation(mock_result)
    
    # Verify the conversation
    assert len(conversation) == 3
    assert conversation[0]["role"] == "human"
    assert conversation[0]["content"] == "Test query"
    assert conversation[1]["role"] == "tool"
    assert conversation[1]["content"] == "Successfully transferred to rag_agent"
    assert conversation[2]["role"] == "ai"
    assert conversation[2]["content"] == "Test response"