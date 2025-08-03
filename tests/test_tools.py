import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utills.tools import WeatherTools, DocumentTools


def test_weather_tool_creation():
    """Test weather tool creation."""
    weather_tools = WeatherTools()
    weather_tool = weather_tools.get_weather_tool
    
    assert callable(weather_tool)
    assert weather_tool.__name__ == "get_weather"
    assert "Get the current weather for a city" in weather_tool.__doc__


@patch("src.utills.tools.requests.get")
def test_weather_tool_success(mock_get):
    """Test weather tool with a successful response."""
    # Setup mock responses
    mock_geo_response = MagicMock()
    mock_geo_response.json.return_value = [{"lat": 51.5074, "lon": -0.1278}]
    
    mock_weather_response = MagicMock()
    mock_weather_response.json.return_value = {
        "main": {"temp": 15.5, "feels_like": 14.8, "humidity": 76},
        "weather": [{"main": "Clouds", "description": "scattered clouds"}],
        "name": "London",
        "sys": {"country": "GB"}
    }
    
    # Configure mock to return different responses for different URLs
    def side_effect(url):
        if "geo/1.0/direct" in url:
            return mock_geo_response
        else:
            return mock_weather_response
    
    mock_get.side_effect = side_effect
    
    # Create and call the tool
    weather_tools = WeatherTools()
    weather_tool = weather_tools.get_weather_tool
    result = weather_tool("London")
    
    # Verify the result
    assert "Weather in London" in result
    assert "15.5°C" in result
    assert "scattered clouds" in result
    assert "humidity: 76%" in result


@patch("src.utills.tools.requests.get")
def test_weather_tool_location_not_found(mock_get):
    """Test weather tool when location is not found."""
    # Setup mock response
    mock_geo_response = MagicMock()
    mock_geo_response.json.return_value = []
    mock_get.return_value = mock_geo_response
    
    # Create and call the tool
    weather_tools = WeatherTools()
    weather_tool = weather_tools.get_weather_tool
    result = weather_tool("NonexistentCity")
    
    # Verify the result
    assert "I couldn't find the location 'NonexistentCity'" in result


def test_document_tool_creation():
    """Test document tool creation."""
    # Create mock LLM manager and retriever
    mock_llm_manager = MagicMock()
    mock_retriever = MagicMock()
    
    # Create document tools
    document_tools = DocumentTools(mock_retriever, mock_llm_manager)
    document_tool = document_tools.query_document_tool
    
    assert callable(document_tool)
    assert document_tool.__name__ == "query_document"
    assert "Query the document to answer a question" in document_tool.__doc__


def test_document_tool_no_retriever():
    """Test document tool when no retriever is available."""
    # Create mock LLM manager
    mock_llm_manager = MagicMock()
    
    # Create document tools with no retriever
    document_tools = DocumentTools(None, mock_llm_manager)
    document_tool = document_tools.query_document_tool
    
    # Call the tool
    result = document_tool("Test query")
    
    # Verify the result
    assert "No documents have been loaded" in result


def test_document_tool_success():
    """Test document tool with a successful response."""
    # Create mock LLM manager and retriever
    mock_llm_manager = MagicMock()
    mock_retriever = MagicMock()
    
    # Configure mock RAG chain
    mock_rag_chain = MagicMock()
    mock_rag_chain.invoke.return_value = "Test response"
    mock_llm_manager.create_rag_chain.return_value = mock_rag_chain
    
    # Create document tools
    document_tools = DocumentTools(mock_retriever, mock_llm_manager)
    document_tool = document_tools.query_document_tool
    
    # Call the tool
    result = document_tool("Test query")
    
    # Verify the result
    assert "Test response" in result
    mock_rag_chain.invoke.assert_called_once_with("Test query")