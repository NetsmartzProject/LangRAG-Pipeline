import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Streamlit app functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit_app


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit functions."""
    with patch("streamlit_app.st") as mock_st:
        yield mock_st


@pytest.fixture
def mock_requests():
    """Mock requests module."""
    with patch("streamlit_app.requests") as mock_requests:
        yield mock_requests


def test_call_api_success(mock_requests):
    """Test the call_api function with a successful response."""
    # Configure mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"messages": [], "processing_time": 1.5}
    mock_requests.post.return_value = mock_response
    
    # Call the function
    result = streamlit_app.call_api("Test query")
    
    # Verify the result
    assert result == {"messages": [], "processing_time": 1.5}
    mock_requests.post.assert_called_once()
    assert mock_requests.post.call_args[0][0] == "http://localhost:8000/chat/query"
    assert "Test query" in mock_requests.post.call_args[1]["data"]


def test_call_api_connection_error(mock_requests, mock_streamlit):
    """Test the call_api function with a connection error."""
    # Configure mock to raise ConnectionError
    mock_requests.post.side_effect = mock_requests.exceptions.ConnectionError("Connection error")
    
    # Call the function
    result = streamlit_app.call_api("Test query")
    
    # Verify the result
    assert result is None
    mock_streamlit.error.assert_called_once()
    assert "API connection error" in mock_streamlit.error.call_args[0][0]


def test_call_api_http_error(mock_requests, mock_streamlit):
    """Test the call_api function with an HTTP error."""
    # Configure mock to raise HTTPError
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_error = mock_requests.exceptions.HTTPError("HTTP Error")
    mock_error.response = mock_response
    mock_requests.post.side_effect = mock_error
    
    # Call the function
    result = streamlit_app.call_api("Test query")
    
    # Verify the result
    assert result is None
    mock_streamlit.error.assert_called_once()
    assert "API Error: 500" in mock_streamlit.error.call_args[0][0]


def test_check_api_health_success(mock_requests):
    """Test the check_api_health function with a successful response."""
    # Configure mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_requests.get.return_value = mock_response
    
    # Call the function
    result = streamlit_app.check_api_health()
    
    # Verify the result
    assert result is True
    mock_requests.get.assert_called_once_with("http://localhost:8000/chat/health")


def test_check_api_health_failure(mock_requests):
    """Test the check_api_health function with a failed response."""
    # Configure mock response
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_requests.get.return_value = mock_response
    
    # Call the function
    result = streamlit_app.check_api_health()
    
    # Verify the result
    assert result is False


def test_format_thinking():
    """Test the format_thinking function."""
    # Create test messages
    messages = [
        {"role": "human", "content": "What's the weather in London?"},
        {"role": "system", "content": "Successfully transferred to weather_agent"},
        {"role": "tool", "content": "Weather in London: 15.5°C, feels like 14.8°C, scattered clouds, humidity: 76%"},
        {"role": "ai", "content": "The weather in London is 15.5°C with scattered clouds."},
        {"role": "system", "content": "Successfully transferred back to supervisor"}
    ]
    
    # Format the thinking
    thinking = streamlit_app.format_thinking(messages)
    
    # Verify the result
    assert "👤 **Human Query**: What's the weather in London?" in thinking
    assert "🔄 **Routing to weather_agent**" in thinking
    assert "🌤️ **Weather Tool**: Weather in London: 15.5°C" in thinking
    assert "🤖 **AI (weather_agent)**: The weather in London is 15.5°C with scattered clouds." in thinking
    assert "🔙 **Returning to Supervisor**" in thinking


def test_extract_professional_response():
    """Test the extract_professional_response function."""
    # Test with weather tool output
    weather_messages = [
        {"role": "human", "content": "What's the weather in London?"},
        {"role": "tool", "content": "Weather in London: 15.5°C, feels like 14.8°C, scattered clouds, humidity: 76%"},
        {"role": "ai", "content": "The weather in London is 15.5°C with scattered clouds."}
    ]
    weather_response = streamlit_app.extract_professional_response(weather_messages)
    assert "The weather in London is 15.5°C with scattered clouds." in weather_response
    
    # Test with document tool output
    document_messages = [
        {"role": "human", "content": "What skills does Abhi have?"},
        {"role": "tool", "content": "answer='Abhi has skills in Python, JavaScript, and machine learning.'"},
        {"role": "ai", "content": "Abhi has skills in Python, JavaScript, and machine learning."}
    ]
    document_response = streamlit_app.extract_professional_response(document_messages)
    assert "Abhi has skills in Python, JavaScript, and machine learning." in document_response
    
    # Test with only AI response
    ai_messages = [
        {"role": "human", "content": "Hello"},
        {"role": "ai", "content": "Hello! How can I help you today?"}
    ]
    ai_response = streamlit_app.extract_professional_response(ai_messages)
    assert "Hello! How can I help you today?" in ai_response