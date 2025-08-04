import pytest
from unittest.mock import patch, MagicMock
import json


def test_root_endpoint(client):
    """Test the root endpoint of the API."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "AI Assistant API is running"}


def test_health_endpoint(client):
    """Test the health endpoint of the API."""
    with patch("src.routes.chat.get_workflow") as mock_get_workflow:
        mock_workflow = MagicMock()
        mock_workflow.collection_name = "test_collection"
        mock_get_workflow.return_value = mock_workflow
        
        response = client.get("/chat/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["workflow_initialized"] == True
        assert response.json()["collection_name"] == "test_collection"


def test_query_endpoint_success(client):
    """Test the query endpoint with a successful response."""
    with patch("src.routes.chat.get_workflow") as mock_get_workflow:
        # Create mock workflow
        mock_workflow = MagicMock()
        mock_result = {
            "messages": [
                {"type": "human", "content": "Test query"},
                {"type": "tool", "content": "Successfully transferred to rag_agent"},
                {"type": "tool", "content": "answer='Test response'"},
                {"type": "ai", "content": "Test response"},
            ]
        }
        mock_workflow.process_query.return_value = mock_result
        
        # Mock the extract_conversation method
        mock_workflow.extract_conversation.return_value = [
            {"role": "human", "content": "Test query"},
            {"role": "tool", "content": "Successfully transferred to rag_agent"},
            {"role": "tool", "content": "answer='Test response'"},
            {"role": "ai", "content": "Test response"},
        ]
        
        mock_get_workflow.return_value = mock_workflow
        
        # Make the request
        response = client.post(
            "/chat/query",
            json={"text": "Test query"}
        )
        
        # Verify the response
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert len(data["messages"]) == 4
        assert data["messages"][0]["role"] == "human"
        assert data["messages"][0]["content"] == "Test query"
        assert "processing_time" in data


def test_query_endpoint_error(client):
    """Test the query endpoint with an error response."""
    with patch("src.routes.chat.get_workflow") as mock_get_workflow:
        # Make the workflow raise an exception
        mock_workflow = MagicMock()
        mock_workflow.process_query.side_effect = ValueError("Test error")
        mock_get_workflow.return_value = mock_workflow
        
        # Make the request
        response = client.post(
            "/chat/query",
            json={"text": "Test query"}
        )
        
        # Verify the response
        assert response.status_code == 500
        assert "detail" in response.json()
        assert "Test error" in response.json()["detail"]