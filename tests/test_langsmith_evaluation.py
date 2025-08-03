import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from src.utills.llm import LLMManager
from src.utills.workflow import AgentWorkflow


class TestLangSmithEvaluation:
    """Tests for LangSmith evaluation."""
    
    @pytest.fixture
    def setup_langsmith(self):
        """Set up LangSmith for testing."""
        # Set environment variables for LangSmith
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGCHAIN_API_KEY"  # Replace with a test API key
        os.environ["LANGCHAIN_PROJECT"] = "ai-assistant-evaluation"
        
        yield
        
        # Clean up environment variables
        del os.environ["LANGCHAIN_TRACING_V2"]
        del os.environ["LANGCHAIN_ENDPOINT"]
        del os.environ["LANGCHAIN_API_KEY"]
        del os.environ["LANGCHAIN_PROJECT"]
    
    @patch("langchain.smith.evaluation.RunEvalConfig")
    @patch("langchain.smith.evaluation.run_on_dataset")
    def test_evaluate_rag_chain(self, mock_run_on_dataset, mock_run_eval_config, setup_langsmith):
        """Test evaluation of RAG chain with LangSmith."""
        # Create mock dataset
        mock_dataset = MagicMock()
        
        # Create mock LLM manager
        mock_llm_manager = MagicMock(spec=LLMManager)
        mock_chain = MagicMock()
        mock_llm_manager.create_rag_chain.return_value = mock_chain
        
        # Create mock retriever
        mock_retriever = MagicMock()
        
        # Configure the mock evaluation
        mock_run_eval_config.return_value = MagicMock()
        mock_run_on_dataset.return_value = MagicMock()
        
        # Create the RAG chain
        rag_chain = mock_llm_manager.create_rag_chain(mock_retriever)
        
        # Run the evaluation
        from langchain.smith import RunEvalConfig
        from langchain.smith import run_on_dataset
        
        # Define evaluation metrics
        eval_config = RunEvalConfig(
            evaluators=["qa_relevance", "answer_relevance", "context_relevance"],
            custom_evaluators=[]
        )
        
        # Run evaluation
        evaluation_result = run_on_dataset(
            dataset_name="rag_evaluation_dataset",
            llm_or_chain_factory=lambda: rag_chain,
            evaluation=eval_config
        )
        
        # Verify the evaluation was called correctly
        mock_run_on_dataset.assert_called_once()
    
    @patch("langchain.smith.evaluation.RunEvalConfig")
    @patch("langchain.smith.evaluation.run_on_dataset")
    def test_evaluate_agent_workflow(self, mock_run_on_dataset, mock_run_eval_config, setup_langsmith):
        """Test evaluation of agent workflow with LangSmith."""
        # Create mock dataset
        mock_dataset = MagicMock()
        
        # Create mock workflow
        mock_workflow = MagicMock(spec=AgentWorkflow)
        mock_workflow.process_query.side_effect = lambda x: {"messages": [MagicMock(type="ai", content=f"Response to {x}")]}
        
        # Configure the mock evaluation
        mock_run_eval_config.return_value = MagicMock()
        mock_run_on_dataset.return_value = MagicMock()
        
        # Define a wrapper function for the workflow
        def workflow_wrapper(query):
            result = mock_workflow.process_query(query)
            # Extract the AI response from the messages
            for message in result["messages"]:
                if hasattr(message, "type") and message.type == "ai":
                    return message.content
            return "No response"
        
        # Run the evaluation
        from langchain.smith import RunEvalConfig
        from langchain.smith import run_on_dataset
        
        # Define evaluation metrics
        eval_config = RunEvalConfig(
            evaluators=["criteria", "labeled_criteria", "embedding_distance"],
            custom_evaluators=[]
        )
        
        # Run evaluation
        evaluation_result = run_on_dataset(
            dataset_name="agent_evaluation_dataset",
            llm_or_chain_factory=lambda: workflow_wrapper,
            evaluation=eval_config
        )
        
        # Verify the evaluation was called correctly
        mock_run_on_dataset.assert_called_once()