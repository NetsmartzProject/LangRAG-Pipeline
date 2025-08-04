import os
import sys

# Add project root and src to path to resolve imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Import LangSmith components
from langsmith import traceable

# Import project components
from src.utills.llm import LLMManager
from src.utills.workflow import AgentWorkflow

# Define a traceable function
@traceable(name="process_user_query")
def process_user_query(query: str):
    """Process a user query through the agent workflow."""
    # Initialize workflow
    workflow = AgentWorkflow("resume_collection")
    workflow.setup()
    
    # Process query
    result = workflow.process_query(query)
    
    # Extract conversation
    messages = workflow.extract_conversation(result)
    
    # Return the final AI response
    ai_responses = [msg["content"] for msg in messages if msg["role"] == "ai"]
    return ai_responses[-1] if ai_responses else "No response"

# Main function
def main():
    # Test query
    query = "What skills does Abhinandan Kumar have?"
    print(f"Processing query: {query}")
    
    # Process query with tracing
    response = process_user_query(query)
    
    print(f"Response: {response}")
    print("\nThis run has been traced in LangSmith.")
    print(f"View it at: https://smith.langchain.com/projects/{os.environ.get('LANGSMITH_PROJECT', 'rag-evaluation')}")

if __name__ == "__main__":
    main()