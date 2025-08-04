import os
import sys
import time
import json
from typing import Dict, Any, List

# Add project root and src to path to resolve imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Import LangSmith components
from langsmith import traceable

# Import project components
from src.utills.llm import LLMManager
from src.utills.embeddings import EmbeddingsManager
from src.utills.workflow import AgentWorkflow
from src.utills.tools import WeatherTools, DocumentTools

# Trace the LLM initialization
@traceable(name="initialize_llm")
def initialize_llm():
    """Initialize the LLM manager and return the LLM."""
    llm_manager = LLMManager()
    return llm_manager.get_llm()

# Trace the retriever initialization
@traceable(name="initialize_retriever")
def initialize_retriever(collection_name="resume_collection"):
    """Initialize the embeddings manager and return the retriever."""
    embeddings_manager = EmbeddingsManager(collection_name)
    return embeddings_manager.get_retriever()

# Trace the document query tool
@traceable(name="query_document")
def query_document(query: str, retriever, llm_manager):
    """Query the document using the RAG chain."""
    document_tools = DocumentTools(retriever, llm_manager)
    tool = document_tools.query_document_tool
    return tool(query)

# Trace the weather tool
@traceable(name="get_weather")
def get_weather(location: str):
    """Get weather information for a location."""
    weather_tools = WeatherTools()
    tool = weather_tools.get_weather_tool
    return tool(location)

# Trace the full workflow
@traceable(name="full_workflow")
def process_with_workflow(query: str):
    """Process a query through the full agent workflow."""
    # Initialize workflow
    workflow = AgentWorkflow("resume_collection")
    workflow.setup()
    
    # Process query
    start_time = time.time()
    result = workflow.process_query(query)
    end_time = time.time()
    
    # Extract conversation
    messages = workflow.extract_conversation(result)
    
    # Format the result
    formatted_result = {
        "messages": messages,
        "processing_time": end_time - start_time
    }
    
    return formatted_result

# Main function
def main():
    # Test queries
    queries = [
            "What is the temperature in Tokyo?",
            "What education does Abhinandan Kumar have?",
            "What is Abhinandan Kumar work experience?",
            "What skills does Abhinandan Kumar have?",
            "Can you explain the offer where Abhinandan Kumar got from the skillzo?",
            "what is the Salary Structure of Skillzo which is provided to Abhinandan Kumar?",
            "What technologies does Abhinandan Kumar know?",
            "What's the weather in London?"
        ]
    
    for query in queries:
        print(f"\nProcessing query: {query}")
        
        # Process query with tracing
        result = process_with_workflow(query)
        
        # Display the full result with all messages
        print(json.dumps(result, indent=2))
        print(f"Processing time: {result['processing_time']:.2f} seconds")
    
    print("\nAll runs have been traced in LangSmith.")
    print(f"View them at: https://smith.langchain.com/projects/{os.environ.get('LANGSMITH_PROJECT', 'rag-evaluation')}")

if __name__ == "__main__":
    main()