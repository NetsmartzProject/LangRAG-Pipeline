import os
import sys
import pandas as pd
import time
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path to resolve imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Import LangSmith components
from langsmith import Client
from langchain.smith import run_on_dataset  # Updated import path
from langsmith.schemas import Example, Run

# Import project components
from src.utills.llm import LLMManager
from src.utills.embeddings import EmbeddingsManager
from src.utills.workflow import AgentWorkflow
from src.config.logger import logger

# Use environment variables that are set in the PowerShell script
# No need to set them here as they're already set in the script

def create_test_dataset():
    """
    Create a test dataset for evaluation.
    
    Returns:
        pd.DataFrame: Dataset with test queries and expected outputs
    """
    # Define test queries and expected outputs
    test_data = [
        {
            "query": "What skills does Abhinandan Kumar have?",
            "expected": "Information about Abhinandan's programming skills, experience, and technical abilities."
        },
        {
            "query": "What is Abhinandan's work experience?",
            "expected": "Details about Abhinandan's work history and professional roles."
        },
        {
            "query": "What is the weather in London?",
            "expected": "Current weather conditions in London."
        },
        {
            "query": "What education does Abhinandan have?",
            "expected": "Information about Abhinandan's educational background and qualifications."
        },
        {
            "query": "What is the temperature in Tokyo?",
            "expected": "Current temperature in Tokyo."
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame(test_data)
    
    # Save to CSV for reference
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"test_dataset_{timestamp}.csv", index=False)
    print(f"Test dataset created with {len(df)} examples")
    
    return df

def create_langsmith_dataset(client, dataset_name, test_data):
    """
    Create a dataset in LangSmith.
    
    Args:
        client: LangSmith client
        dataset_name: Name for the dataset
        test_data: Test data DataFrame
        
    Returns:
        str: Dataset ID
    """
    try:
        # Check if dataset already exists
        datasets = client.list_datasets()
        for dataset in datasets:
            if dataset.name == dataset_name:
                print(f"Dataset '{dataset_name}' already exists with ID: {dataset.id}")
                # Delete existing examples to avoid duplicates
                examples = client.list_examples(dataset_name=dataset_name)
                for example in examples:
                    client.delete_example(example.id)
                return dataset.id
        
        # Create new dataset
        dataset = client.create_dataset(dataset_name=dataset_name)
        dataset_id = dataset.id
        print(f"Created new dataset '{dataset_name}' with ID: {dataset_id}")
        
        # Add examples to dataset
        for _, row in test_data.iterrows():
            client.create_example(
                inputs={"query": row["query"]},
                outputs={"expected": row["expected"]},
                dataset_id=dataset_id
            )
        
        print(f"Added {len(test_data)} examples to dataset")
        return dataset_id
    
    except Exception as e:
        print(f"Error creating LangSmith dataset: {str(e)}")
        raise

def evaluate_rag_chain(client, dataset_id):
    """
    Evaluate the RAG chain using LangSmith.
    
    Args:
        client: LangSmith client
        dataset_id: Dataset ID
    """
    try:
        # Initialize components
        print("Initializing components for RAG chain evaluation...")
        embeddings_manager = EmbeddingsManager("resume_collection")
        retriever = embeddings_manager.get_retriever()
        
        if not retriever:
            print("Error: Retriever not available. Make sure the collection exists.")
            return
        
        llm_manager = LLMManager()
        rag_chain = llm_manager.create_rag_chain(retriever)
        
        # Create a wrapper function that matches LangSmith's expected interface
        def rag_chain_wrapper(query):
            try:
                result = rag_chain.invoke(query)
                # Extract answer from format answer='...'
                if isinstance(result, str) and result.startswith("answer='") and result.endswith("'"):
                    return result[8:-1]  # Remove answer=' and trailing '
                return result
            except Exception as e:
                print(f"Error in RAG chain: {str(e)}")
                return f"Error: {str(e)}"
        
        # Run evaluation - based on the function signature we discovered
        print("Running RAG chain evaluation...")
        results = run_on_dataset(
            client,  # First positional argument: client
            "rag-evaluation-dataset",  # Second positional argument: dataset_name
            lambda: rag_chain_wrapper,  # Third positional argument: llm_or_chain_factory
            project_name=os.environ.get("LANGSMITH_PROJECT", "rag-evaluation"),
            tags=["rag_chain"],
            verbose=True,
            evaluators=["criteria", "embedding_distance"]
        )
        
        print(f"RAG chain evaluation completed. Results available in LangSmith project.")
        return results
    
    except Exception as e:
        print(f"Error evaluating RAG chain: {str(e)}")
        raise

def evaluate_agent_workflow(client, dataset_id):
    """
    Evaluate the agent workflow using LangSmith.
    
    Args:
        client: LangSmith client
        dataset_id: Dataset ID
    """
    try:
        # Initialize components
        print("Initializing components for agent workflow evaluation...")
        workflow = AgentWorkflow("resume_collection")
        workflow.setup()
        
        # Define a wrapper function to match LangSmith's expected interface
        def workflow_wrapper(query):
            try:
                result = workflow.process_query(query)
                messages = workflow.extract_conversation(result)
                
                # Find the final AI response
                ai_responses = [msg["content"] for msg in messages if msg["role"] == "ai"]
                final_response = ai_responses[-1] if ai_responses else "No response"
                
                # If the response is from a tool and in answer='...' format, extract it
                tool_responses = [msg["content"] for msg in messages if msg["role"] == "tool"]
                for resp in tool_responses:
                    if resp.startswith("answer='") and resp.endswith("'"):
                        extracted = resp[8:-1]  # Remove answer=' and trailing '
                        return extracted
                
                return final_response
            except Exception as e:
                print(f"Error in workflow: {str(e)}")
                return f"Error: {str(e)}"
        
        # Run evaluation - based on the function signature we discovered
        print("Running agent workflow evaluation...")
        results = run_on_dataset(
            client,  # First positional argument: client
            "rag-evaluation-dataset",  # Second positional argument: dataset_name
            lambda: workflow_wrapper,  # Third positional argument: llm_or_chain_factory
            project_name=os.environ.get("LANGSMITH_PROJECT", "rag-evaluation"),
            tags=["agent_workflow"],
            verbose=True,
            evaluators=["criteria", "embedding_distance"]
        )
        
        print(f"Agent workflow evaluation completed. Results available in LangSmith project.")
        return results
    
    except Exception as e:
        print(f"Error evaluating agent workflow: {str(e)}")
        raise

def main():
    """Main function to run evaluations"""
    try:
        print("Checking environment variables...")
        required_vars = ["LANGSMITH_API_KEY", "OPENAI_API_KEY"]
        for var in required_vars:
            if not os.environ.get(var):
                print(f"Warning: {var} environment variable is not set")
        
        # Create LangSmith client
        client = Client()
        print("Connected to LangSmith")
        
        # Create test dataset
        test_data = create_test_dataset()
        
        # Create LangSmith dataset
        dataset_id = create_langsmith_dataset(client, "rag-evaluation-dataset", test_data)
        
        # Evaluate RAG chain
        print("\n=== EVALUATING RAG CHAIN ===")
        evaluate_rag_chain(client, dataset_id)
        
        # Evaluate agent workflow
        print("\n=== EVALUATING AGENT WORKFLOW ===")
        evaluate_agent_workflow(client, dataset_id)
        
        print("\n=== EVALUATION COMPLETE ===")
        print(f"View all results in the LangSmith UI: https://smith.langchain.com/projects/{os.environ.get('LANGSMITH_PROJECT', 'rag-evaluation')}")
    
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()