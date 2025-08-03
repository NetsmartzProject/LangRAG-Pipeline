import os
import sys
import pandas as pd
import time
import random

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Fix import paths by adding src to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Now we can import from src modules
from src.utills.llm import LLMManager
from src.utills.embeddings import EmbeddingsManager
from src.utills.workflow import AgentWorkflow

print("=" * 50)
print("STARTING LOCAL EVALUATION")
print("=" * 50)

def create_dataset():
    """Create a simple evaluation dataset."""
    print("Creating evaluation dataset...")
    
    # Create example queries and expected outputs
    data = [
        {
            "input": "What's the weather in London?",
            "expected_output": "Weather information for London including temperature and conditions."
        },
        {
            "input": "Tell me about Abhinandan's skills",
            "expected_output": "Information about Abhinandan's programming skills, experience, and technical abilities."
        },
        {
            "input": "What experience does Abhinandan have?",
            "expected_output": "Details about Abhinandan's work experience and professional background."
        },
        {
            "input": "What is Abhinandan's educational background?",
            "expected_output": "Information about Abhinandan's education, degrees, and academic achievements."
        },
        {
            "input": "Do the Abhinandan Kumar any Job offers?",
            "expected_output": "Yes they have from Skillzo Company for AGentic Ai"
        },
        {
            "input": "what is the Salary Structure of Skillzo which is provided to Abhinandan Kumar?",
            "expected_output": "Detail Salary slip of Skillzo"
        },
        {
            "input": "What technologies does Abhinandan know?",
            "expected_output": "List of technologies, programming languages, and frameworks that Abhinandan is proficient in."
        }
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV for reference
    df.to_csv("evaluation_dataset.csv", index=False)
    print(f"Created dataset with {len(data)} examples and saved to evaluation_dataset.csv")
    
    return df

def run_local_evaluation(chain_or_wrapper, dataset, name=""):
    """Run a local evaluation without using LangSmith."""
    print(f"\nRunning local evaluation for {name}...")
    
    results = []
    for i, (_, row) in enumerate(dataset.iterrows()):
        query = row["input"]
        expected = row["expected_output"]
        
        print(f"\n--- Example {i+1}/{len(dataset)} ---")
        print(f"Query: {query}")
        print(f"Expected: {expected}")
        
        try:
            # Add random delay to avoid rate limiting
            delay = random.uniform(1.0, 3.0)
            print(f"Waiting {delay:.1f} seconds before processing to avoid rate limits...")
            time.sleep(delay)
            
            start_time = time.time()
            result = chain_or_wrapper(query)
            end_time = time.time()
            
            print(f"Result: {result[:200]}..." if len(result) > 200 else f"Result: {result}")
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            
            # Simple relevance check
            query_words = set(query.lower().split())
            result_words = set(result.lower().split())
            common_words = query_words.intersection(result_words)
            relevance = len(common_words) / len(query_words) if query_words else 0
            
            print(f"Simple relevance score: {relevance:.2f}")
            
            results.append({
                "query": query,
                "expected": expected,
                "result": result,
                "time": end_time - start_time,
                "relevance": relevance
            })
            
            # Add additional delay after processing to avoid rate limits
            delay = random.uniform(3.0, 5.0)
            print(f"Waiting {delay:.1f} seconds after processing to avoid rate limits...")
            time.sleep(delay)
            
        except Exception as e:
            print(f"Error processing query: {e}")
            results.append({
                "query": query,
                "expected": expected,
                "result": f"Error: {str(e)}",
                "time": 0,
                "relevance": 0
            })
            
            # Add longer delay after error to recover
            print("Rate limit or error detected. Waiting 10 seconds...")
            time.sleep(10)
    
    # Calculate average metrics
    avg_time = sum(r["time"] for r in results) / len(results)
    avg_relevance = sum(r["relevance"] for r in results) / len(results)
    
    print(f"\n--- Evaluation Summary for {name} ---")
    print(f"Total examples: {len(dataset)}")
    print(f"Average processing time: {avg_time:.2f} seconds")
    print(f"Average relevance score: {avg_relevance:.2f}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_file = f"evaluation_results_{name}_{int(time.time())}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Detailed results saved to {results_file}")
    
    return results

def evaluate_rag_chain():
    """Evaluate the RAG chain using local evaluation."""
    print("\n===== EVALUATING RAG CHAIN =====")
    
    try:
        # Initialize components
        print("Initializing components...")
        embeddings_manager = EmbeddingsManager()
        retriever = embeddings_manager.get_retriever()
        
        if not retriever:
            print("Error: Retriever not available. Make sure your vector store is set up.")
            return
        
        llm_manager = LLMManager()
        rag_chain = llm_manager.create_rag_chain(retriever)
        
        # Create dataset
        dataset = create_dataset()
        
        # Create a wrapper for the chain
        def rag_chain_wrapper(query):
            try:
                result = rag_chain.invoke(query)
                # If result is in the format answer='...' extract just the answer
                if isinstance(result, str) and result.startswith("answer='") and result.endswith("'"):
                    result = result[8:-1]  # Remove answer=' and trailing '
                return result
            except Exception as e:
                print(f"Error invoking chain: {e}")
                return f"Error processing query: {str(e)}"
        
        # Run local evaluation
        results = run_local_evaluation(rag_chain_wrapper, dataset, "rag_chain")
        
        print("\n✅ RAG chain evaluation complete!")
        return results
    
    except Exception as e:
        print(f"❌ Error in RAG chain evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_agent_workflow():
    """Evaluate the complete agent workflow using local evaluation."""
    print("\n===== EVALUATING AGENT WORKFLOW =====")
    
    try:
        # Initialize workflow
        print("Initializing agent workflow...")
        workflow = AgentWorkflow()
        workflow.setup()
        
        # Create dataset
        dataset = create_dataset()
        
        # Create a wrapper for the workflow
        def workflow_wrapper(query):
            try:
                result = workflow.process_query(query)
                
                # Extract the final AI response
                final_response = None
                for message in reversed(result["messages"]):
                    if hasattr(message, "type") and message.type == "ai":
                        if not any(phrase in message.content.lower() for phrase in [
                            "transferring back", 
                            "conversation completed", 
                            "please enter your next query"
                        ]):
                            final_response = message.content
                            break
                
                if not final_response:
                    # Look for tool outputs as fallback
                    for message in result["messages"]:
                        if hasattr(message, "type") and message.type == "tool":
                            if "answer=" in message.content:
                                # Extract the answer from answer='...'
                                answer_text = message.content
                                if answer_text.startswith("answer='") and answer_text.endswith("'"):
                                    answer_text = answer_text[8:-1]  # Remove answer=' and trailing '
                                final_response = answer_text
                                break
                
                if final_response:
                    return final_response
                else:
                    return "No appropriate response found"
            except Exception as e:
                print(f"Error processing query: {e}")
                return f"Error processing query: {str(e)}"
        
        # Run local evaluation
        results = run_local_evaluation(workflow_wrapper, dataset, "agent_workflow")
        
        print("\n✅ Agent workflow evaluation complete!")
        return results
    
    except Exception as e:
        print(f"❌ Error in agent workflow evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        # Evaluate the RAG chain
        rag_result = evaluate_rag_chain()
        
        # Evaluate the agent workflow
        agent_result = evaluate_agent_workflow()
        
        print("\n" + "=" * 50)
        print("EVALUATION COMPLETE")
        print("=" * 50)
        print("Local evaluation completed. Results saved to CSV files.")
        print("=" * 50)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user. Partial results may have been saved.")
    except Exception as e:
        print(f"\n\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()