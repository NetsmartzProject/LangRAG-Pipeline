# Evaluating RAG System with LangSmith

This guide explains how to use LangSmith to evaluate your RAG (Retrieval-Augmented Generation) system.

## Prerequisites

1. A LangSmith account
2. LangSmith API key
3. Python 3.8+
4. Required packages: `langchain`, `langsmith`, `pandas`

## Setup Instructions

### 1. Create a LangSmith Account

1. Go to [LangSmith](https://smith.langchain.com/) and sign up
2. Create a new project called "rag-evaluation"

### 2. Get Your API Key

1. In LangSmith, go to Settings → API Keys
2. Create a new API key
3. Copy the API key for later use

### 3. Update Configuration Files

1. Open `run_eval.ps1` and replace `YOUR_LANGSMITH_API_KEY` with your actual API key
2. Open `langsmith_evaluation.py` and replace `YOUR_LANGSMITH_API_KEY` with your actual API key

## Running the Evaluation

### Using PowerShell Script (Windows)

1. Open PowerShell
2. Navigate to your project directory
3. Run the evaluation script:

```powershell
.\run_eval.ps1
```

### Manual Execution

1. Set environment variables:
   ```
   $env:LANGCHAIN_TRACING_V2 = "true"
   $env:LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
   $env:LANGCHAIN_API_KEY = "your-api-key"
   $env:LANGCHAIN_PROJECT = "rag-evaluation"
   ```

2. Run the evaluation script:
   ```
   python langsmith_evaluation.py
   ```

## Understanding the Evaluation

The evaluation process:

1. Creates a test dataset with sample queries and expected outputs
2. Uploads the dataset to LangSmith
3. Evaluates two components:
   - RAG Chain: The direct retrieval and generation chain
   - Agent Workflow: The full agent system with supervisor and specialized agents

## Evaluation Metrics

The evaluation uses these metrics:

1. **Criteria-based evaluation**: Checks if responses meet specific criteria
   - Relevance: How relevant is the response to the query?
   - Helpfulness: How helpful is the response?
   - Accuracy: How accurate is the information?

2. **Embedding distance**: Measures semantic similarity between responses and expected outputs

## Viewing Results

After running the evaluation:

1. Go to [LangSmith](https://smith.langchain.com/)
2. Navigate to your "rag-evaluation" project
3. View the evaluation results under the "Datasets" and "Evaluations" tabs

## Troubleshooting

If you encounter issues:

1. **API Key errors**: Ensure your API key is correctly set in both files
2. **Import errors**: Make sure all required packages are installed
3. **Connection issues**: Check your internet connection and firewall settings
4. **Missing data**: Ensure your Qdrant collection exists and contains documents

## Next Steps

After reviewing the evaluation results:

1. Identify areas for improvement
2. Adjust your RAG system based on feedback
3. Re-run evaluations to measure improvement
4. Consider creating more comprehensive test datasets