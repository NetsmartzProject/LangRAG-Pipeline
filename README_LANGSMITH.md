# LangSmith Evaluation Guide

This guide explains how to use LangSmith to evaluate the AI Assistant's performance.

## Prerequisites

1. A LangSmith account (sign up at [smith.langchain.com](https://smith.langchain.com/))
2. Your LangSmith API key
3. Python packages: `langchain`, `langsmith`, `pandas`

## Setup

### 1. Set Environment Variables

In PowerShell:

```powershell
$env:LANGCHAIN_TRACING_V2 = "true"
$env:LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
$env:LANGCHAIN_API_KEY = "your_api_key_here"
$env:LANGCHAIN_PROJECT = "ai-assistant-evaluation"
```

Or use the provided script:

```powershell
.\run_langsmith_eval.ps1
```

### 2. Run the Evaluation

```powershell
python langsmith_evaluation.py
```

## What the Evaluation Does

The script evaluates two components:

1. **RAG Chain Evaluation**:
   - Tests how well the retrieval-augmented generation works
   - Evaluates relevance, correctness, and helpfulness of responses
   - Checks if the retrieved context is relevant to the query

2. **Agent Workflow Evaluation**:
   - Tests the complete agent system including routing and tool use
   - Evaluates accuracy, completeness, relevance, and hallucination avoidance
   - Checks if the system routes queries to the appropriate agent

## Viewing Results

1. Go to [LangSmith](https://smith.langchain.com/)
2. Navigate to your project (default: "ai-assistant-evaluation")
3. View datasets and evaluations
4. Click on individual runs to see detailed metrics and traces

## Evaluation Metrics

### RAG Chain Metrics
- **QA Relevance**: Is the answer relevant to the question?
- **Answer Relevance**: Is the answer relevant in general?
- **Context Relevance**: Is the retrieved context relevant?
- **Helpfulness**: How helpful is the response?
- **Correctness**: Is the information factually correct?

### Agent Workflow Metrics
- **Criteria**: Custom evaluation based on specific criteria
- **QA Correctness**: Is the answer correct?
- **Helpfulness**: How helpful is the response?
- **Relevance**: Is the response relevant to the query?

## Customizing the Evaluation

You can modify `langsmith_evaluation.py` to:
- Add more test examples
- Change evaluation metrics
- Adjust custom evaluation criteria
- Test different components of the system

## Troubleshooting

- **Authentication errors**: Check your API key and environment variables
- **"Dataset already exists"**: The script handles this by creating timestamped datasets
- **Slow evaluation**: LangSmith evaluations can take time, especially with many examples
- **Errors in evaluation**: Check the console output for detailed error messages