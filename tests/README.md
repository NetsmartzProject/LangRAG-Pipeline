# AI Assistant Tests

This directory contains tests for the AI Assistant application. The tests cover API handling, LLM processing, retrieval logic, and LangSmith evaluation.

## Test Structure

- `conftest.py`: Contains pytest fixtures used across multiple test files
- `test_api.py`: Tests for the FastAPI endpoints
- `test_llm.py`: Tests for the LLM manager and RAG chain
- `test_tools.py`: Tests for the weather and document tools
- `test_workflow.py`: Tests for the agent workflow
- `test_langsmith_evaluation.py`: Tests for LangSmith evaluation
- `test_streamlit_app.py`: Tests for the Streamlit application

## Running Tests

To run the tests, use the following command:

```bash
pytest tests/
```

To run a specific test file:

```bash
pytest tests/test_api.py
```

To run tests with coverage:

```bash
pytest tests/ --cov=src
```

## LangSmith Evaluation

The LangSmith evaluation tests require a LangSmith API key. Set the following environment variables:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
export LANGCHAIN_API_KEY=your_langchain_api_key
export LANGCHAIN_PROJECT=ai-assistant-evaluation
```

## Creating Evaluation Datasets

To create evaluation datasets for LangSmith:

1. Create a dataset with example queries and expected responses
2. Upload the dataset to LangSmith
3. Run the evaluation tests

Example dataset format:

```json
[
  {
    "input": "What's the weather in London?",
    "expected_output": "The weather in London is..."
  },
  {
    "input": "What skills does Abhinandan have?",
    "expected_output": "Abhinandan has skills in..."
  }
]
```