# Testing and Evaluation

This folder contains tests and evaluation scripts for the AI Assistant project.

## Test Structure

The tests are organized as follows:

- `test_api.py`: Tests for API endpoints
- `test_llm.py`: Tests for LLM manager and chain functionality
- `test_tools.py`: Tests for weather and document tools
- `test_workflow.py`: Tests for agent workflow

## LangSmith Evaluation

For more comprehensive evaluation of the LLM responses, we use LangSmith. This is set up in the root directory:

- `langsmith_evaluation.py`: Script to evaluate RAG chain and agent workflow
- `run_eval.ps1`: PowerShell script to run the evaluation
- `README_LANGSMITH.md`: Documentation for LangSmith evaluation

## Running Tests

To run the tests:

```bash
pytest tests/
```

To run a specific test file:

```bash
pytest tests/test_api.py
```

## Running LangSmith Evaluation

To evaluate the system using LangSmith:

```powershell
.\run_eval.ps1
```

See `README_LANGSMITH.md` for more details on LangSmith evaluation.