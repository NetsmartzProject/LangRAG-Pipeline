# AI Assistant System

A production-ready multi-agent system with weather and RAG capabilities, built with LangChain, LangGraph, and FastAPI.

## Overview

This system provides a modular, scalable AI assistant that can:
- Answer questions about documents using RAG (Retrieval-Augmented Generation)
- Provide real-time weather information
- Route queries intelligently between specialized agents

## Architecture

The system follows a modular architecture with these key components:

### Core Components
- **FastAPI Backend**: Provides API endpoints for querying the assistant
- **LLM Integration**: Uses Groq (llama3-70b-8192) or OpenAI (gpt-4o) models
- **Vector Database**: Qdrant for document embeddings storage
- **Embeddings**: HuggingFaceBgeEmbeddings for text vectorization
- **Agent System**: LangGraph for orchestrating specialized agents

### Modules
- **Config**: Environment variables and logging setup
- **Schema**: Pydantic models for data validation
- **Utilities**:
  - `embeddings.py`: Vector embeddings and retrieval
  - `llm.py`: LLM model management and chain creation
  - `node.py`: Agent definitions and supervisor
  - `tools.py`: Specialized tools for weather and document queries
  - `workflow.py`: Overall agent workflow orchestration
- **Routes**: API endpoint definitions

## Setup Instructions

### Prerequisites
- Python 3.8+
- Qdrant server running (local or remote)
- API keys for Groq and/or OpenAI
- API key for OpenWeather

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install poetry
poetry install
```

4. Create a `.env` file with your API keys:
```
OPENWEATHER_API_KEY=your_openweather_api_key
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
LLM_MODEL=llama3-70b-8192
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=resume_collection
```

### Running the API

Start the FastAPI server:
```bash
cd <repository-directory>
uvicorn src.main:app --reload
```

The API will be available at http://localhost:8000

### API Endpoints

- **GET /**: Root endpoint
- **GET /chat/health**: Health check endpoint
- **POST /chat/query**: Process a user query
  ```json
  {
    "text": "What skills does Abhinandan Kumar have?"
  }
  ```

## Evaluation and Testing

### LangSmith Evaluation

The system includes LangSmith integration for evaluating LLM responses:

1. Set up LangSmith environment variables:
```powershell
$env:LANGSMITH_TRACING = "true"
$env:LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
$env:LANGSMITH_API_KEY = "your_langsmith_api_key"
$env:LANGSMITH_PROJECT = "rag-evaluation"
```

2. Run the evaluation script:
```powershell
python langsmith_evaluation.py
```

3. View results in the [LangSmith UI](https://smith.langchain.com/projects/rag-evaluation)

### Tracing with LangSmith

For detailed tracing of your application:

1. Set the same environment variables as above
2. Run the tracing example:
```powershell
python trace_full_workflow.py
```

### Running Tests

Run the test suite with pytest:
```bash
pytest tests/
```

## Implementation Details

### Singleton Pattern

The system uses the Singleton pattern for key components to prevent redundant initializations:
- `EmbeddingsManager`: Ensures the embedding model is loaded only once
- `LLMManager`: Ensures the LLM is initialized only once
- `AgentNode`: Ensures agents are created only once
- `AgentWorkflow`: Ensures the workflow is set up only once

### Supervisor Agent

The system uses a LangGraph supervisor agent to route queries:
- Weather-related queries go to the weather agent
- All other queries go to the RAG agent

### RAG Implementation

The RAG system:
1. Retrieves relevant documents from Qdrant
2. Formats documents into a context prompt
3. Sends the context and query to the LLM
4. Returns the response in a standardized format

### Error Handling

The system implements robust error handling:
- Graceful fallbacks when components fail
- Detailed logging for troubleshooting
- API error responses with meaningful messages

## Folder Structure

```
.
├── .env                    # Environment variables
├── README.md               # This file
├── README_LANGSMITH.md     # LangSmith documentation
├── langsmith_evaluation.py # Evaluation script
├── pyproject.toml          # Poetry dependencies
├── requirements.txt        # Pip dependencies
├── run_eval.ps1            # PowerShell evaluation script
├── src/
│   ├── config/             # Configuration
│   │   ├── logger.py       # Logging setup
│   │   └── settings.py     # Environment settings
│   ├── routes/             # API routes
│   │   └── chat.py         # Chat endpoints
│   ├── schema/             # Data models
│   │   └── schema.py       # Pydantic models
│   ├── utills/             # Utility modules
│   │   ├── embeddings.py   # Vector embeddings
│   │   ├── llm.py          # LLM management
│   │   ├── node.py         # Agent definitions
│   │   ├── tools.py        # Tool definitions
│   │   └── workflow.py     # Agent workflow
│   └── main.py             # FastAPI app
└── tests/                  # Test suite
    ├── conftest.py         # Test fixtures
    ├── test_api.py         # API tests
    ├── test_llm.py         # LLM tests
    └── ...                 # Other tests
```

## Future Improvements

- Implement caching for common queries
- Add more specialized agents for different domains
- Implement a request queue for better rate limit handling
- Add more comprehensive evaluation metrics