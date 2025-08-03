# AI Engineer Assistant

An intelligent agent system that can answer questions about weather and documents using LangChain, LangGraph, and Groq LLM.

## Features

- Weather information retrieval using OpenWeatherMap API
- Document question answering using RAG (Retrieval Augmented Generation)
- Supervisor agent that routes queries to specialized agents
- Streamlit web interface for easy interaction

## Requirements

- Python 3.9+
- Poetry for dependency management
- Qdrant running locally or remotely for vector storage
- Groq API key
- OpenWeatherMap API key

## Installation

1. Clone the repository
2. Install dependencies with Poetry:

```bash
poetry install
```

3. Create a `.env` file based on the `.env.example` template:

```bash
cp .env.example .env
```

4. Edit the `.env` file with your API keys and configuration

## Usage

1. Start the Streamlit application:

```bash
poetry run streamlit run main.py
```

2. Upload a resume PDF file (optional)
3. Ask questions about weather or the uploaded resume

## Project Structure

- `main.py`: Streamlit application entry point
- `src/config/`: Configuration settings and logging
- `src/schema/`: Pydantic models for structured data
- `src/utills/`: Utility modules
  - `embeddings.py`: Vector embeddings and document processing
  - `llm.py`: LLM initialization and chain creation
  - `node.py`: Agent node definitions
  - `tools.py`: Tool definitions for agents
  - `workflow.py`: Agent workflow management

