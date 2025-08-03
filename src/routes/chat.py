import time
from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config.logger import logger
from utills.workflow import AgentWorkflow
from schema.schema import Message, ConversationResponse

router = APIRouter()

# Initialize workflow
workflow = None

def get_workflow():
    """Get or initialize the workflow"""
    global workflow
    if workflow is None:
        workflow = AgentWorkflow()
        workflow.setup()
    return workflow

class Query(BaseModel):
    text: str = Field(..., description="The query text")

@router.post("/query", response_model=ConversationResponse)
async def process_query(query: Query):
    """Process a user query"""
    try:
        wf = get_workflow()
        
        # Process the query
        start_time = time.time()
        result = wf.process_query(query.text)
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Extract messages
        messages = wf.extract_conversation(result)
        
        return ConversationResponse(
            messages=[Message(role=msg["role"], content=msg["content"]) for msg in messages],
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        wf = get_workflow()
        return {
            "status": "healthy",
            "workflow_initialized": wf is not None,
            "collection_name": wf.collection_name if wf else None
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }