import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.routes.chat import router as chat_router
from config.logger import logger

# Create FastAPI app
app = FastAPI(title="AI Assistant API", description="API for weather and document queries")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(chat_router, prefix="/chat", tags=["chat"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AI Assistant API is running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)