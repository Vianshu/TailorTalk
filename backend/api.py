"""
FastAPI Backend for Titanic Chat Agent.

Provides REST API endpoints for the Streamlit frontend to communicate
with the LangChain agent.
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent import run_agent_query, load_dataset, _build_schema as get_dataset_info

# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TailorTalk — Titanic Chat Agent API",
    description="A conversational AI agent for analyzing the Titanic dataset",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""
    answer: str
    visualization: Optional[str] = None  # Base64 encoded image


class DatasetInfoResponse(BaseModel):
    """Response body for dataset info endpoint."""
    info: str
    row_count: int
    column_count: int
    columns: list[str]


class HealthResponse(BaseModel):
    """Response body for health check."""
    status: str
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="TailorTalk Titanic Chat Agent API is running! 🚢",
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="API is running",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message through the LangChain agent.

    Accepts a natural language question about the Titanic dataset
    and returns an answer with optional visualization.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        result = run_agent_query(request.message)
        return ChatResponse(
            answer=result["answer"],
            visualization=result.get("visualization"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}",
        )


@app.get("/dataset/info", response_model=DatasetInfoResponse)
async def dataset_info():
    """Get information about the Titanic dataset."""
    try:
        df = load_dataset()
        info = get_dataset_info()
        return DatasetInfoResponse(
            info=info,
            row_count=len(df),
            column_count=len(df.columns),
            columns=df.columns.tolist(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading dataset: {str(e)}",
        )


@app.get("/dataset/sample")
async def dataset_sample():
    """Get a sample of the Titanic dataset."""
    try:
        df = load_dataset()
        return df.head(10).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading dataset: {str(e)}",
        )
