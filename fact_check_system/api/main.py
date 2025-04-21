"""
API endpoints for the Fact Checking System.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, status # Added BackgroundTasks, status
from pydantic import BaseModel
import uvicorn
import sys
import os
import logging # Added for logging
from typing import List, Dict, Any, Optional # Added Optional
import uuid # Added for task IDs

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import necessary components from your fact-checking logic
from fact_check_system.fact_checker import run_fact_check
from fact_check_system.models import FactCheckResult, EvidenceSource

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tathya Fact Checking System API",
    description="API for verifying claims using various sources and LLMs.",
    version="0.1.0",
)

# In-memory storage for task results (Replace with Redis/Celery/DB for production)
task_results: Dict[str, Any] = {}


class ClaimRequest(BaseModel):
    claim: str
    language: str = "en"  # Default language

class ClaimResponse(BaseModel):
    claim: str
    result: str # e.g., "True", "False", "Uncertain"
    confidence_score: float
    explanation: str
    sources: List[Dict[str, Any]] = [] # Make sure this is List[Dict[str, Any]]

# New response model for initiating a task
class TaskResponse(BaseModel):
    task_id: str
    message: str

# New response model for task status/result
class ResultResponse(BaseModel):
    status: str # e.g., "processing", "completed", "error"
    result: Optional[ClaimResponse] = None # Include result if completed
    error_detail: Optional[str] = None # Include error detail if failed


# --- Helper function to run the check in the background ---
def run_background_fact_check(task_id: str, claim: str, language: str):
    """Runs the fact check and stores the result."""
    logger.info(f"Background task started for task_id: {task_id}, claim: '{claim}'")
    try:
        fact_check_result: FactCheckResult = run_fact_check(claim)
        logger.info(f"Background task completed for task_id: {task_id}. Verdict: {fact_check_result.verdict}")

        # --- Rebuild the source list correctly as List[Dict[str, Any]] ---
        response_sources = []
        if fact_check_result.evidence_sources:
            for source in fact_check_result.evidence_sources:
                source_dict = {}
                if isinstance(source, EvidenceSource):
                    source_dict['url'] = getattr(source, 'url', None)
                    source_dict['title'] = getattr(source, 'title', getattr(source, 'url', 'No Title'))
                    source_dict['snippet'] = getattr(source, 'snippet', 'No snippet available')
                    tool_name = getattr(source, 'source_tool', 'Unknown')
                    # Simplified tool name mapping for brevity
                    if tool_name.startswith('Tavily'): tool_name = 'Tavily Search'
                    elif tool_name == 'Wikidata': tool_name = 'Wikidata'
                    elif tool_name == 'DuckDuckGo': tool_name = 'DuckDuckGo'
                    elif tool_name.startswith('Gemini'): tool_name = 'Google AI'
                    source_dict['tool'] = tool_name
                elif isinstance(source, dict):
                    source_dict['url'] = source.get('url')
                    source_dict['title'] = source.get('title', source.get('url', 'No Title'))
                    source_dict['snippet'] = source.get('snippet', 'No snippet available')
                    tool_name = source.get('source_tool', 'Unknown')
                    if tool_name.startswith('Tavily'): tool_name = 'Tavily Search'
                    elif tool_name == 'Wikidata': tool_name = 'Wikidata'
                    elif tool_name == 'DuckDuckGo': tool_name = 'DuckDuckGo'
                    elif tool_name.startswith('Gemini'): tool_name = 'Google AI'
                    source_dict['tool'] = tool_name
                elif isinstance(source, str) and source.startswith('http'):
                    tool_name = "Web Source"
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(source).netloc
                        if domain: tool_name = domain.replace('www.', '')
                    except: pass
                    source_dict['url'] = source
                    source_dict['title'] = source
                    source_dict['snippet'] = 'No preview available'
                    source_dict['tool'] = tool_name
                else:
                    logger.warning(f"Task {task_id}: Skipping source of unexpected type: {type(source)}")
                    continue

                if source_dict.get('url'):
                    response_sources.append(source_dict)
                else:
                     logger.warning(f"Task {task_id}: Skipping source with no URL: {source_dict.get('title')}")

        # Prepare response data
        response_data = {
            "claim": claim,
            "result": fact_check_result.verdict or "Uncertain",
            "confidence_score": fact_check_result.confidence_score or 0.0,
            "explanation": fact_check_result.explanation or "No explanation provided.",
            "sources": response_sources
        }

        # Store the final result
        task_results[task_id] = ResultResponse(status="completed", result=ClaimResponse(**response_data))
        logger.info(f"Result stored for task_id: {task_id}")

    except Exception as e:
        logger.error(f"Error during background task for task_id {task_id}, claim '{claim}': {e}", exc_info=True)
        # Store error information
        task_results[task_id] = ResultResponse(status="error", error_detail=f"Internal server error during fact-checking: {str(e)}")


# --- API Endpoints ---\

@app.post("/check", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def check_claim_endpoint(request: ClaimRequest, background_tasks: BackgroundTasks):
    """
    Receives a claim, starts the fact-checking process in the background,
    and returns a task ID.
    """
    logger.info(f"Received claim: '{request.claim}' with language '{request.language}'. Starting background task.")
    task_id = str(uuid.uuid4())


    background_tasks.add_task(run_background_fact_check, task_id, request.claim, request.language)


    task_results[task_id] = ResultResponse(status="processing")

    return TaskResponse(task_id=task_id, message="Fact-checking process started.")


@app.get("/results/{task_id}", response_model=ResultResponse)
async def get_results_endpoint(task_id: str):
    """
    Retrieves the status or result of a fact-checking task.
    """
    logger.debug(f"Checking results for task_id: {task_id}")
    result = task_results.get(task_id)

    if not result:
        logger.warning(f"Task ID not found: {task_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task ID not found")

    logger.debug(f"Current status for task_id {task_id}: {result.status}")
    # Optionally remove completed/error tasks after retrieval? Or implement TTL?
    # if result.status in ["completed", "error"]:
    #     del task_results[task_id] # Be careful with concurrent requests

    return result


@app.get("/")
async def root():
    """
    Root endpoint providing a welcome message.
    """
    return {"message": "Welcome to the Tathya Fact Checking System API!"}


if __name__ == "__main__":
    print("Starting API server with background task support...")
    uvicorn.run("fact_check_system.api.main:app", host="127.0.0.1", port=8000, reload=True) 