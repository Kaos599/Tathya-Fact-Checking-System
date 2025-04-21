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
        # Run the fact check using the updated function
        fact_check_result: FactCheckResult = run_fact_check(claim)
        logger.info(f"Background task completed for task_id: {task_id}. Verdict: {fact_check_result.verdict}")

        # --- Rebuild the source list correctly as List[Dict[str, Any]] --- #
        # The result from run_fact_check should now be a FactCheckResult object
        # containing EvidenceSource objects (or similar dicts) with potentially
        # pre-populated source_tool info.
        response_sources = []
        if fact_check_result.evidence_sources:
            for source in fact_check_result.evidence_sources:
                source_dict = {}
                tool_name = "Unknown"
                title = "No Title"
                snippet = "No snippet available"
                url = None

                if isinstance(source, EvidenceSource):
                    url = getattr(source, 'url', None)
                    title = getattr(source, 'title', url or 'No Title')
                    snippet = getattr(source, 'snippet', 'No snippet available')
                    # Use the source_tool if provided by the agent
                    tool_name = getattr(source, 'source_tool', tool_name)
                    # Simple heuristic if still unknown
                    if tool_name == "Unknown" and url and "tavily" in url:
                        tool_name = "Tavily Search"
                    elif tool_name == "Unknown" and url and "wikidata" in url:
                        tool_name = "Wikidata"
                    elif tool_name == "Unknown" and url and "duckduckgo" in url:
                        tool_name = "DuckDuckGo"
                    # Add more heuristics if needed

                elif isinstance(source, dict):
                    url = source.get('url')
                    title = source.get('title', url or 'No Title')
                    snippet = source.get('snippet', 'No snippet available')
                    # Use the source_tool if provided
                    tool_name = source.get('source_tool', tool_name)
                    # Heuristics again
                    if tool_name == "Unknown" and url and "tavily" in url:
                        tool_name = "Tavily Search"
                    elif tool_name == "Unknown" and url and "wikidata" in url:
                        tool_name = "Wikidata"
                    elif tool_name == "Unknown" and url and "duckduckgo" in url:
                        tool_name = "DuckDuckGo"

                elif isinstance(source, str) and source.startswith('http'): # Handle plain URLs if they sneak through
                    url = source
                    title = source
                    snippet = 'No preview available'
                    tool_name = "Web Source"
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(source).netloc
                        if domain: tool_name = domain.replace('www.', '')
                    except: pass
                else:
                    logger.warning(f"Task {task_id}: Skipping source of unexpected type: {type(source)}")
                    continue

                if url:
                    source_dict['url'] = url
                    source_dict['title'] = title
                    source_dict['snippet'] = snippet
                    source_dict['tool'] = tool_name # Assign the determined tool name
                    response_sources.append(source_dict)
                else:
                     logger.warning(f"Task {task_id}: Skipping source with no URL: {title}")
        else:
            logger.info(f"Task {task_id}: No evidence sources found in the FactCheckResult.")

        # Prepare response data using the FactCheckResult fields
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