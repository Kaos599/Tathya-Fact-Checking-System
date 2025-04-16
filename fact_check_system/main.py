"""
Main module for the fact-checking system.
"""

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging
import os
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from fact_check_system.workflow import process_claim
from fact_check_system.schemas import ComprehensiveVerdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Tathya Fact-Checking System",
    description="An advanced agentic fact-checking system with confidence scoring",
    version="1.0.0"
)

# Define API models
class FactCheckRequest(BaseModel):
    """Request model for fact checking."""
    claim: str
    background_processing: bool = False

class FactCheckResponse(BaseModel):
    """Response model for fact checking."""
    request_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None

class FactCheckStatus(BaseModel):
    """Status model for background fact checking."""
    request_id: str
    status: str
    
# Store for background processing tasks
processing_tasks = {}

def background_fact_check(request_id: str, claim: str):
    """
    Perform fact checking in the background.
    
    Args:
        request_id: Unique request ID
        claim: The claim to verify
    """
    start_time = time.time()
    
    try:
        # Update status to processing
        processing_tasks[request_id]["status"] = "processing"
        
        # Process the claim
        result = process_claim(claim)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update status to completed
        processing_tasks[request_id].update({
            "status": "completed",
            "result": result,
            "processing_time": processing_time
        })
        
        logger.info(f"Completed fact checking for request {request_id} in {processing_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {str(e)}")
        processing_tasks[request_id].update({
            "status": "failed",
            "error": str(e)
        })

@app.get("/")
def read_root():
    """Root endpoint."""
    return {
        "name": "Tathya Fact-Checking System",
        "version": "1.0.0",
        "description": "An advanced agentic fact-checking system with confidence scoring"
    }

@app.post("/check", response_model=FactCheckResponse)
async def check_fact(request: FactCheckRequest, background_tasks: BackgroundTasks):
    """
    Check a factual claim.
    
    Args:
        request: The request containing the claim to verify
        background_tasks: FastAPI background tasks
        
    Returns:
        Fact checking result or status
    """
    # Generate a unique request ID
    request_id = f"req_{int(time.time())}_{hash(request.claim) % 10000}"
    
    if request.background_processing:
        # Initialize task status
        processing_tasks[request_id] = {
            "status": "queued",
            "claim": request.claim,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add background task
        background_tasks.add_task(background_fact_check, request_id, request.claim)
        
        logger.info(f"Queued request {request_id} for background processing")
        
        return FactCheckResponse(
            request_id=request_id,
            status="queued"
        )
    
    else:
        # Process the claim synchronously
        start_time = time.time()
        
        try:
            logger.info(f"Processing request {request_id}")
            result = process_claim(request.claim)
            processing_time = time.time() - start_time
            
            logger.info(f"Completed fact checking for request {request_id} in {processing_time:.2f} seconds")
            
            return FactCheckResponse(
                request_id=request_id,
                status="completed",
                result=result,
                processing_time=processing_time
            )
        
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{request_id}", response_model=FactCheckResponse)
async def check_status(request_id: str):
    """
    Check the status of a background fact checking request.
    
    Args:
        request_id: The unique request ID
        
    Returns:
        Current status and result if available
    """
    if request_id not in processing_tasks:
        raise HTTPException(status_code=404, detail="Request not found")
    
    task = processing_tasks[request_id]
    
    response = FactCheckResponse(
        request_id=request_id,
        status=task["status"]
    )
    
    if task["status"] == "completed":
        response.result = task["result"]
        response.processing_time = task.get("processing_time")
    
    return response

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

def main():
    """Run the API server."""
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "fact_check_system.main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )

if __name__ == "__main__":
    main() 