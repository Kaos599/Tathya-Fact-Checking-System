"""
API endpoints for the Fact Checking System.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sys
import os
import logging # Added for logging
from typing import List, Dict, Any # Ensure Any is imported

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


class ClaimRequest(BaseModel):
    claim: str
    language: str = "en"  # Default language

class ClaimResponse(BaseModel):
    claim: str
    result: str # e.g., "True", "False", "Uncertain"
    confidence_score: float
    explanation: str
    sources: List[Dict[str, Any]] = [] # Make sure this is List[Dict[str, Any]]


# --- API Endpoints ---

@app.post("/check", response_model=ClaimResponse)
async def check_claim_endpoint(request: ClaimRequest):
    """
    Receives a claim and returns a fact-checking result.
    """
    logger.info(f"Received claim: '{request.claim}' with language '{request.language}'") # Basic logging

    try:
        fact_check_result: FactCheckResult = run_fact_check(request.claim)

        logger.info(f"Fact check completed for claim: '{request.claim}'. Verdict: {fact_check_result.verdict}")

        # --- Rebuild the source list correctly as List[Dict[str, Any]] --- 
        response_sources = []
        if fact_check_result.evidence_sources:
            for source in fact_check_result.evidence_sources:
                source_dict = {}
                if isinstance(source, EvidenceSource):
                    # Map attributes from EvidenceSource object
                    source_dict['url'] = getattr(source, 'url', None)
                    source_dict['title'] = getattr(source, 'title', getattr(source, 'url', 'No Title')) # Use URL as fallback title
                    source_dict['snippet'] = getattr(source, 'snippet', 'No snippet available')
                    
                    # Get a better tool name if possible
                    tool_name = getattr(source, 'source_tool', 'Unknown')
                    if tool_name == 'Unknown' and hasattr(source, 'raw_content') and source.raw_content:
                        # Try to get tool from raw_content
                        if isinstance(source.raw_content, dict):
                            tool_name = source.raw_content.get('source_tool', tool_name)
                    
                    # Make generic names more specific
                    if tool_name == 'Tavily':
                        tool_name = 'Tavily Search'
                    elif tool_name == 'Wikidata':
                        tool_name = 'Wikidata'
                    elif tool_name == 'DuckDuckGo':
                        tool_name = 'DuckDuckGo'
                    elif tool_name.startswith('Gemini'):
                        tool_name = 'Google AI'
                        
                    source_dict['tool'] = tool_name
                    
                elif isinstance(source, dict):
                    # Map keys from the dictionary source
                    source_dict['url'] = source.get('url')
                    source_dict['title'] = source.get('title', source.get('url', 'No Title')) # Use URL as fallback title
                    source_dict['snippet'] = source.get('snippet', 'No snippet available')
                    
                    # Get a better tool name if possible
                    tool_name = source.get('source_tool', 'Unknown')
                    
                    # Make generic names more specific
                    if tool_name == 'Tavily':
                        tool_name = 'Tavily Search'
                    elif tool_name == 'Wikidata':
                        tool_name = 'Wikidata'
                    elif tool_name == 'DuckDuckGo':
                        tool_name = 'DuckDuckGo'
                    elif tool_name.startswith('Gemini'):
                        tool_name = 'Google AI'
                        
                    source_dict['tool'] = tool_name
                    
                elif isinstance(source, str) and source.startswith('http'):
                    # Handle case where source is just a URL string
                    # Try to extract domain name as the tool
                    tool_name = "Web Source"
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(source).netloc
                        if domain:
                            if 'wikipedia.org' in domain:
                                tool_name = 'Wikipedia'
                            elif 'wikidata.org' in domain:
                                tool_name = 'Wikidata'
                            elif 'google.com' in domain:
                                tool_name = 'Google'
                            else:
                                tool_name = domain.replace('www.', '')
                    except:
                        pass
                        
                    source_dict['url'] = source
                    source_dict['title'] = source
                    source_dict['snippet'] = 'No preview available'
                    source_dict['tool'] = tool_name
                else:
                    # Skip if source is neither EvidenceSource nor dict nor URL string
                    logger.warning(f"Skipping source of unexpected type: {type(source)}")
                    continue
                
                # Only add the source if it has a URL
                if source_dict.get('url'):
                    response_sources.append(source_dict)
                else:
                    logger.warning(f"Skipping source with no URL: {source_dict.get('title')}")

        # Prepare response data with the list of dictionaries
        response_data = {
            "claim": request.claim,
            "result": fact_check_result.verdict or "Uncertain", 
            "confidence_score": fact_check_result.confidence_score or 0.0,
            "explanation": fact_check_result.explanation or "No explanation provided.",
            "sources": response_sources # Ensure this is the list of dicts
        }
        
        # Validate and return the response
        # FastAPI automatically validates against ClaimResponse model here
        return ClaimResponse(**response_data)

    except Exception as e:
        logger.error(f"Error during fact-checking for claim '{request.claim}': {e}", exc_info=True) # Log traceback
        raise HTTPException(status_code=500, detail=f"Internal server error during fact-checking: {str(e)}")

@app.get("/")
async def root():
    """
    Root endpoint providing a welcome message.
    """
    return {"message": "Welcome to the Tathya Fact Checking System API!"}

# --- Running the API (for local development) ---

if __name__ == "__main__":
    print("Starting API server...")
    # Ensure the script is run from the project root or adjust path accordingly
    uvicorn.run("fact_check_system.api.main:app", host="127.0.0.1", port=8000, reload=True) 