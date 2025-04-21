"""
Pydantic models for structured data input/output and validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class EvidenceSource(BaseModel):
    """Model representing a single piece of evidence."""
    id: str = Field(..., description="Unique identifier for the evidence.")
    source_tool: str = Field(..., description="The tool that retrieved this evidence (e.g., Tavily, Wikidata, Gemini+GoogleSearch).")
    url: Optional[str] = Field(None, description="URL of the source, if applicable.")
    title: Optional[str] = Field(None, description="Title of the source document or page.")
    snippet: Optional[str] = Field(None, description="Relevant snippet or summary from the source.")
    raw_content: Optional[Dict[str, Any]] = Field(None, description="Original raw result from the source API.")
    retrieval_date: Optional[str] = Field(None, description="Timestamp string when the evidence was retrieved.")
    relevance_score: Optional[float] = Field(None, description="Score indicating relevance to the sub-claim or query (0.0 to 1.0).")
    # credibility_score: Optional[float] = Field(None, description="Estimated credibility score of the source (0.0 to 1.0).", ge=0.0, le=1.0)

class GeminiParsedOutput(BaseModel):
    """Model for the structured output expected from parsing the Gemini+GoogleSearch result."""
    summary: str = Field(..., description="Concise summary of the findings related to the claim, based on the search.")
    key_facts: List[str] = Field(default_factory=list, description="List of key facts or pieces of information identified in the search results.")
    sources: List[str] = Field(default_factory=list, description="List of source URLs mentioned or used in the Gemini output.")
    source_tool: str = Field(default="Gemini+GoogleSearch", description="Indicates the source tool.")

class FactCheckResult(BaseModel):
    """Model for the final output of the fact-checking process."""
    claim: str = Field(..., description="The original claim being investigated.")
    claim_id: str = Field(..., description="Unique identifier for this fact-check request.")
    verdict: str = Field(..., description="The final verdict (e.g., TRUE, FALSE, PARTIALLY TRUE/MIXTURE, UNCERTAIN).")
    confidence_score: float = Field(..., description="Confidence score for the verdict (0.0 to 1.0).")
    explanation: str = Field(..., description="Detailed explanation justifying the verdict, referencing evidence.")
    evidence_sources: List[EvidenceSource] = Field(default_factory=list, description="List of evidence sources used in the investigation.")
    # Optional fields for intermediate steps or debugging
    decomposition: Optional[List[str]] = Field(None, description="Sub-questions or entities the claim was decomposed into.")
    timestamp: Optional[str] = Field(None, description="Timestamp string when the fact-check was completed.") 