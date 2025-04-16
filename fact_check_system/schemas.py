"""
Pydantic models for the fact-checking system.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Annotated, Literal
from typing_extensions import TypedDict
from datetime import datetime
from langgraph.graph import add_messages

# Basic types
EntityType = Dict[str, Any]
SourceType = Dict[str, Any]
EvidenceType = Dict[str, Any]

class AtomicClaim(BaseModel):
    """An atomic verifiable unit extracted from a complex claim."""
    id: str
    statement: str
    entities: List[str] = Field(default_factory=list)
    time_references: Optional[List[str]] = Field(default_factory=list)
    location_references: Optional[List[str]] = Field(default_factory=list)
    numeric_values: Optional[List[str]] = Field(default_factory=list)
    importance: float = Field(default=1.0, ge=0.0, le=1.0, description="Relative importance of this atomic claim")

class Evidence(BaseModel):
    """Evidence collected to verify a claim."""
    id: str
    source: str = Field(description="Source of the evidence (e.g., 'web', 'wikidata', 'news')")
    url: Optional[str] = None
    content: str
    retrieval_date: datetime = Field(default_factory=datetime.now)
    credibility_score: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)

class EvidenceCollection(BaseModel):
    """Collection of evidence for a specific claim."""
    claim_id: str
    search_results: List[Evidence] = Field(default_factory=list)
    knowledge_base: List[Evidence] = Field(default_factory=list)
    news_sources: List[Evidence] = Field(default_factory=list)

class VerificationResult(BaseModel):
    """Result of claim verification by a single agent."""
    agent_id: str
    verdict: Literal["TRUE", "FALSE", "PARTIALLY_TRUE", "UNVERIFIABLE", "NEEDS_CONTEXT"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evidence_assessment: Dict[str, float] = Field(description="Assessment of each evidence piece")
    reasoning_difficulty: float = Field(default=0.0, ge=0.0, le=1.0)
    claim_ambiguity: float = Field(default=0.0, ge=0.0, le=1.0)

class ClaimVerificationResults(BaseModel):
    """Compilation of verification results from multiple agents for a single claim."""
    claim_id: str
    primary: VerificationResult
    cross: Optional[VerificationResult] = None
    historical: Optional[VerificationResult] = None
    reconciled: Optional[VerificationResult] = None

class AtomicClaimList(BaseModel):
    """A container for a list of atomic claims."""
    claims: List[AtomicClaim]

class FinalVerdict(BaseModel):
    """Final verdict for a claim with confidence scoring and evidence summary."""
    claim_id: str
    claim_text: str
    verdict: Literal["TRUE", "FALSE", "PARTIALLY_TRUE", "UNVERIFIABLE", "NEEDS_CONTEXT"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evidence_summary: str
    sources_used: List[Dict[str, Any]] = Field(default_factory=list, description="Sources used to verify this claim")
    verification_date: datetime = Field(default_factory=datetime.now)

class ComprehensiveVerdict(BaseModel):
    """Comprehensive verdict for the original input claim, synthesizing atomic claim verdicts."""
    original_claim: str
    atomic_verdicts: Dict[str, FinalVerdict]
    overall_verdict: Literal["TRUE", "FALSE", "PARTIALLY_TRUE", "UNVERIFIABLE", "NEEDS_CONTEXT"]
    overall_confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    summary: str
    process_steps: List[Dict[str, Any]] = Field(default_factory=list, description="Steps taken during the fact-checking process")
    sources_used: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, description="Sources used for each claim")
    verification_date: datetime = Field(default_factory=datetime.now)

# LangGraph state for workflow orchestration
class FactCheckState(BaseModel):
    """State model for the fact-checking workflow."""
    input_claim: str
    decomposed_claims: List[AtomicClaim] = Field(default_factory=list)
    evidence_collections: Dict[str, EvidenceCollection] = Field(default_factory=dict)
    verification_results: Dict[str, ClaimVerificationResults] = Field(default_factory=dict)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    final_verdict: Optional[ComprehensiveVerdict] = None
    messages: Annotated[List, add_messages] = Field(default_factory=list)
    intermediate_steps: List[Dict[str, Any]] = Field(default_factory=list, description="Track all intermediate steps in the fact-checking process")
    sources_used: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict, description="Track all sources used for each claim") 