"""
Workflow orchestration for the fact-checking system using LangGraph.
"""

import uuid
from typing import Dict, List, Any, Tuple, Annotated
from langgraph.graph import StateGraph, add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime

from fact_check_system.schemas import (
    FactCheckState, 
    AtomicClaim, 
    Evidence,
    EvidenceCollection,
    VerificationResult,
    ClaimVerificationResults,
    FinalVerdict,
    ComprehensiveVerdict
)
from fact_check_system.agents import (
    extract_claims_agent,
    decompose_claims_agent,
    primary_verification_agent,
    cross_verification_agent,
    adjudication_agent,
    synthesis_agent
)
from fact_check_system.tools import (
    tavily_search,
    wikidata_entity_search,
    news_search,
    scrape_webpage
)

def create_fact_check_graph():
    """
    Create the main LangGraph for fact checking.
    
    Returns:
        A compiled StateGraph for fact checking.
    """
    # Initialize the graph
    graph = StateGraph(FactCheckState)
    
    # Define nodes
    
    # Node 1: Decompose the input claim into atomic verifiable units
    def decompose_claims(state: FactCheckState):
        """
        Break down the input claim into atomic verifiable units.
        """
        # First, extract factual claims if not already provided
        extracted_claims = extract_claims_agent(state.input_claim)
        
        # Process each extracted claim
        all_atomic_claims = []
        for extracted_claim in extracted_claims:
            # Decompose into atomic claims
            atomic_claims = decompose_claims_agent(extracted_claim)
            all_atomic_claims.extend(atomic_claims)
        
        # Add system message about completed decomposition
        messages = [
            SystemMessage(content=f"Claim has been decomposed into {len(all_atomic_claims)} atomic verifiable units.")
        ]
        
        # Track intermediate step
        intermediate_step = {
            "step": "decompose_claims",
            "timestamp": datetime.now().isoformat(),
            "extracted_claims": extracted_claims,
            "atomic_claims": [claim.model_dump() for claim in all_atomic_claims]
        }
        
        # Return updated state
        return {
            "decomposed_claims": all_atomic_claims,
            "messages": messages,
            "intermediate_steps": state.intermediate_steps + [intermediate_step]
        }
    
    # Node 2: Gather evidence for each atomic claim
    def retrieve_evidence(state: FactCheckState):
        """
        Gather evidence for each atomic claim from multiple sources.
        """
        evidence_collections = {}
        sources_used = {}
        
        for claim in state.decomposed_claims:
            claim_id = claim.id
            search_query = claim.statement
            
            # Enhance search query with entities and references
            if claim.entities:
                search_query += f" {' '.join(claim.entities)}"
            
            # Initialize evidence collection
            evidence_collection = EvidenceCollection(claim_id=claim_id)
            
            # Track sources for this claim
            claim_sources = []
            
            # 1. Web search via Tavily
            search_results = tavily_search(search_query)
            evidence_collection.search_results = search_results
            
            # Track web search sources
            for result in search_results:
                claim_sources.append({
                    "source_type": "web_search",
                    "source": result.get("source", "web"),
                    "url": result.get("url", ""),
                    "retrieval_date": result.get("retrieval_date", datetime.now()).isoformat() if not isinstance(result.get("retrieval_date", ""), str) else result.get("retrieval_date"),
                    "relevance_score": result.get("relevance_score", 0.5)
                })
            
            # 2. Knowledge base search via Wikidata
            # Search for each entity in the claim
            kb_results = []
            for entity in claim.entities:
                kb_results.extend(wikidata_entity_search(entity))
            evidence_collection.knowledge_base = kb_results
            
            # Track knowledge base sources
            for result in kb_results:
                claim_sources.append({
                    "source_type": "knowledge_base",
                    "source": "Wikidata",
                    "entity": entity,
                    "retrieval_date": result.get("retrieval_date", datetime.now()).isoformat() if not isinstance(result.get("retrieval_date", ""), str) else result.get("retrieval_date"),
                    "relevance_score": result.get("relevance_score", 0.5)
                })
            
            # 3. News search if the claim has time references
            if claim.time_references:
                time_query = f"{search_query} {' '.join(claim.time_references)}"
                news_results = news_search(time_query)
                evidence_collection.news_sources = news_results
                
                # Track news sources
                for result in news_results:
                    claim_sources.append({
                        "source_type": "news",
                        "source": result.get("source", "news"),
                        "url": result.get("url", ""),
                        "retrieval_date": result.get("retrieval_date", datetime.now()).isoformat() if not isinstance(result.get("retrieval_date", ""), str) else result.get("retrieval_date"),
                        "publication_date": result.get("publication_date", datetime.now()).isoformat() if not isinstance(result.get("publication_date", ""), str) else result.get("publication_date"),
                        "relevance_score": result.get("relevance_score", 0.5)
                    })
            
            # Store the evidence collection and sources used
            evidence_collections[claim_id] = evidence_collection.model_dump()
            sources_used[claim_id] = claim_sources
        
        # Add system message about evidence retrieval
        messages = [
            SystemMessage(content=f"Evidence has been collected for {len(evidence_collections)} claims from multiple sources.")
        ]
        
        # Track intermediate step
        intermediate_step = {
            "step": "retrieve_evidence",
            "timestamp": datetime.now().isoformat(),
            "evidence_by_claim": {claim_id: {"count": len(sources)} for claim_id, sources in sources_used.items()}
        }
        
        # Return updated state
        return {
            "evidence_collections": evidence_collections,
            "messages": messages,
            "intermediate_steps": state.intermediate_steps + [intermediate_step],
            "sources_used": sources_used
        }
    
    # Node 3: Verify claims using multiple agents
    def verify_claims(state: FactCheckState):
        """
        Verify each atomic claim using multiple specialized agents.
        """
        verification_results = {}
        confidence_scores = {}
        verification_steps = {}
        
        for claim in state.decomposed_claims:
            claim_id = claim.id
            if claim_id not in state.evidence_collections:
                continue
            
            # Convert to dict if it's a model instance, otherwise use as is
            evidence_collection_data = state.evidence_collections[claim_id]
            
            # Check if it's a dictionary or EvidenceCollection instance
            if isinstance(evidence_collection_data, dict):
                # Dict access
                evidence_by_source = {
                    "search_results": evidence_collection_data.get("search_results", []),
                    "knowledge_base": evidence_collection_data.get("knowledge_base", []),
                    "news_sources": evidence_collection_data.get("news_sources", [])
                }
            else:
                # Model attribute access
                evidence_by_source = {
                    "search_results": evidence_collection_data.search_results,
                    "knowledge_base": evidence_collection_data.knowledge_base,
                    "news_sources": evidence_collection_data.news_sources
                }
            
            # Run primary verification
            primary_result = primary_verification_agent(claim, evidence_by_source)
            
            # Run cross-verification
            cross_result = cross_verification_agent(claim, evidence_by_source)
            
            # Store the verification results
            verification_results[claim_id] = ClaimVerificationResults(
                claim_id=claim_id,
                primary=primary_result,
                cross=cross_result
            ).model_dump()
            
            # Calculate initial confidence score (average of both agents)
            confidence_scores[claim_id] = (primary_result.confidence + cross_result.confidence) / 2
            
            # Track verification steps for this claim
            verification_steps[claim_id] = {
                "claim": claim.statement,
                "primary_result": {
                    "verdict": primary_result.verdict,
                    "confidence": primary_result.confidence,
                    "reasoning": primary_result.reasoning
                },
                "cross_result": {
                    "verdict": cross_result.verdict,
                    "confidence": cross_result.confidence,
                    "reasoning": cross_result.reasoning
                },
                "combined_confidence": confidence_scores[claim_id]
            }
        
        # Add system message about verification
        messages = [
            SystemMessage(content=f"Verification completed for {len(verification_results)} claims using multiple agents.")
        ]
        
        # Track intermediate step
        intermediate_step = {
            "step": "verify_claims",
            "timestamp": datetime.now().isoformat(),
            "verification_details": verification_steps
        }
        
        # Return updated state
        return {
            "verification_results": verification_results,
            "confidence_scores": confidence_scores,
            "messages": messages,
            "intermediate_steps": state.intermediate_steps + [intermediate_step]
        }
    
    # Node 4: Adjudicate and produce final verdicts for each atomic claim
    def adjudicate_results(state: FactCheckState):
        """
        Produce final verdicts for each atomic claim.
        """
        atomic_verdicts = {}
        adjudication_steps = {}
        
        for claim in state.decomposed_claims:
            claim_id = claim.id
            if claim_id not in state.verification_results:
                continue
            
            # Extract verification results
            verification_results = state.verification_results[claim_id]
            
            # Convert dict back to model objects if it's a dictionary
            from fact_check_system.schemas import VerificationResult
            results = {}
            if isinstance(verification_results, dict):
                # Process dictionary
                for k, v in verification_results.items():
                    if k != "claim_id":  # Exclude non-result fields
                        results[k] = VerificationResult(**v) if isinstance(v, dict) else v
            else:
                # Handle Pydantic model directly
                results = {
                    "primary": verification_results.primary,
                    "cross": verification_results.cross
                }
            
            # Run adjudication
            final_verdict = adjudication_agent(claim, results)
            
            # Add sources used for this claim if available
            if claim_id in state.sources_used:
                final_verdict.sources_used = state.sources_used[claim_id]
            
            # Store the final verdict
            atomic_verdicts[claim_id] = final_verdict
            
            # Track adjudication steps
            adjudication_steps[claim_id] = {
                "claim": claim.statement,
                "final_verdict": final_verdict.verdict,
                "confidence": final_verdict.confidence,
                "reasoning": final_verdict.reasoning,
                "evidence_summary": final_verdict.evidence_summary
            }
        
        # Synthesize a comprehensive verdict
        comprehensive_verdict = synthesis_agent(state.input_claim, atomic_verdicts)
        
        # Add process steps and sources to the comprehensive verdict
        comprehensive_verdict.process_steps = state.intermediate_steps
        comprehensive_verdict.sources_used = state.sources_used
        
        # Add system message about adjudication
        messages = [
            SystemMessage(content=f"Final verdict generated with {comprehensive_verdict.overall_verdict} at {comprehensive_verdict.overall_confidence:.2f} confidence.")
        ]
        
        # Track intermediate step
        intermediate_step = {
            "step": "adjudicate_results",
            "timestamp": datetime.now().isoformat(),
            "adjudication_details": adjudication_steps,
            "overall_verdict": {
                "verdict": comprehensive_verdict.overall_verdict,
                "confidence": comprehensive_verdict.overall_confidence,
                "reasoning": comprehensive_verdict.reasoning,
                "summary": comprehensive_verdict.summary
            }
        }
        
        # Return updated state
        return {
            "final_verdict": comprehensive_verdict.model_dump(),
            "messages": messages,
            "intermediate_steps": state.intermediate_steps + [intermediate_step]
        }
    
    # Add nodes to the graph
    graph.add_node("decompose_claims", decompose_claims)
    graph.add_node("retrieve_evidence", retrieve_evidence)
    graph.add_node("verify_claims", verify_claims)
    graph.add_node("adjudicate_results", adjudicate_results)
    
    # Define edges (the flow of the graph)
    graph.add_edge("decompose_claims", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "verify_claims")
    graph.add_edge("verify_claims", "adjudicate_results")
    
    # Set the entry point
    graph.set_entry_point("decompose_claims")
    
    # Compile the graph
    return graph.compile()

def process_claim(claim: str) -> Dict[str, Any]:
    """
    Process a claim through the fact-checking pipeline.
    
    Args:
        claim: The claim to verify
        
    Returns:
        Dictionary with the verdict and supporting information
    """
    # Create the workflow graph
    workflow = create_fact_check_graph()
    
    # Set up initial state
    config = {"input_claim": claim, "intermediate_steps": [], "sources_used": {}}
    
    # Execute the workflow
    result = workflow.invoke(config)
    
    # Extract the final verdict
    final_verdict = result.get("final_verdict", {})
    
    # Add intermediate steps and sources to the verdict for transparency
    final_verdict["intermediate_steps"] = result.get("intermediate_steps", [])
    final_verdict["sources_used"] = result.get("sources_used", {})
    
    return final_verdict 