"""
Main workflow for the fact-checking system.
"""

import time
import re
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

from .config import (
    get_primary_llm,
    get_secondary_llm,
    get_tavily_client,
    get_duckduckgo_client,
    get_news_api_client,
    get_wikidata_client,
    get_entity_details,
    HIGH_CONFIDENCE_THRESHOLD,
    MEDIUM_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD
)
from .prompts import (
    claim_decomposition_prompt_template,
    fact_verification_prompt_template,
    answer_formatting_prompt_template,
    credibility_assessment_prompt_template,
    source_reliability_prompt_template,
    search_query_reformulation_prompt_template,
    final_judgment_prompt_template
)
from .tools import (
    tavily_search_tool,
    duckduckgo_search_tool,
    news_api_search_tool,
    wikidata_tool,
    entity_extraction_tool,
    date_extraction_tool,
    url_extraction_tool
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FactCheckResult(BaseModel):
    """Result of a fact-checking operation."""
    claim: str = Field(..., description="The original claim that was fact-checked")
    verdict: str = Field(..., description="The overall verdict (TRUE/FALSE/PARTIALLY TRUE/UNCERTAIN)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    explanation: str = Field(..., description="Detailed explanation of the fact-check")
    evidence: List[Dict[str, Any]] = Field(..., description="Evidence sources used")
    sub_claims: Optional[List[Dict[str, Any]]] = Field(None, description="Decomposed sub-claims if applicable")

def fact_check_claim(claim: str, search_depth: int = 3, decompose: bool = True) -> FactCheckResult:
    """
    Perform a comprehensive fact-check on the given claim.
    
    Args:
        claim: The claim to fact-check
        search_depth: Number of search results to consider per search engine
        decompose: Whether to decompose complex claims into sub-claims
        
    Returns:
        A FactCheckResult containing the verdict, confidence, explanation, and evidence
    """
    logger.info(f"Starting fact-check for claim: {claim}")
    
    # Step 1: Decompose the claim into simpler, atomic sub-claims if needed
    sub_claims = []
    if decompose:
        llm = get_primary_llm()
        decomposition_chain = (
            claim_decomposition_prompt_template
            | llm
            | StrOutputParser()
        )
        
        decomposition_result = decomposition_chain.invoke({"claim": claim})
        
        # Parse the decomposition result
        try:
            # Try to parse as JSON first
            if decomposition_result.strip().startswith("[") or decomposition_result.strip().startswith("{"):
                parsed_result = json.loads(decomposition_result)
                if isinstance(parsed_result, list):
                    sub_claims = parsed_result
                else:
                    sub_claims = parsed_result.get("sub_claims", [])
            else:
                # Otherwise, try to extract sub-claims using regex
                matches = re.findall(r"(?:^|\n)(\d+\.\s*.*?)(?=\n\d+\.|\n\n|$)", decomposition_result, re.DOTALL)
                if matches:
                    sub_claims = [{"text": m.strip(), "id": i + 1} for i, m in enumerate(matches)]
        except Exception as e:
            logger.warning(f"Failed to parse sub-claims: {e}")
            # Fall back to treating the entire claim as a single claim
            sub_claims = []
    
    # If no sub-claims were successfully extracted or decomposition is disabled,
    # treat the entire claim as one claim
    if not sub_claims:
        logger.info("Processing claim as a single statement (no decomposition)")
        sub_claims = [{"text": claim, "id": 1}]
    else:
        logger.info(f"Decomposed claim into {len(sub_claims)} sub-claims")
    
    # Process each sub-claim
    sub_claim_results = []
    for sub_claim_info in sub_claims:
        sub_claim = sub_claim_info["text"]
        sub_claim_id = sub_claim_info.get("id", 0)
        
        logger.info(f"Processing sub-claim {sub_claim_id}: {sub_claim}")
        
        # Extract key information from the claim
        entities = entity_extraction_tool(sub_claim)
        dates = date_extraction_tool(sub_claim)
        urls = url_extraction_tool(sub_claim)
        
        logger.info(f"Extracted entities: {entities}")
        logger.info(f"Extracted dates: {dates}")
        logger.info(f"Extracted URLs: {urls}")
        
        # Step 2: Generate search queries based on the claim
        llm = get_primary_llm()
        query_generation_chain = (
            search_query_reformulation_prompt_template
            | llm
            | StrOutputParser()
        )
        
        search_query_input = {
            "claim": sub_claim,
            "entities": entities,
            "dates": dates,
            "urls": urls
        }
        
        search_queries_text = query_generation_chain.invoke(search_query_input)
        
        # Parse the generated search queries
        search_queries = []
        try:
            # Try to parse as JSON first
            if search_queries_text.strip().startswith("["):
                search_queries = json.loads(search_queries_text)
            else:
                # Otherwise, try to extract queries using regex
                matches = re.findall(r"(?:^|\n)(?:\d+\.\s*|\")(.*?)(?:\"|(?=\n\d+\.|\n\n|$))", search_queries_text, re.DOTALL)
                if matches:
                    search_queries = [m.strip() for m in matches if m.strip()]
        except Exception as e:
            logger.warning(f"Failed to parse search queries: {e}")
            # Fall back to using the original claim as the query
            search_queries = [sub_claim]
        
        if not search_queries:
            search_queries = [sub_claim]
        
        logger.info(f"Generated search queries: {search_queries}")
        
        # Step 3: Gather evidence from multiple sources
        all_search_results = []
        
        # Use the first search query for primary searches
        primary_query = search_queries[0]
        
        # Tavily search
        try:
            tavily_results = tavily_search_tool(primary_query, search_depth)
            for result in tavily_results:
                result["source"] = "Tavily"
            all_search_results.extend(tavily_results)
            logger.info(f"Retrieved {len(tavily_results)} results from Tavily")
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
        
        # DuckDuckGo search
        try:
            ddg_results = duckduckgo_search_tool(primary_query, search_depth)
            for result in ddg_results:
                result["source"] = "DuckDuckGo"
            all_search_results.extend(ddg_results)
            logger.info(f"Retrieved {len(ddg_results)} results from DuckDuckGo")
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
        
        # If we have entities, try to get information from Wikidata
        wiki_results = []
        try:
            for entity in entities[:3]:  # Limit to top 3 entities
                wiki_data = wikidata_tool(entity, 2)
                if wiki_data:
                    wiki_results.extend(wiki_data)
            
            # Add source information
            for result in wiki_results:
                result["source"] = "Wikidata"
            
            all_search_results.extend(wiki_results)
            logger.info(f"Retrieved {len(wiki_results)} results from Wikidata")
        except Exception as e:
            logger.error(f"Wikidata search failed: {e}")
        
        # News API search for recent information if dates indicate recency
        try:
            # Check if the claim might be about recent events
            recent_dates = [d for d in dates if "2023" in d or "2024" in d]
            if recent_dates or any(term in sub_claim.lower() for term in ["recent", "latest", "new", "current"]):
                news_results = news_api_search_tool(primary_query, search_depth)
                for result in news_results:
                    result["source"] = "NewsAPI"
                all_search_results.extend(news_results)
                logger.info(f"Retrieved {len(news_results)} results from NewsAPI")
        except Exception as e:
            logger.error(f"NewsAPI search failed: {e}")
        
        # For more comprehensive searches, use additional queries if available
        if len(search_queries) > 1:
            for query in search_queries[1:3]:  # Limit to next 2 queries
                try:
                    # Use DuckDuckGo for additional queries instead of Yahoo News
                    additional_results = duckduckgo_search_tool(query, max_results=2)
                    for result in additional_results:
                        result["source"] = "DuckDuckGo"
                    all_search_results.extend(additional_results)
                    logger.info(f"Retrieved {len(additional_results)} results from DuckDuckGo for query: {query}")
                except Exception as e:
                    logger.error(f"Additional search failed: {e}")
        
        # If we have very few results, try direct URL extraction
        if len(all_search_results) < 3 and urls:
            logger.info(f"Few search results, attempting to use extracted URLs directly: {urls}")
            # Logic for handling URLs could be added here
        
        # Deduplicate search results based on URL
        unique_urls = set()
        deduplicated_results = []
        
        for result in all_search_results:
            if "url" in result and result["url"] not in unique_urls:
                unique_urls.add(result["url"])
                deduplicated_results.append(result)
        
        all_search_results = deduplicated_results
        logger.info(f"Total unique search results after deduplication: {len(all_search_results)}")
        
        # Step 4: Assess the credibility of each source
        llm = get_secondary_llm()
        credibility_chain = (
            credibility_assessment_prompt_template
            | llm
            | StrOutputParser()
        )
        
        reliability_chain = (
            source_reliability_prompt_template
            | llm
            | StrOutputParser()
        )
        
        # Process each search result to assess credibility
        for i, result in enumerate(all_search_results):
            try:
                # Skip credibility assessment for Wikidata results
                if result.get("source") == "Wikidata":
                    result["credibility_score"] = 0.9  # Wikidata is generally reliable
                    result["credibility_explanation"] = "Wikidata is a structured knowledge base and generally reliable for factual information."
                    continue
                
                # Basic source reliability check
                source_url = result.get("url", "")
                source_name = result.get("source_name", result.get("source", "Unknown"))
                
                reliability_input = {
                    "source_url": source_url,
                    "source_name": source_name
                }
                
                reliability_assessment = reliability_chain.invoke(reliability_input)
                
                # Parse reliability score
                reliability_score = 0.5  # Default moderate reliability
                reliability_match = re.search(r"reliability_score:\s*(0\.\d+)", reliability_assessment)
                if reliability_match:
                    reliability_score = float(reliability_match.group(1))
                
                # Content credibility assessment
                content = result.get("content", "")
                if not content and "snippet" in result:
                    content = result["snippet"]
                
                credibility_input = {
                    "claim": sub_claim,
                    "content": content,
                    "source_url": source_url,
                    "source_name": source_name,
                    "reliability_score": reliability_score
                }
                
                credibility_assessment = credibility_chain.invoke(credibility_input)
                
                # Parse credibility score
                credibility_score = 0.5  # Default moderate credibility
                credibility_match = re.search(r"credibility_score:\s*(0\.\d+)", credibility_assessment)
                if credibility_match:
                    credibility_score = float(credibility_match.group(1))
                
                # Extract explanation if available
                explanation = ""
                explanation_match = re.search(r"explanation:(.*?)(?=\n\n|$)", credibility_assessment, re.DOTALL)
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                
                # Store the credibility assessment
                result["credibility_score"] = credibility_score
                result["credibility_explanation"] = explanation
                
                # Calculate relevance to the claim
                result["relevance_score"] = min(1.0, 0.3 + credibility_score * 0.7)  # Weighted relevance based on credibility
                
                logger.info(f"Assessed credibility for result {i+1}: score={credibility_score:.2f}")
            
            except Exception as e:
                logger.error(f"Error during credibility assessment for result {i+1}: {e}")
                # Assign default values if assessment fails
                result["credibility_score"] = 0.5
                result["credibility_explanation"] = "Credibility assessment failed."
                result["relevance_score"] = 0.5
        
        # Sort results by credibility and relevance
        all_search_results.sort(key=lambda x: (x.get("credibility_score", 0) + x.get("relevance_score", 0)), reverse=True)
        
        # Select top results for fact verification (max 8 to avoid context limitations)
        top_results = all_search_results[:8]
        
        # Step 5: Verify the claim against the evidence
        llm = get_primary_llm()
        verification_chain = (
            fact_verification_prompt_template
            | llm
            | StrOutputParser()
        )
        
        verification_input = {
            "claim": sub_claim,
            "search_results": top_results
        }
        
        verification_result = verification_chain.invoke(verification_input)
        logger.info(f"Completed verification for sub-claim {sub_claim_id}")
        
        # Step 6: Extract the verdict and confidence
        verdict = "UNCERTAIN"
        confidence = 0.0
        explanation = ""
        
        # Try to parse the structured output
        try:
            # First check for JSON format
            if verification_result.strip().startswith("{") and verification_result.strip().endswith("}"):
                parsed_result = json.loads(verification_result)
                verdict = parsed_result.get("verdict", "UNCERTAIN")
                confidence = float(parsed_result.get("confidence", 0.5))
                explanation = parsed_result.get("explanation", "")
            else:
                # Otherwise use regex to extract fields
                verdict_match = re.search(r"verdict:\s*(TRUE|FALSE|PARTIALLY TRUE|UNCERTAIN)", verification_result, re.IGNORECASE)
                if verdict_match:
                    verdict = verdict_match.group(1).upper()
                
                confidence_match = re.search(r"confidence:\s*(0\.\d+)", verification_result)
                if confidence_match:
                    confidence = float(confidence_match.group(1))
                
                explanation_match = re.search(r"explanation:(.*?)(?=\n\n|$)", verification_result, re.DOTALL)
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
        except Exception as e:
            logger.warning(f"Failed to parse verification result: {e}")
            # Use the full verification result as the explanation
            explanation = verification_result
        
        # Store the sub-claim result
        sub_claim_result = {
            "id": sub_claim_id,
            "text": sub_claim,
            "verdict": verdict,
            "confidence": confidence,
            "explanation": explanation,
            "evidence": top_results
        }
        
        sub_claim_results.append(sub_claim_result)
    
    # Step 7: Make a final judgment if we have multiple sub-claims
    if len(sub_claim_results) > 1:
        llm = get_primary_llm()
        final_judgment_chain = (
            final_judgment_prompt_template
            | llm
            | StrOutputParser()
        )
        
        judgment_input = {
            "claim": claim,
            "sub_claim_results": sub_claim_results
        }
        
        final_judgment = final_judgment_chain.invoke(judgment_input)
        
        # Parse the final judgment
        overall_verdict = "UNCERTAIN"
        overall_confidence = 0.0
        overall_explanation = ""
        
        try:
            # First check for JSON format
            if final_judgment.strip().startswith("{") and final_judgment.strip().endswith("}"):
                parsed_judgment = json.loads(final_judgment)
                overall_verdict = parsed_judgment.get("verdict", "UNCERTAIN")
                overall_confidence = float(parsed_judgment.get("confidence", 0.5))
                overall_explanation = parsed_judgment.get("explanation", "")
            else:
                # Otherwise use regex to extract fields
                verdict_match = re.search(r"verdict:\s*(TRUE|FALSE|PARTIALLY TRUE|UNCERTAIN)", final_judgment, re.IGNORECASE)
                if verdict_match:
                    overall_verdict = verdict_match.group(1).upper()
                
                confidence_match = re.search(r"confidence:\s*(0\.\d+)", final_judgment)
                if confidence_match:
                    overall_confidence = float(confidence_match.group(1))
                
                explanation_match = re.search(r"explanation:(.*?)(?=\n\n|$)", final_judgment, re.DOTALL)
                if explanation_match:
                    overall_explanation = explanation_match.group(1).strip()
        except Exception as e:
            logger.warning(f"Failed to parse final judgment: {e}")
            # Use the full judgment text as the explanation
            overall_explanation = final_judgment
    else:
        # If we only have one sub-claim, use its results directly
        overall_verdict = sub_claim_results[0]["verdict"]
        overall_confidence = sub_claim_results[0]["confidence"]
        overall_explanation = sub_claim_results[0]["explanation"]
    
    # Step 8: Format the final answer
    llm = get_secondary_llm()
    formatting_chain = (
        answer_formatting_prompt_template
        | llm
        | StrOutputParser()
    )
    
    formatting_input = {
        "claim": claim,
        "verdict": overall_verdict,
        "confidence": overall_confidence,
        "explanation": overall_explanation,
        "sub_claim_results": sub_claim_results if len(sub_claim_results) > 1 else []
    }
    
    formatted_answer = formatting_chain.invoke(formatting_input)
    
    # Ensure the explanation is properly formatted
    if not overall_explanation or overall_explanation == "":
        overall_explanation = formatted_answer
    
    # Gather all evidence sources
    all_evidence = []
    for sub_result in sub_claim_results:
        all_evidence.extend(sub_result["evidence"])
    
    # Create and return the final result
    result = FactCheckResult(
        claim=claim,
        verdict=overall_verdict,
        confidence=overall_confidence,
        explanation=overall_explanation,
        evidence=all_evidence,
        sub_claims=sub_claim_results if len(sub_claim_results) > 1 else None
    )
    
    logger.info(f"Completed fact-check for claim with verdict: {overall_verdict}, confidence: {overall_confidence:.2f}")
    
    return result

# Add process_claim function for use in manual_query.py
def process_claim(claim: str) -> Dict[str, Any]:
    """
    Process a claim for fact-checking (used by manual_query.py).
    
    Args:
        claim: The claim to fact-check
        
    Returns:
        Dictionary containing verdict and explanation
    """
    logger.info(f"Processing claim: {claim}")
    
    try:
        # Perform fact-checking
        result = fact_check_claim(claim)
        
        # Convert to dictionary format expected by manual_query.py
        verdict_dict = {
            "claim": result.claim,
            "verdict": result.verdict,
            "confidence": result.confidence,
            "explanation": result.explanation,
            "evidence": result.evidence,
            "sub_claims": result.sub_claims if result.sub_claims else []
        }
        
        return verdict_dict
    except Exception as e:
        logger.error(f"Error processing claim: {e}")
        # Return a default error response
        return {
            "claim": claim,
            "verdict": "ERROR",
            "confidence": 0.0,
            "explanation": f"An error occurred during fact-checking: {str(e)}",
            "evidence": [],
            "sub_claims": []
        } 