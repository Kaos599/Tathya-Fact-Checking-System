"""
Main workflow for the fact-checking system.
"""

import time
import re
import json
import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
import os

from .config import (
    get_primary_llm,
    get_secondary_llm,
    get_gemini_llm,
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
    final_judgment_prompt_template,
    gemini_cross_check_prompt_template,
    question_generation_prompt_template,
    question_answering_prompt_template,
    verdict_synthesis_prompt_template
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
    cross_check_verdict: Optional[str] = Field(None, description="Verdict from Gemini cross-check")
    cross_check_confidence: Optional[float] = Field(None, description="Confidence from Gemini cross-check")
    cross_check_explanation: Optional[str] = Field(None, description="Explanation from Gemini cross-check")

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
            tavily_results = tavily_search_tool(primary_query, max_results=search_depth)
            for result in tavily_results:
                result["source"] = "Tavily"
            all_search_results.extend(tavily_results)
            logger.info(f"Retrieved {len(tavily_results)} results from Tavily")
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
        
        # DuckDuckGo search
        try:
            ddg_results = duckduckgo_search_tool(primary_query, max_results=search_depth)
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
                news_results = news_api_search_tool(primary_query, max_results=search_depth)
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
        
        # Log the input being sent to the verification LLM
        logger.debug(f"--- Verification Input for Sub-claim {sub_claim_id} ---")
        logger.debug(f"Claim: {sub_claim}")
        # Log evidence carefully, potentially large
        try:
            evidence_preview = json.dumps(top_results, indent=2, default=datetime_serializer)[:2000] # Log preview
            logger.debug(f"Evidence (Preview):\n{evidence_preview}...") 
        except Exception as log_e:
            logger.debug(f"Evidence (Preview): [Error logging evidence: {log_e}]")
        logger.debug("--------------------------------------------------")
        
        verification_input = {
            "claim": sub_claim,
            "search_results": top_results
        }
        
        # Direct print statements for debugging (guaranteed to appear in console)
        print("\n" + "="*80)
        print(f"SENDING TO PRIMARY LLM FOR VERIFICATION:")
        print(f"CLAIM: {sub_claim}")
        print("-"*40)
        print(f"EVIDENCE SAMPLE (first 2 results only):")
        if len(top_results) > 0:
            try:
                print(f"Result 1: {top_results[0].get('title', 'No title')} | URL: {top_results[0].get('url', 'No URL')}")
                print(f"Content snippet: {top_results[0].get('content', top_results[0].get('snippet', 'No content'))[:200]}...")
            except Exception as e:
                print(f"Error displaying first result: {e}")
                
            if len(top_results) > 1:
                try:
                    print(f"Result 2: {top_results[1].get('title', 'No title')} | URL: {top_results[1].get('url', 'No URL')}")
                    print(f"Content snippet: {top_results[1].get('content', top_results[1].get('snippet', 'No content'))[:200]}...")
                except Exception as e:
                    print(f"Error displaying second result: {e}")
        else:
            print("No evidence available!")
        print("="*80 + "\n")
        
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
        
        # Adjust confidence if UNCERTAIN and 0.0
        if verdict == "UNCERTAIN" and confidence == 0.0:
            confidence = 0.1 # Set to a low non-zero value
            logger.info("Adjusted 0.0 confidence for UNCERTAIN verdict to 0.1")
        
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
    
    # Adjust final confidence if UNCERTAIN and 0.0 (after potential judgment)
    if overall_verdict == "UNCERTAIN" and overall_confidence == 0.0:
        overall_confidence = 0.1 # Set to a low non-zero value
        logger.info("Adjusted final 0.0 confidence for UNCERTAIN verdict to 0.1")
    
    # Helper function to serialize datetime objects for JSON
    def datetime_serializer(obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    # Step 7.5: Perform Gemini Cross-Check
    cross_check_verdict = None
    cross_check_confidence = None
    cross_check_explanation = None
    try:
        gemini_llm = get_gemini_llm()
        cross_check_chain = (
            gemini_cross_check_prompt_template
            | gemini_llm
            | StrOutputParser()
        )
        
        # Gather all evidence sources before formatting
        all_evidence = []
        for sub_result in sub_claim_results:
            if "evidence" in sub_result:
                all_evidence.extend(sub_result["evidence"])
        
        # Prepare evidence for the prompt (limit length if necessary)
        try:
            evidence_str = json.dumps(all_evidence, indent=2, default=datetime_serializer)[:10000] # Limit evidence length
        except TypeError as e:
            logger.error(f"Failed to serialize evidence for Gemini: {e}. Trying basic serialization.")
            # Fallback: attempt basic serialization, might still fail if other types exist
            evidence_str = json.dumps([str(item) for item in all_evidence])[:10000] 
        
        cross_check_input = {
            "claim": claim,
            "evidence": evidence_str,
            "initial_verdict": overall_verdict,
            "initial_explanation": overall_explanation
        }
        
        # Direct print statements for debugging (guaranteed to appear in console)
        print("\n" + "="*80)
        print(f"SENDING TO GEMINI FOR CROSS-CHECK:")
        print(f"CLAIM: {claim}")
        print(f"INITIAL VERDICT: {overall_verdict}")
        print(f"INITIAL EXPLANATION: {overall_explanation[:300]}...")
        print("-"*40)
        print(f"EVIDENCE SAMPLE (first 200 chars):")
        if evidence_str:
            print(f"{evidence_str[:200]}...")
        else:
            print("No evidence string available!")
        print("="*80 + "\n")
        
        logger.info("Performing Gemini cross-check...")
        cross_check_result_raw = cross_check_chain.invoke(cross_check_input)
        logger.info("Completed Gemini cross-check.")
        
        # Parse Gemini's response (similar to parsing verification/judgment)
        try:
            if cross_check_result_raw.strip().startswith("{") and cross_check_result_raw.strip().endswith("}"):
                parsed_cross_check = json.loads(cross_check_result_raw)
                cross_check_verdict = parsed_cross_check.get("verdict", "UNCERTAIN")
                cross_check_confidence = float(parsed_cross_check.get("confidence", 0.5))
                cross_check_explanation = parsed_cross_check.get("explanation", "")
                logger.info(f"Gemini cross-check result: Verdict={cross_check_verdict}, Confidence={cross_check_confidence:.2f}")
            else:
                logger.warning("Gemini cross-check did not return valid JSON. Using raw output as explanation.")
                # Simplified fallback: Use raw output and default values
                cross_check_verdict = "UNCERTAIN" # Default verdict if parsing fails
                cross_check_confidence = 0.5 # Default confidence if parsing fails
                cross_check_explanation = cross_check_result_raw # Store raw output
                logger.info(f"Gemini cross-check fallback: Verdict={cross_check_verdict}, Confidence={cross_check_confidence:.2f}")
        except Exception as parse_e:
            logger.warning(f"Failed to parse Gemini cross-check result: {parse_e}")
            cross_check_explanation = cross_check_result_raw
            
    except Exception as gemini_e:
        logger.error(f"Error during Gemini cross-check: {gemini_e}")
        cross_check_explanation = f"Gemini cross-check failed: {str(gemini_e)}"
    
    # Step 8: Format the final answer
    # llm = get_secondary_llm() # Removed: No longer needed for formatting
    
    # Create and return the final result
    result = FactCheckResult(
        claim=claim,
        verdict=overall_verdict,
        confidence=overall_confidence,
        explanation=overall_explanation,
        evidence=all_evidence,
        sub_claims=sub_claim_results if len(sub_claim_results) > 1 else None,
        cross_check_verdict=cross_check_verdict,
        cross_check_confidence=cross_check_confidence,
        cross_check_explanation=cross_check_explanation
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
        result = fact_check_claim_v2(claim)
        
        # Convert to dictionary format expected by manual_query.py
        verdict_dict = {
            "original_claim": claim,
            "claim": result.claim,
            "verdict": result.verdict,
            "confidence": result.confidence,
            "explanation": result.explanation,
            "evidence": result.evidence,
            "sub_claims": result.sub_claims if result.sub_claims else [],
            "cross_check_verdict": result.cross_check_verdict,
            "cross_check_confidence": result.cross_check_confidence,
            "cross_check_explanation": result.cross_check_explanation
        }
        
        return verdict_dict
    except Exception as e:
        logger.error(f"Error processing claim: {e}", exc_info=True)
        # Return a default error response with a user-friendly message
        error_message = "An error occurred during fact-checking. This might be due to an API failure or network issue."
        
        # Include more details if in development environment
        if os.getenv("ENVIRONMENT") == "development":
            error_message += f" Technical details: {str(e)}"
            
        return {
            "original_claim": claim,
            "claim": claim,
            "verdict": "ERROR",
            "confidence": 0.0,
            "explanation": error_message,
            "evidence": [],
            "sub_claims": [],
            "cross_check_verdict": None,
            "cross_check_confidence": None,
            "cross_check_explanation": None
        }

# ----------------------------------------------------------------------------------
# New Question‑Driven Fact‑Checking Pipeline (v2)
# ----------------------------------------------------------------------------------

def fact_check_claim_v2(claim: str, search_depth: int = 5, num_questions: int = 10) -> FactCheckResult:
    """
    Overhauled pipeline implementing the following steps:
      1) Interpret the claim and pose *exactly* `num_questions` investigative questions.
      2) For each question, generate search queries, gather evidence from Wikidata, DuckDuckGo, Tavily and others.
      3) Answer each question individually based on the evidence.
      4) After all questions are answered, synthesise an overall verdict & justification.
      5) Optionally perform Gemini cross‑check (reuse existing logic).
    """
    logger.info("Starting V2 fact‑check for claim: %s", claim)

    primary_llm = get_primary_llm()

    # STEP 1: Generate investigative questions
    question_chain = (
        question_generation_prompt_template
        | primary_llm
        | StrOutputParser()
    )
    questions_raw = question_chain.invoke({"claim": claim})

    # Parse questions (expect numbered list)
    questions: List[str] = []
    try:
        if questions_raw.strip().startswith("["):
            questions = json.loads(questions_raw)
        else:
            matches = re.findall(r"(?:^|\n)\d+\.\s*(.*?)(?=\n\d+\.|$)", questions_raw, re.DOTALL)
            questions = [q.strip() for q in matches if q.strip()]
    except Exception as e:
        logger.warning("Failed to parse questions: %s", e)

    # Fallback: ensure list length
    if not questions:
        questions = [questions_raw.strip()]
    if len(questions) > num_questions:
        questions = questions[:num_questions]
    while len(questions) < num_questions:
        questions.append(questions[-1])  # duplicate last to keep length (should rarely happen)

    logger.info("Generated %d investigative questions", len(questions))

    secondary_llm = get_secondary_llm()

    qa_pairs = []  # store answers and evidence per question
    all_evidence = []

    # Helper to gather evidence for a query
    def gather_evidence(query: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        # Tavily
        try:
            tav = tavily_search_tool(query, max_results=search_depth)
            for r in tav: r["source"] = "Tavily"
            results.extend(tav)
        except Exception as e:
            logger.error("Tavily search error: %s", e)
        # DuckDuckGo
        try:
            ddg = duckduckgo_search_tool(query, max_results=search_depth)
            for r in ddg: r["source"] = "DuckDuckGo"
            results.extend(ddg)
        except Exception as e:
            logger.error("DDG search error: %s", e)
        # Wikidata
        try:
            wiki = wikidata_tool(query, max_results=search_depth)
            for r in wiki: r["source"] = "Wikidata"
            results.extend(wiki)
        except Exception as e:
            logger.error("Wikidata search error: %s", e)
        # Deduplicate by URL
        seen = set()
        dedup = []
        for r in results:
            url = r.get("url")
            if url and url not in seen:
                dedup.append(r)
                seen.add(url)
        return dedup[:search_depth*3]  # limit

    # Process each question
    for idx, question in enumerate(questions, start=1):
        logger.info("Processing question %d: %s", idx, question)

        # STEP 2a: Generate search queries for the question
        reform_chain = (
            search_query_reformulation_prompt_template
            | primary_llm
            | StrOutputParser()
        )
        q_queries_text = reform_chain.invoke({
            "claim": question,
            "entities": [],
            "dates": [],
            "urls": []
        })
        q_queries: List[str] = []
        try:
            if q_queries_text.strip().startswith("["):
                q_queries = json.loads(q_queries_text)
            else:
                q_queries = re.findall(r"(?:^|\n)\d+\.\s*(.*?)(?=\n\d+\.|$)", q_queries_text, re.DOTALL)
                q_queries = [s.strip() for s in q_queries if s.strip()]
        except Exception as e:
            logger.warning("Failed to parse search queries for question %d: %s", idx, e)

        if not q_queries:
            q_queries = [question]

        # STEP 3: Retrieve evidence for each query (limit to first 2 queries to save cost)
        evidence_for_question = []
        for q in q_queries[:2]:
            evidence_for_question.extend(gather_evidence(q))

        # Keep top N by credibility + relevance
        evidence_for_question.sort(key=lambda x: (x.get("credibility_score", 0)+x.get("relevance_score", 0)), reverse=True)
        evidence_snippets = evidence_for_question[:8]
        all_evidence.extend(evidence_snippets)

        # Prepare evidence string (truncate to avoid context blow‑up)
        def safe_serialize(obj):
            if isinstance(obj, datetime.datetime):
                return obj.isoformat()
            return str(obj)
        ev_str = json.dumps(evidence_snippets, default=safe_serialize)[:6000]

        # STEP 4: Answer the question using evidence
        qa_chain = (
            question_answering_prompt_template
            | secondary_llm
            | StrOutputParser()
        )
        answer_raw = qa_chain.invoke({
            "question": question,
            "evidence": ev_str
        })

        # Parse answer JSON
        answer_json = {
            "answer": answer_raw,
            "verdict_component": "INSUFFICIENT",
            "confidence": 0.0,
            "explanation": "Parsing failed"
        }
        try:
            if answer_raw.strip().startswith("{"):
                answer_json = json.loads(answer_raw)
        except Exception as e:
            logger.warning("Failed to parse answer JSON for question %d: %s", idx, e)

        qa_pairs.append({
            "question": question,
            "answer": answer_json.get("answer"),
            "verdict_component": answer_json.get("verdict_component"),
            "confidence": answer_json.get("confidence"),
            "explanation": answer_json.get("explanation"),
            "evidence": evidence_snippets
        })

    # STEP 5: Synthesize final verdict
    synthesis_chain = (
        verdict_synthesis_prompt_template
        | primary_llm
        | StrOutputParser()
    )

    qa_pairs_serialized = json.dumps(qa_pairs, default=lambda o: str(o))[:10000]
    synthesis_raw = synthesis_chain.invoke({
        "claim": claim,
        "qa_pairs": qa_pairs_serialized
    })

    overall_verdict = "UNCERTAIN"
    overall_confidence = 0.0
    overall_explanation = synthesis_raw
    try:
        if synthesis_raw.strip().startswith("{"):
            parsed = json.loads(synthesis_raw)
            overall_verdict = parsed.get("verdict", overall_verdict)
            overall_confidence = float(parsed.get("confidence", 0.5))
            overall_explanation = parsed.get("explanation", overall_explanation)
    except Exception as e:
        logger.warning("Failed to parse synthesis result: %s", e)

    # STEP 6: Gemini cross‑check (reuse existing helper)
    cross_check_verdict = cross_check_confidence = cross_check_explanation = None
    try:
        gemini_llm = get_gemini_llm()
        cross_check_chain = (
            gemini_cross_check_prompt_template
            | gemini_llm
            | StrOutputParser()
        )
        evidence_str = json.dumps(all_evidence, default=lambda o: str(o))[:10000]
        cross_in = {
            "claim": claim,
            "evidence": evidence_str,
            "initial_verdict": overall_verdict,
            "initial_explanation": overall_explanation
        }
        cross_raw = cross_check_chain.invoke(cross_in)
        if cross_raw.strip().startswith("{"):
            cross = json.loads(cross_raw)
            cross_check_verdict = cross.get("verdict")
            cross_check_confidence = float(cross.get("confidence", 0.5))
            cross_check_explanation = cross.get("explanation")
        else:
            cross_check_explanation = cross_raw
    except Exception as e:
        logger.warning("Gemini cross‑check failed: %s", e)

    # Build result dataclass
    result = FactCheckResult(
        claim=claim,
        verdict=overall_verdict,
        confidence=overall_confidence,
        explanation=overall_explanation,
        evidence=all_evidence,
        sub_claims=qa_pairs,  # using this field to store question answers
        cross_check_verdict=cross_check_verdict,
        cross_check_confidence=cross_check_confidence,
        cross_check_explanation=cross_check_explanation
    )

    logger.info("Completed V2 fact‑check with verdict: %s", overall_verdict)
    return result 