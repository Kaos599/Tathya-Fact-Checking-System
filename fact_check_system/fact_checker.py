"""
Main orchestrator for the fact-checking process.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field # Import BaseModel and Field

from . import config, tools, prompts, models
from .models import FactCheckResult, EvidenceSource, GeminiParsedOutput
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, ListOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Pydantic model for the decomposition output
class DecompositionOutput(BaseModel):
    decomposition: List[str] = Field(description="List of distinct, meaningful components extracted from the claim for verification.")

def format_evidence_for_prompt(evidence_list: List[EvidenceSource]) -> str:
    """Formats the collected evidence into a string suitable for the final prompt."""
    formatted = ""
    for i, evidence in enumerate(evidence_list):
        formatted += f"Evidence {i+1} (Source: {evidence.source_tool}):\n"
        if evidence.title: formatted += f"  Title: {evidence.title}\n"
        if evidence.url: formatted += f"  URL: {evidence.url}\n"
        if evidence.snippet: formatted += f"  Snippet: {evidence.snippet[:500]}...\n"
        # Optionally include key facts from Gemini parsed output
        if isinstance(evidence.raw_content, dict) and evidence.source_tool == 'Gemini+GoogleSearch':
             if 'key_facts' in evidence.raw_content and evidence.raw_content['key_facts']:
                 formatted += f"  Key Facts: {'; '.join(evidence.raw_content['key_facts'])}\n"
        formatted += "---\n"
    return formatted if formatted else "No evidence was gathered."

def run_fact_check(claim: str) -> FactCheckResult:
    """
    Runs the complete fact-checking pipeline for a given claim.

    Args:
        claim: The claim string to investigate.

    Returns:
        A FactCheckResult object containing the verdict, explanation, evidence, etc.
    """
    claim_id = f"fc-{uuid.uuid4()}"
    logger.info(f"Starting fact check for claim_id: {claim_id}, Claim: '{claim}'")

    all_evidence: List[EvidenceSource] = [] # List to store EvidenceSource objects
    decomposition: Optional[List[str]] = None

    # TODO: Implement actual LLM call for decomposition using prompts.CLAIM_DECOMPOSITION_TEMPLATE
    try:
        if tools.primary_llm:
            # Use with_structured_output for claim decomposition
            decomposition_chain = prompts.CLAIM_DECOMPOSITION_TEMPLATE | tools.primary_llm.with_structured_output(DecompositionOutput)
            decomposition_result = decomposition_chain.invoke({"claim": claim})
            decomposition = decomposition_result.decomposition # Access the list via the attribute
            logger.info(f"Claim decomposed by LLM into: {decomposition}")
        else:
             logger.error("Primary LLM not available, skipping decomposition.")
             decomposition = [] # Handle case where decomposition is needed but LLM is unavailable
    except Exception as e:
        logger.error(f"Error during claim decomposition: {e}")
        decomposition = []

    logger.info("Starting evidence gathering phase...")
    collected_results: List[Dict[str, Any]] = [] # Store raw results before converting to EvidenceSource

    tavily_results = tools.perform_tavily_search(claim)
    if tavily_results and tavily_results.get('results'):
        logger.info(f"Got {len(tavily_results['results'])} results from Tavily.")
        collected_results.extend(tavily_results['results'])
    else:
        logger.warning("No results or error from Tavily.")

    gemini_parsed: Optional[GeminiParsedOutput] = tools.perform_gemini_search_and_parse(claim)
    if gemini_parsed:
        logger.info(f"Gemini search and parse successful. Summary: {gemini_parsed.summary[:100]}...")
        collected_results.append(gemini_parsed.model_dump())
    else:
        logger.warning("Gemini search failed or parsing failed.")

    wikidata_queries = decomposition if decomposition else []
    if not wikidata_queries and claim: # Fallback if decomposition failed
        # Improved fallback: use the whole claim instead of just the first word
        wikidata_queries = [claim] 
        logger.warning(f"Decomposition failed or empty, using full claim for Wikidata: {wikidata_queries}")
        
    for term in wikidata_queries:
        wikidata_results = tools.search_wikidata_entities(term)
        if wikidata_results and wikidata_results.get('search'):
            logger.info(f"Got {len(wikidata_results['search'])} results from Wikidata for term '{term}'.")
            collected_results.extend(wikidata_results['search'])

    # TODO: Optionally generate specific queries for DDG/News based on decomposition using LLM
    # Fallback here already uses the full claim if decomposition is empty
    search_queries = decomposition if decomposition else [claim] 
    if not search_queries:
         logger.warning("No search terms available for DuckDuckGo/NewsAPI after decomposition failure.")

    ddg_raw_results = []
    for query in search_queries:
        ddg_res = tools.perform_duckduckgo_search(query)
        if ddg_res:
            logger.info(f"Got {len(ddg_res)} results from DuckDuckGo for query '{query}'.")
            ddg_raw_results.extend(ddg_res)

    news_raw_results = []
    for query in search_queries:
        news_res = tools.perform_news_search(query)
        if news_res and news_res.get('articles'):
            logger.info(f"Got {len(news_res['articles'])} articles from NewsAPI for query '{query}'.")
            news_raw_results.extend(news_res['articles'])

    logger.info("Starting evidence verification phase...")
    verified_ddg_results = tools.verify_duckduckgo_results(claim, ddg_raw_results)
    verified_news_results = tools.verify_news_results(claim, news_raw_results)

    collected_results.extend(verified_ddg_results)
    collected_results.extend(verified_news_results)

    logger.info(f"Collected {len(collected_results)} raw results before deduplication.")
    unique_raw_results = tools.deduplicate_results(collected_results, key='href')
    logger.info(f"Have {len(unique_raw_results)} unique results after deduplication.")

    for i, raw_res in enumerate(unique_raw_results):
        try:
            source_tool = raw_res.get('source_tool', 'Unknown')
            snippet = raw_res.get('body') or raw_res.get('content') or raw_res.get('snippet') or raw_res.get('description')
            
            if source_tool == 'Gemini+GoogleSearch':
                 snippet = raw_res.get('summary', 'No summary available')
                 title = f"Gemini Summary for '{claim[:50]}...'"
                 url = None
            else:
                 title = raw_res.get('title')
                 url = raw_res.get('href') or raw_res.get('url')
            
            evidence = EvidenceSource(
                id=f"evid-{claim_id}-{i}",
                source_tool=source_tool,
                url=url,
                title=title,
                snippet=snippet,
                raw_content=raw_res
            )
            all_evidence.append(evidence)
        except Exception as e:
             logger.warning(f"Could not format raw result into EvidenceSource: {raw_res}. Error: {e}")


    logger.info("Starting final synthesis phase...")
    formatted_evidence_str = format_evidence_for_prompt(all_evidence)

    verdict = "UNCERTAIN"
    confidence_score = 0.1
    explanation = "Synthesis failed or Primary LLM unavailable."

    if tools.primary_llm:
        try:
            parser = JsonOutputParser()
            chain = prompts.FINAL_ANSWER_TEMPLATE | tools.primary_llm | parser

            synthesis_input = {
                "claim": claim,
                "formatted_evidence": formatted_evidence_str,
            }
            final_result_dict = chain.invoke(synthesis_input)

            verdict = final_result_dict.get("verdict", "UNCERTAIN")
            confidence_score = float(final_result_dict.get("confidence_score", 0.1))
            explanation = final_result_dict.get("explanation", "LLM did not provide an explanation.")
            logger.info(f"Synthesis complete. Verdict: {verdict}, Confidence: {confidence_score}")

        except Exception as e:
            logger.error(f"Error during final synthesis: {e}. Falling back to default uncertain verdict.")
    else:
        logger.error("Primary LLM not available for final synthesis.")

    final_output = FactCheckResult(
        claim=claim,
        claim_id=claim_id,
        verdict=verdict,
        confidence_score=confidence_score,
        explanation=explanation,
        evidence_sources=all_evidence,
        decomposition=decomposition
    )

    logger.info(f"Fact check completed for claim_id: {claim_id}. Verdict: {final_output.verdict}")
    return final_output 