"""
This module contains wrapper functions for various search and information gathering tools
used by the fact-checking system.
"""

import logging
from typing import List, Dict, Any, Optional
from . import config  # Import from the current package
from .models import GeminiParsedOutput # Import the Pydantic model for parsed output
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.exceptions import OutputParserException

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize clients using factory functions from config
try:
    tavily_client = config.get_tavily_client()
    logger.info("Tavily client initialized.")
except ValueError as e:
    tavily_client = None
    logger.warning(f"Failed to initialize Tavily client: {e}")

try:
    ddg_client = config.get_duckduckgo_client()
    logger.info("DuckDuckGo client initialized.")
except Exception as e: # Catch broader exceptions if DDGS init fails
    ddg_client = None
    logger.warning(f"Failed to initialize DuckDuckGo client: {e}")

try:
    news_client = config.get_news_api_client()
    logger.info("NewsAPI client initialized.")
except ValueError as e:
    news_client = None
    logger.warning(f"Failed to initialize NewsAPI client: {e}")

try:
    wikidata_query_func = config.get_wikidata_client()
    logger.info("Wikidata query function obtained.")
except Exception as e:
    wikidata_query_func = None
    logger.warning(f"Failed to obtain Wikidata query function: {e}")

try:
    primary_llm = config.get_primary_llm()
    logger.info("Primary LLM client initialized.")
except ValueError as e:
    primary_llm = None
    logger.error(f"Failed to initialize primary LLM: {e}. Verification steps will fail.")

try:
    parser_llm = config.get_azure_openai_parser_llm()
    logger.info("Azure OpenAI Parser LLM client initialized.")
except ValueError as e:
    parser_llm = None
    logger.error(f"Failed to initialize Azure OpenAI Parser LLM: {e}. Gemini parsing will fail.")


def perform_tavily_search(query: str, max_results: int = 5) -> Optional[Dict[str, Any]]:
    """
    Performs a search using the Tavily API.

    Args:
        query: The search query.
        max_results: The maximum number of results to return.

    Returns:
        A dictionary containing the search results, or None if an error occurs or client is unavailable.
    """
    if not tavily_client:
        logger.error("Tavily client is not available.")
        return None
    try:
        logger.info(f"Performing Tavily search for query: '{query}'")
        response = tavily_client.search(query=query, search_depth="advanced", max_results=max_results)
        logger.info(f"Tavily search completed. Found {len(response.get('results', []))} results.")
        if 'results' in response:
             for result in response['results']:
                 result['source_tool'] = 'Tavily'
        return response
    except Exception as e:
        logger.error(f"Error during Tavily search for query '{query}': {e}")
        return None

def perform_duckduckgo_search(query: str, max_results: int = 5) -> Optional[List[Dict[str, str]]]:
    """
    Performs a search using the DuckDuckGo Search API.

    Args:
        query: The search query.
        max_results: The maximum number of results to return.

    Returns:
        A list of search result dictionaries [{title, href, body}], or None if an error occurs or client is unavailable.
    """
    if not ddg_client:
        logger.error("DuckDuckGo client is not available.")
        return None
    try:
        logger.info(f"Performing DuckDuckGo search for query: '{query}'")
        results = ddg_client.text(query, max_results=max_results)
        logger.info(f"DuckDuckGo search completed. Found {len(results)} results.")
        for result in results:
            result['source_tool'] = 'DuckDuckGo'
        return results
    except Exception as e:
        logger.error(f"Error during DuckDuckGo search for query '{query}': {e}")
        return None

def perform_news_search(query: str, language: str = 'en', page_size: int = 10) -> Optional[Dict[str, Any]]:
    """
    Performs a search using the NewsAPI.

    Args:
        query: The search query (keywords or phrase).
        language: The 2-letter ISO-639-1 code of the language.
        page_size: The number of results to return per page (max 100).

    Returns:
        A dictionary containing the news articles, or None if an error occurs or client is unavailable.
    """
    if not news_client:
        logger.error("NewsAPI client is not available.")
        return None
    try:
        logger.info(f"Performing NewsAPI search for query: '{query}'")
        # Use get_everything for broader search, consider get_top_headlines for recent major news
        response = news_client.get_everything(q=query, language=language, sort_by='relevancy', page_size=page_size)
        logger.info(f"NewsAPI search completed. Found {response.get('totalResults', 0)} articles.")
        if 'articles' in response:
             for article in response['articles']:
                 article['source_tool'] = 'NewsAPI'
                 # Standardize keys slightly if needed (e.g., 'url' instead of 'href')
                 article['href'] = article.get('url')
                 article['body'] = article.get('description') or article.get('content')

        return response
    except Exception as e:
        logger.error(f"Error during NewsAPI search for query '{query}': {e}")
        return None

def search_wikidata_entities(query: str, limit: int = 3) -> Optional[Dict[str, Any]]:
    """
    Searches for entities on Wikidata.

    Args:
        query: The entity name to search for.
        limit: The maximum number of entities to return.

    Returns:
        A dictionary containing the search results, or None if an error occurs or function is unavailable.
    """
    if not wikidata_query_func:
        logger.error("Wikidata query function is not available.")
        return None
    try:
        logger.info(f"Searching Wikidata for entities matching: '{query}'")
        results = wikidata_query_func(search_term=query, limit=limit)
        logger.info(f"Wikidata entity search completed. Found {len(results.get('search', []))} potential matches.")
        if 'search' in results:
             for result in results['search']:
                 result['source_tool'] = 'Wikidata'
                 # Add standard keys for compatibility if needed
                 result['title'] = result.get('label')
                 result['href'] = result.get('concepturi')
                 result['body'] = result.get('description')
        return results
    except Exception as e:
        logger.error(f"Error during Wikidata entity search for '{query}': {e}")
        return None

def parse_gemini_output_with_llm(gemini_raw_output: str, claim: str) -> Optional[GeminiParsedOutput]:
    """
    Uses the secondary Azure OpenAI LLM to parse the raw text output from the
    Gemini+GoogleSearch call into a structured format.

    Args:
        gemini_raw_output: The raw string output from perform_gemini_google_search.
        claim: The original claim, provided for context.

    Returns:
        A GeminiParsedOutput Pydantic object, or None if parsing fails or LLM is unavailable.
    """
    if not parser_llm:
        logger.error("Azure OpenAI Parser LLM is not available.")
        return None

    parser = JsonOutputParser(pydantic_object=GeminiParsedOutput)

    prompt_template = ChatPromptTemplate.from_template(
        """You are an expert assistant tasked with parsing the output of a Google Search-enabled Gemini model call.
        The Gemini model was asked to investigate the following claim: '{claim}'
        Its raw output, potentially containing summaries, facts, and source information, is provided below.
        Your goal is to extract the key information and structure it into a JSON object matching the requested format.

        Focus on identifying:
        1.  A concise summary of the findings regarding the claim.
        2.  A list of key facts or pieces of information presented.
        3.  A list of URLs identified as sources in the text.

        Raw Gemini Output:
        ---
        {gemini_raw_output}
        ---

        Format Instructions:
        {format_instructions}
        """
    )

    chain = prompt_template | parser_llm | parser
    logger.info("Attempting to parse Gemini output using Azure OpenAI Parser LLM.")

    try:
        parsed_result = chain.invoke({
            "claim": claim,
            "gemini_raw_output": gemini_raw_output,
            "format_instructions": parser.get_format_instructions()
        })
        logger.info("Successfully parsed Gemini output.")
        parsed_result['source_tool'] = 'Gemini+GoogleSearch'
        return GeminiParsedOutput(**parsed_result)
    except OutputParserException as ope:
         logger.error(f"Failed to parse Gemini output. Parser Error: {ope}. Raw output was:\n{gemini_raw_output[:500]}...") 
         return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during Gemini output parsing: {e}")
        return None


def perform_gemini_search_and_parse(claim: str) -> Optional[GeminiParsedOutput]:
    """
    Performs a Gemini+Google Search for the claim and then parses the output.

    Args:
        claim: The claim to investigate.

    Returns:
        A GeminiParsedOutput object containing the structured results, or None if any step fails.
    """
    logger.info(f"Starting Gemini+Google Search and Parse for claim: '{claim}'")
    raw_gemini_output = config.perform_gemini_google_search(claim)

    if raw_gemini_output is None or raw_gemini_output.startswith("Error:"):
        logger.error(f"Gemini search failed or returned an error: {raw_gemini_output}")
        return None

    parsed_output = parse_gemini_output_with_llm(raw_gemini_output, claim)

    if parsed_output is None:
        logger.error("Failed to parse the output from Gemini search.")
        return None

    logger.info("Gemini search and parse completed successfully.")
    return parsed_output


def verify_duckduckgo_results(claim: str, ddg_results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    (Placeholder) Verifies the relevance of DuckDuckGo results against the original claim using the primary LLM.

    Args:
        claim: The original claim.
        ddg_results: The list of results from DuckDuckGo.

    Returns:
        A filtered list containing only the relevant results.
    """
    if not primary_llm:
        logger.warning("Primary LLM not available. Skipping DuckDuckGo verification, returning all results.")
        return ddg_results

    logger.info(f"Verifying {len(ddg_results)} DuckDuckGo results for relevance to claim: '{claim}'")
    # TODO: Implement LLM call to check relevance of each result's body/title against the claim.
    # Example structure:
    # relevant_results = []
    # for result in ddg_results:
    #     # Construct prompt for relevance check
    #     # response = primary_llm.invoke(...)
    #     # if response indicates relevance:
    #     #     relevant_results.append(result)
    # return relevant_results
    logger.warning("DuckDuckGo verification logic not yet implemented. Returning all results.")
    return ddg_results # Placeholder: return all results for now

def verify_news_results(claim: str, news_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    (Placeholder) Verifies the relevance of NewsAPI results against the original claim using the primary LLM.

    Args:
        claim: The original claim.
        news_articles: The list of articles from NewsAPI.

    Returns:
        A filtered list containing only the relevant articles.
    """
    if not primary_llm:
        logger.warning("Primary LLM not available. Skipping NewsAPI verification, returning all results.")
        return news_articles

    logger.info(f"Verifying {len(news_articles)} NewsAPI articles for relevance to claim: '{claim}'")
    # TODO: Implement LLM call to check relevance of each article's description/content against the claim.
    # Example structure similar to verify_duckduckgo_results
    logger.warning("NewsAPI verification logic not yet implemented. Returning all results.")
    return news_articles # Placeholder: return all articles for now

def deduplicate_results(results: List[Dict[str, Any]], key: str = 'href') -> List[Dict[str, Any]]:
    """
    Deduplicates a list of result dictionaries based on a specific key (defaulting to 'href' for URL).

    Args:
        results: A list of dictionaries, where each dictionary represents a search result.
        key: The dictionary key to use for deduplication.

    Returns:
        A list of unique result dictionaries.
    """
    seen = set()
    unique_results = []
    for result in results:
        value = result.get(key)
        if value is not None and value not in seen:
            seen.add(value)
            unique_results.append(result)
    logger.info(f"Deduplicated results based on key '{key}'. Original: {len(results)}, Unique: {len(unique_results)}.")
    return unique_results