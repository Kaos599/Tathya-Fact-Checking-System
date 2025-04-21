"""
This module contains wrapper functions for various search and information gathering tools
used by the fact-checking system, refactored as Langchain Tools.
"""

import logging
from typing import List, Dict, Any, Optional, Type
from . import config  # Import from the current package
from .models import GeminiParsedOutput # Import the Pydantic model for parsed output
from .prompts import RESULT_VERIFICATION_TEMPLATE, VERIFICATION_PROMPT # Keep for potential future use or direct calls if needed
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.tools import tool, BaseTool, Tool
from .models import FactCheckResult, EvidenceSource # Make sure this is imported
from .config import get_primary_llm # Import config helper
from langchain.tools import BaseTool, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.pydantic_v1 import BaseModel, Field # Use v1 for tool compatibility
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.wikidata.tool import WikidataQueryRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.wikidata import WikidataAPIWrapper
from langchain_community.document_loaders import WebBaseLoader # For Web Scraping
from newsapi import NewsApiClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



# --- Internal Helper Functions (like parsing) ---
# This remains internal, called by the gemini tool
def parse_gemini_output_with_llm(gemini_raw_output: str, claim: str) -> Optional[GeminiParsedOutput]:
    """
    Uses the secondary Azure OpenAI LLM to parse the raw text output from the
    Gemini+GoogleSearch call into a structured format. Internal helper function.

    Args:
        gemini_raw_output: The raw string output from perform_gemini_google_search.
        claim: The original claim, provided for context.

    Returns:
        A GeminiParsedOutput Pydantic object, or None if parsing fails or LLM is unavailable.
    """
    parser_llm = config.get_azure_openai_parser_llm() # Get client inside function
    if not parser_llm:
        logger.error("Azure OpenAI Parser LLM is not available for Gemini parsing.")
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
        # Add source tool info within the Pydantic model if possible, or handle in aggregation step
        # For now, return the Pydantic object directly
        return GeminiParsedOutput(**parsed_result)
    except OutputParserException as ope:
         logger.error(f"Failed to parse Gemini output. Parser Error: {ope}. Raw output was:\n{gemini_raw_output[:500]}...")
         return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during Gemini output parsing: {e}")
        return None

# --- Tool Input Schemas ---

class SearchInput(BaseModel):
    query: str = Field(description="The search query string.")

class WikidataInput(BaseModel):
    query: str = Field(description="A specific entity name (e.g., 'Eiffel Tower') or concept to query Wikidata for structured data.") # Simplified input

class WebScraperInput(BaseModel):
    urls: List[str] = Field(description="A list of specific, highly relevant URLs to scrape for full content.")

class NewsSearchInput(BaseModel):
    query: str = Field(description="Keywords or phrase to search for in recent news articles.")
    language: str = Field(default='en', description="The 2-letter ISO 639-1 code of the language.")
    page_size: int = Field(default=10, description="Number of news results (max 100).")


# --- Claim Decomposition Tool ---
class ClaimDecompositionInput(BaseModel):
    claim: str = Field(..., description="The complex claim to decompose into simpler sub-claims.")

class ClaimDecompositionOutput(BaseModel):
    sub_claims: List[str] = Field(..., description="A list of simpler, verifiable sub-claims derived from the original claim.")

DECOMPOSITION_PROMPT = """
You are an expert in analyzing claims. Your task is to decompose the given complex claim into a list of simpler, distinct, and verifiable sub-claims or questions. Each sub-claim should represent a single factual assertion that can be independently verified.

Focus on atomicity: Break down the claim into the smallest possible factual units.
Focus on coverage: Ensure all key aspects of the original claim are covered by the sub-claims.
Focus on verifiability: Each sub-claim should be a statement or question that can potentially be answered with facts.

Original Claim:
"{claim}"

Decompose the claim into a list of sub-claims. Provide your response ONLY as a JSON object matching the following schema:
{{
    "sub_claims": ["sub-claim 1", "sub-claim 2", ...]
}}
"""

class ClaimDecompositionTool(BaseTool):
    """Tool to decompose complex claims."""
    name: str = "claim_decomposition_tool"
    description: str = (
        "Decomposes a complex claim (containing multiple assertions or entities) into a list of simpler, "
        "independently verifiable sub-claims or questions. Use this *first* on complex claims to create focused queries for other tools."
    )
    args_schema: Type[BaseModel] = ClaimDecompositionInput

    def _run(self, claim: str) -> Dict[str, Any]:
        """Executes the claim decomposition."""
        logger.info(f"Executing Claim Decomposition Tool for claim: '{claim}'")
        llm = get_primary_llm()
        if not llm:
            logger.error("Primary LLM not available for Claim Decomposition Tool.")
            return {"error": "LLM unavailable for decomposition."}

        parser = JsonOutputParser(pydantic_object=ClaimDecompositionOutput)
        prompt = PromptTemplate(template=DECOMPOSITION_PROMPT, input_variables=["claim"])
        chain = prompt | llm | parser

        try:
            result = chain.invoke({"claim": claim})
            logger.info(f"Claim decomposition successful. Found {len(result.get('sub_claims', []))} sub-claims.")
            return result
        except Exception as e:
            logger.error(f"Error during claim decomposition for '{claim}': {e}", exc_info=True)
            return {"error": f"Claim decomposition failed: {str(e)}"}

    async def _arun(self, claim: str) -> Dict[str, Any]:
        """Async execution."""
        logger.info(f"Executing async Claim Decomposition Tool for claim: '{claim}'")
        llm = get_primary_llm()
        if not llm:
            logger.error("Primary LLM not available for async Claim Decomposition Tool.")
            return {"error": "LLM unavailable for decomposition."}

        parser = JsonOutputParser(pydantic_object=ClaimDecompositionOutput)
        prompt = PromptTemplate(template=DECOMPOSITION_PROMPT, input_variables=["claim"])
        chain = prompt | llm | parser

        try:
            result = await chain.ainvoke({"claim": claim})
            logger.info(f"Async claim decomposition successful. Found {len(result.get('sub_claims', []))} sub-claims.")
            return result
        except Exception as e:
            logger.error(f"Error during async claim decomposition for '{claim}': {e}", exc_info=True)
            return {"error": f"Async claim decomposition failed: {str(e)}"}


# --- Search & Retrieval Tools ---

@tool
def tavily_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Performs a comprehensive web search using Tavily. Ideal for initial investigation of a claim or sub-claim.
    Returns a list of relevant documents with URLs, titles, and content snippets.
    Use this first to get a broad overview and identify key sources or entities.
    """
    tavily_client = config.get_tavily_client()
    if not tavily_client:
        logger.error("Tavily client is not available.")
        return {"error": "Tavily client unavailable."}
    try:
        logger.info(f"Performing Tavily search for query: '{query}'")
        # Tavily tool from langchain community automatically handles dict -> list conversion if needed
        response = tavily_client.search(query=query, search_depth="advanced", max_results=max_results)
        # Ensure response is a dict suitable for the agent state (Tavily client might return a list)
        if isinstance(response, list):
            response_dict = {"results": response}
        else:
            response_dict = response

        logger.info(f"Tavily search completed. Found {len(response_dict.get('results', []))} results.")
        if 'results' in response_dict:
             for result in response_dict['results']:
                 result['source_tool'] = 'Tavily'
        return response_dict
    except Exception as e:
        logger.error(f"Error during Tavily search for query '{query}': {e}")
        return {"error": f"Tavily search failed: {e}"}

@tool(args_schema=WebScraperInput)
def scrape_webpages_tool(urls: List[str]) -> List[Dict[str, Any]]:
    """
    Retrieves the *full text content* from a list of specific URLs.
    Use this *only* when a search result snippet (from Tavily, DuckDuckGo, etc.) is insufficient but the URL seems highly relevant and promising.
    This tool is slower and more resource-intensive than reading search snippets.
    """
    results = []
    loader = WebBaseLoader(urls, continue_on_failure=True) # Continue if one URL fails
    loader.requests_per_second = 2 # Respect rate limits
    try:
        logger.info(f"Attempting to scrape {len(urls)} URLs: {urls}")
        docs = loader.load()
        loaded_data = {doc.metadata.get('source'): doc.page_content for doc in docs}
        logger.info(f"Successfully loaded content for {len(loaded_data)} URLs.")

        for url in urls:
            content = loaded_data.get(url)
            if content:
                results.append({"url": url, "content": content[:5000], "source_tool": "WebScraper"}) # Limit content length
                logger.info(f"Finished scraping URL: {url}. Content length: {len(content)}")
            else:
                error_msg = f"Failed to load or extract content for URL: {url}"
                results.append({"url": url, "content": error_msg, "error": error_msg, "source_tool": "WebScraper"})
                logger.warning(error_msg)

    except Exception as e:
        logger.error(f"Error during scraping URLs {urls}: {e}", exc_info=True)
        # Append error status for any URLs not already processed
        processed_urls = {res['url'] for res in results}
        for url in urls:
            if url not in processed_urls:
                results.append({"url": url, "content": f"Scraping failed: {e}", "error": str(e), "source_tool": "WebScraper"})
    return results


@tool(args_schema=NewsSearchInput)
def news_search(query: str, language: str = 'en', page_size: int = 10) -> Dict[str, Any]:
    """
    Searches *recent news articles* (past ~30 days) related to a query using NewsAPI.
    Crucial for time-sensitive claims, recent events, or verifying information reported in the news.
    Returns articles with titles, sources, URLs, snippets, and publication dates.
    """
    news_client = config.get_news_client()
    if not news_client:
        logger.error("NewsAPI client is not available.")
        return {"error": "NewsAPI client unavailable."}
    try:
        logger.info(f"Performing NewsAPI search for query: '{query}', lang: {language}, size: {page_size}")
        # Use get_everything for broader search, or top_headlines for major news
        response = news_client.get_everything(
            q=query,
            language=language,
            page_size=min(page_size, 100),
            sort_by='relevancy' # 'publishedAt' or 'popularity' also options
        )
        logger.info(f"NewsAPI search completed. Status: {response.get('status')}, Found {response.get('totalResults', 0)} results.")
        if 'articles' in response:
             for article in response['articles']:
                 article['source_tool'] = 'NewsAPI'
                 article['snippet'] = article.get('description') or article.get('content') # Standardize snippet
                 article['title'] = article.get('title')
                 article['url'] = article.get('url')
        return response
    except Exception as e:
        logger.error(f"Error during NewsAPI search for query '{query}': {e}")
        return {"error": f"NewsAPI search failed: {e}"}


@tool(args_schema=SearchInput)
def duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Performs a web search using DuckDuckGo. Provides an *alternative search perspective* to Tavily/Google.
    Useful for corroborating findings or when other search tools yield poor results.
    Returns a list of result dictionaries, each containing 'title', 'href'(URL), and 'body'(snippet).
    """
    # Using the community wrapper directly
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)
    search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
    try:
        logger.info(f"Performing DuckDuckGo search for query: '{query}'")
        # The wrapper now expects a single string and returns a string.
        # To maintain consistency (returning a list of dicts), we might need to parse or adjust.
        # Let's try invoking the base wrapper's 'results' method if available, or parse the string.
        # **Correction**: DuckDuckGoSearchRun itself returns a string. We need the underlying wrapper.
        # response_str = search_tool.run(query) # This returns a formatted string
        # Let's use the wrapper directly if possible, or parse the string result
        results = wrapper.results(query, max_results=max_results)  # This returns List[Dict]
        logger.info(f"DuckDuckGo search completed. Found {len(results)} results.")
        # Standardize output fields for agent extraction
        standardized = []
        for item in results:
            # Determine URL from possible keys
            link = item.get('href') or item.get('link') or item.get('url') or item.get('body')
            # Determine snippet
            snippet = item.get('body') or item.get('snippet') or ''
            standardized_item = {
                'title': item.get('title', ''),
                'url': link,
                'snippet': snippet,
                'source_tool': 'DuckDuckGo',
            }
            # Keep raw content if needed
            standardized_item['raw_content'] = item
            standardized.append(standardized_item)
        return standardized
    except Exception as e:
        logger.error(f"Error during DuckDuckGo search for query '{query}': {e}")
        return [{"error": f"DuckDuckGo search failed: {e}"}]


@tool(args_schema=WikidataInput)
def wikidata_entity_search(query: str) -> Dict[str, Any]:
    """
    Retrieves *structured data* about a specific entity (person, place, organization, concept) from Wikidata.
    Use this *after* identifying a key entity in the claim via other search tools. Input the entity name directly.
    Returns official labels, descriptions, aliases, and potentially related facts (depending on query complexity handled by the wrapper).
    NOT suitable for general questions or broad topic searches.
    """
    api_wrapper = WikidataAPIWrapper()
    runnable = WikidataQueryRun(api_wrapper=api_wrapper)
    try:
        logger.info(f"Performing Wikidata entity search for: '{query}'")
        # Invoke the runnable - it handles basic entity lookup
        response = runnable.invoke(query)
        log_response = str(response)[:200] + "..." if isinstance(response, (str, dict, list)) else type(response)
        logger.info(f"Wikidata search completed. Response: {log_response}")

        # Attempt to structure the response slightly for consistency
        # WikidataQueryRun often returns a string summary.
        if isinstance(response, str):
             # Check if it looks like an error or 'No good Wikidata result found'
             if "No good Wikidata result found" in response or "Error" in response:
                 return {"error": response, "source_tool": "Wikidata"}
             else:
                 # Assume it's a descriptive summary
                 return {"summary": response, "source_tool": "Wikidata"}
        elif isinstance(response, dict): # Should ideally return dict for structured data
             response['source_tool'] = 'Wikidata'
             return response
        elif isinstance(response, list): # Can happen with complex SPARQL implicitly generated
            return {"results": response, "source_tool": "Wikidata"}
        else: # Fallback
            return {"result": str(response), "source_tool": "Wikidata"} # Convert unexpected types to string

    except Exception as e:
        logger.error(f"Error during Wikidata search for '{query}': {e}", exc_info=True)
        return {"error": f"Wikidata search failed: {e}", "source_tool": "Wikidata"}


# --- Gemini Tool ---
class GeminiGoogleSearchInput(BaseModel):
    query: str = Field(description="The query or sub-claim to investigate using Google Search synthesized by Gemini.")

# Note: This function itself is not decorated with @tool here,
# but is wrapped by `gemini_google_search_tool` below, which *is* decorated.
def _gemini_google_search_and_parse_internal(query: str) -> Dict[str, Any]:
    """Internal logic for Gemini search and parsing."""
    logger.info(f"Performing Gemini Google Search for query: '{query}'")
    
    # --- Call the existing function from config.py to perform the search ---
    try:
        raw_gemini_output = config.perform_gemini_google_search(query)
        
        if raw_gemini_output is None or raw_gemini_output.startswith("Error:"):
            logger.error(f"Gemini search failed or returned an error: {raw_gemini_output}")
            return {"error": f"Gemini search failed: {raw_gemini_output}", "source_tool": "GeminiGoogleSearch"}
        
        logger.info("Gemini search successful, attempting to parse output.")

        # --- Call the existing helper function to parse the raw output ---
        # The original claim (query in this context) is needed for the parser prompt
        parsed_data: Optional[GeminiParsedOutput] = parse_gemini_output_with_llm(raw_gemini_output, query)

        if parsed_data is None:
            logger.error("Failed to parse the output from Gemini search.")
            # Return raw output along with error for context
            return {"error": "Failed to parse Gemini output.", "raw_output": raw_gemini_output[:1000], "source_tool": "GeminiGoogleSearch"}
        
        logger.info("Gemini search and parse completed successfully.")
        # Convert Pydantic model to dict for consistent tool output type
        output_dict = parsed_data.dict()
        output_dict['source_tool'] = 'GeminiGoogleSearch' # Ensure source tool is added
        return output_dict

    except Exception as e:
        logger.error(f"Error during Gemini Google Search orchestration for query '{query}': {e}", exc_info=True)
        return {"error": f"Gemini Google Search orchestration failed: {e}", "source_tool": "GeminiGoogleSearch"}

@tool(args_schema=GeminiGoogleSearchInput)
def gemini_google_search_tool(query: str) -> Dict[str, Any]:
    """
    Performs a Google Search via Gemini, synthesizing results into a summary.
    Use this for complex questions or sub-claims where an initial LLM synthesis of search results is helpful.
    Returns a dictionary with the synthesized summary and cited source URLs found by Gemini.
    """
    # This tool simply wraps the internal function.
    return _gemini_google_search_and_parse_internal(query)



# --- REMOVED Verification Tool ---
# The verification logic will now be implicitly handled by the agent's decision
# to call FINISH based on its analysis of gathered evidence.


# --- Tool Creation Function ---

def create_agent_tools(cfg: Optional[Dict] = None) -> List[BaseTool]:
    """Creates a list of tools based on the provided configuration."""
    if cfg is None:
        cfg = {}
    tools = []
    logger.info(f"Creating tools with configuration: {cfg}")

    # Core Tools (Should generally be enabled)
    if cfg.get("enable_claim_decomposition", True):
        tools.append(ClaimDecompositionTool())
        logger.info("Claim Decomposition Tool enabled.")

    if cfg.get("enable_tavily", True): # Defaulting Tavily to True as primary search
        if config.TAVILY_API_KEY:
            tools.append(tavily_search)
            logger.info("Tavily Search Tool enabled.")
        else:
            logger.warning("Tavily Search Tool requested but TAVILY_API_KEY not set.")

    # Optional / Secondary Tools
    if cfg.get("enable_duckduckgo", True): # Enable DDG as alternative
        tools.append(duckduckgo_search)
        logger.info("DuckDuckGo Search Tool enabled.")

    if cfg.get("enable_wikidata", True): # Enable Wikidata for specific entity lookups
        tools.append(wikidata_entity_search)
        logger.info("Wikidata Search Tool enabled.")

    if cfg.get("enable_web_scraper", True): # Enable scraper for deep dives
        tools.append(scrape_webpages_tool)
        logger.info("Web Scraper Tool enabled.")

    if cfg.get("enable_news_search", True): # Enable News for recent events
        if config.NEWS_API_KEY:
            tools.append(news_search)
            logger.info("News Search Tool enabled.")
        else:
            logger.warning("News Search Tool requested but NEWS_API_KEY not set.")

    if cfg.get("enable_gemini_search", True): # Enable Gemini as another powerful option
        if config.GEMINI_API_KEY:
             tools.append(gemini_google_search_tool)
             logger.info("Gemini Google Search Tool enabled.")
        else:
             logger.warning("Gemini Google Search Tool requested but GEMINI_API_KEY not set.")


    # --- Add a FINISH tool ---
    # Represents the agent's decision to end the investigation phase.
    def finish_func(reason: str = "Investigation complete."): # Add optional reason arg
        """Signals that the agent has finished all research and analysis, and is ready for the final answer synthesis based on the gathered evidence. Provide a brief reason for finishing."""
        logger.info(f"Agent signaled FINISH. Reason: {reason}")
        # The function doesn't need to *do* anything, its invocation signals the graph.
        return f"FINISH signal received. Reason: {reason}"

    # Define args schema for the FINISH tool
    class FinishSchema(BaseModel):
        reason: str = Field(default="Investigation complete.", description="A brief reason why the investigation is being concluded (e.g., 'sufficient evidence found', 'conflicting evidence found', 'searches exhausted').")

    tools.append(
        Tool(
            name="FINISH",
            func=finish_func,
            description="Call this ONLY when you have gathered sufficient evidence (or exhausted all relevant search strategies) and are ready to conclude the investigation phase. Provide a brief reason.",
            args_schema=FinishSchema # Add schema for the reason
        )
    )
    logger.info("FINISH tool enabled.")

    if not tools or all(t.name == "FINISH" for t in tools): # Check if only FINISH tool is left
        logging.error("No operational tools were enabled for the agent! It cannot investigate.")
        raise ValueError("Agent requires at least one operational tool (like search or scrape) besides FINISH.")

    logger.info(f"Total tools created: {len(tools)}")
    return tools