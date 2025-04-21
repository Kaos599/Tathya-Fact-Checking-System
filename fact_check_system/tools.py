"""
This module contains wrapper functions for various search and information gathering tools
used by the fact-checking system, refactored as Langchain Tools.
"""

import logging
from typing import List, Dict, Any, Optional
from . import config  # Import from the current package
from .models import GeminiParsedOutput # Import the Pydantic model for parsed output
from .prompts import RESULT_VERIFICATION_TEMPLATE, VERIFICATION_PROMPT # Keep for potential future use or direct calls if needed
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.tools import tool
from .models import FactCheckResult, EvidenceSource # Make sure this is imported
from .config import get_primary_llm # Import config helper
from langchain.tools import BaseTool, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.wikidata import WikidataAPIWrapper

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

# --- Langchain Tools ---

@tool
def tavily_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Performs a web search using the Tavily Search API to find relevant documents and sources for a given query.
    Returns a dictionary containing a list of search results, including URLs, titles, and content snippets.
    Use this for general web searches and evidence gathering.
    """
    tavily_client = config.get_tavily_client() # Get client inside function
    if not tavily_client:
        logger.error("Tavily client is not available.")
        # Raise error or return specific structure indicating failure
        return {"error": "Tavily client unavailable."}
    try:
        logger.info(f"Performing Tavily search for query: '{query}'")
        response = tavily_client.search(query=query, search_depth="advanced", max_results=max_results)
        logger.info(f"Tavily search completed. Found {len(response.get('results', []))} results.")
        # Add source tool info to each result for easier tracking later
        if 'results' in response:
             for result in response['results']:
                 result['source_tool'] = 'Tavily'
        return response
    except Exception as e:
        logger.error(f"Error during Tavily search for query '{query}': {e}")
        return {"error": f"Tavily search failed: {e}"}

@tool
def duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Performs a web search using the DuckDuckGo Search API.
    Returns a list of search result dictionaries, each containing 'title', 'href' (URL), and 'body' (snippet).
    Useful for alternative web search results or when Tavily doesn't provide enough information.
    """
    ddg_client = config.get_duckduckgo_client() # Get client inside function
    if not ddg_client:
        logger.error("DuckDuckGo client is not available.")
        return [{"error": "DuckDuckGo client unavailable."}]
    try:
        logger.info(f"Performing DuckDuckGo search for query: '{query}'")
        results = ddg_client.text(query, max_results=max_results)
        logger.info(f"DuckDuckGo search completed. Found {len(results)} results.")
        for result in results:
            result['source_tool'] = 'DuckDuckGo'
        return results
    except Exception as e:
        logger.error(f"Error during DuckDuckGo search for query '{query}': {e}")
        return [{"error": f"DuckDuckGo search failed: {e}"}]

@tool
def news_search(query: str, language: str = 'en', page_size: int = 10) -> Dict[str, Any]:
    """
    Searches for recent news articles related to a query using the NewsAPI.
    Returns a dictionary containing a list of articles, including titles, URLs, descriptions, and publication dates.
    Use this specifically for finding recent news coverage about a topic or claim.
    """
    news_client = config.get_news_api_client() # Get client inside function
    if not news_client:
        logger.error("NewsAPI client is not available.")
        return {"error": "NewsAPI client unavailable."}
    try:
        logger.info(f"Performing NewsAPI search for query: '{query}'")
        response = news_client.get_everything(q=query, language=language, sort_by='relevancy', page_size=page_size)
        logger.info(f"NewsAPI search completed. Found {response.get('totalResults', 0)} articles.")
        if 'articles' in response:
             for article in response['articles']:
                 article['source_tool'] = 'NewsAPI'
                 # Standardize keys slightly if needed (e.g., 'url' instead of 'href')
                 article['href'] = article.get('url')
                 article['body'] = article.get('description') or article.get('content')
        else:
            # Ensure 'articles' key exists even if empty
            response['articles'] = []
        return response
    except Exception as e:
        logger.error(f"Error during NewsAPI search for query '{query}': {e}")
        return {"error": f"NewsAPI search failed: {e}", "articles": []}

@tool
def wikidata_entity_search(query: str, limit: int = 3) -> Dict[str, Any]:
    """
    Searches for entities (people, places, organizations, concepts) on Wikidata.
    Returns a dictionary containing potential matches with descriptions and URIs.
    Use this to find structured information or confirm details about specific entities mentioned in a claim.
    """
    wikidata_query_func = config.get_wikidata_client() # Get client inside function
    if not wikidata_query_func:
        logger.error("Wikidata query function is not available.")
        return {"error": "Wikidata client unavailable."}
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
        else:
            results['search'] = [] # Ensure key exists
        return results
    except Exception as e:
        logger.error(f"Error during Wikidata entity search for '{query}': {e}")
        return {"error": f"Wikidata search failed: {e}", "search": []}


@tool
def gemini_google_search_and_parse(claim: str) -> Dict[str, Any]:
    """
    Performs a Google Search via Gemini about a specific claim and parses the output.
    Returns a dictionary containing a summary, key facts, and source URLs identified by Gemini.
    Use this for an initial, comprehensive investigation directly addressing the claim.
    Input should be the claim itself.
    """
    logger.info(f"Starting Gemini+Google Search and Parse tool for claim: '{claim}'")
    raw_gemini_output = config.perform_gemini_google_search(claim) # Get client implicitly via config call

    if raw_gemini_output is None or raw_gemini_output.startswith("Error:"):
        logger.error(f"Gemini search failed or returned an error: {raw_gemini_output}")
        return {"error": f"Gemini search failed: {raw_gemini_output}"}

    # Call the internal parsing function
    parsed_output: Optional[GeminiParsedOutput] = parse_gemini_output_with_llm(raw_gemini_output, claim)

    if parsed_output is None:
        logger.error("Failed to parse the output from Gemini search.")
        # Return raw output along with error? Or just error?
        return {"error": "Failed to parse Gemini output.", "raw_output": raw_gemini_output[:1000]} # Return partial raw for context

    logger.info("Gemini search and parse tool completed successfully.")
    # Convert Pydantic model to dict for consistent tool output type
    output_dict = parsed_output.dict()
    output_dict['source_tool'] = 'Gemini+GoogleSearch'
    return output_dict


# --- Deduplication Helper (Keep as internal utility) ---
def deduplicate_results(results: List[Dict[str, Any]], key: str = 'href') -> List[Dict[str, Any]]:
    """
    Deduplicates a list of result dictionaries based on a specific key (defaulting to 'href' for URL).
    Internal helper function.
    """
    seen = set()
    unique_results = []
    for result in results:
        value = result.get(key)
        # Ensure value is hashable (e.g., not a list/dict) and not None
        if value is not None and isinstance(value, (str, int, float, bool, tuple)):
            if value not in seen:
                seen.add(value)
                unique_results.append(result)
        elif value is None:
             # Decide whether to keep items with None key
             # unique_results.append(result) # Option: Keep them
             pass # Option: Discard them
        else: # Unhashable type
            # Try to convert to string or skip
            try:
                str_value = str(value)
                if str_value not in seen:
                    seen.add(str_value)
                    unique_results.append(result)
            except Exception:
                 logger.warning(f"Could not process value for deduplication key '{key}': {value}. Skipping result.")


    logger.info(f"Deduplicated results based on key '{key}'. Original: {len(results)}, Unique: {len(unique_results)}.")
    return unique_results

# --- Verification functions (Removed from agent flow for now) ---
# def verify_duckduckgo_results(...) -> ...: ...
# def verify_news_results(...) -> ...: ...

# --- Placeholder Tool ---

@tool
def scrape_webpages_tool(urls: List[str]) -> str:
    """
    Placeholder tool to simulate scraping web pages. 
    Takes a list of URLs and returns a generic message indicating content would be scraped.
    Replace this with a real implementation (e.g., using Playwright or BeautifulSoup).
    """
    logger.info(f"Placeholder: Pretending to scrape {len(urls)} URLs: {urls}")
    # In a real implementation, you'd fetch and parse content here.
    return f"Scraped content for URLs: {', '.join(urls)}" 

@tool
def verification_tool(intermediate_steps: list, original_claim: str) -> str:
    """
    Verifies the collected evidence and intermediate conclusions against the original claim.
    Use this tool ONLY when you have gathered sufficient evidence and synthesized a preliminary answer.
    This tool assesses the consistency, relevance, and sufficiency of the evidence.
    Provide the original claim and all intermediate steps (tool calls and observations).
    The intermediate_steps argument should be a list of dictionaries, each with keys 'tool', 'tool_input', and 'observation'.
    Returns a verification assessment string (e.g., "Evidence supports the claim", "Evidence contradicts the claim", "Evidence is insufficient or mixed").
    """
    # Use the specific getter function from config
    verification_model = config.get_primary_llm()
    if not verification_model:
        logger.warning("Verification model (using primary LLM) not configured. Skipping verification.")
        return "Verification model not configured. Skipping verification."

    # Format the intermediate steps (which are dicts) for the verification prompt
    formatted_steps = "\n".join([
        f"Tool Call: {step.get('tool', 'N/A')}({step.get('tool_input', 'N/A')})\nObservation: {step.get('observation', 'N/A')}"
        for step in intermediate_steps  # Iterate through the list of dictionaries
    ])

    prompt_content = VERIFICATION_PROMPT.format(
        claim=original_claim,
        evidence=formatted_steps
    )

    try:
        response = verification_model.invoke(prompt_content)
        assessment = response.content if hasattr(response, 'content') else str(response)
        logging.info(f"Verification Tool Assessment: {assessment}")
        return assessment
    except Exception as e:
        logging.error(f"Error during verification tool execution: {e}")
        return f"Error during verification: {e}"


def create_agent_tools(config: dict) -> List[BaseTool]:
    """Creates and returns a list of tools available to the agent."""
    tools = []
    if config.get("enable_tavily", True):
        try:
            tavily_search_tool = TavilySearchResults(max_results=config.get("tavily_max_results", 5))
            tavily_search_tool.name = "tavily_search_results_json" # Default name
            tools.append(tavily_search_tool)
            logging.info("Tavily Search tool enabled.")
        except Exception as e:
            logging.warning(f"Failed to initialize Tavily Search tool: {e}")

    # Add other tools based on config...
    if config.get("enable_wikidata", False):
        try:
            wikidata_tool_wrapper = WikidataAPIWrapper()
            tools.append(
                 Tool(
                    name="Wikidata",
                    func=wikidata_tool_wrapper.run,
                    description="Useful for querying Wikidata for structured data about entities, concepts, or facts. Input should be a Wikidata query or entity ID.",
                )
            )
            logging.info("Wikidata tool enabled.")
        except Exception as e:
            logging.warning(f"Failed to initialize Wikidata tool: {e}")

    if config.get("enable_duckduckgo", False):
        try:
            # Requires 'duckduckgo-search' package
            from langchain_community.tools import DuckDuckGoSearchRun
            ddg_search = DuckDuckGoSearchRun()
            tools.append(ddg_search)
            logging.info("DuckDuckGo Search tool enabled.")
        except ImportError:
             logging.warning("DuckDuckGo Search tool requires 'duckduckgo-search'. Skipping.")
        except Exception as e:
            logging.warning(f"Failed to initialize DuckDuckGo Search tool: {e}")

    if config.get("enable_google_search", False):
        try:
            # Requires 'google-search-results' package and SERPAPI_API_KEY
            from langchain_community.tools import GoogleSearchRun
            google_search = GoogleSearchRun()
            tools.append(google_search)
            logging.info("Google Search tool enabled.")
        except ImportError:
            logging.warning("Google Search tool requires 'google-search-results'. Skipping.")
        except Exception as e:
            logging.warning(f"Failed to initialize Google Search tool: {e} (Ensure SERPAPI_API_KEY is set)")

    if config.get("enable_web_scraper", True):
        # Add the placeholder scrape_webpages_tool defined above
        tools.append(scrape_webpages_tool)
        logging.info("Web Scraper tool enabled (Placeholder).")

    # Add the verification tool
    if config.get("enable_verification_tool", True):
        tools.append(verification_tool)
        logging.info("Verification tool enabled.")

    # Add FINISH tool to signal completion and trigger synthesis
    def finish_func(*args, **kwargs):
        """No-op tool: signals the agent to finish and synthesize final answer."""
        return ""
    tools.append(
        Tool(
            name="FINISH",
            func=finish_func,
            description="Signal that the agent has finished research and wants to synthesize the final answer."
        )
    )
    logging.info("FINISH tool enabled.")

    if not tools:
        logging.warning("No tools were enabled for the agent!")
        # Add a fallback basic search if nothing else is enabled?
        # tools.append(DuckDuckGoSearchRun()) # Example fallback

    return tools