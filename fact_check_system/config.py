"""
Configuration for the fact-checking system.
"""

import os
import requests
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv
from tavily import TavilyClient
from google.genai import types
from google.ai import generativelanguage as glm # Use this for types if needed

try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    genai = None
    ChatGoogleGenerativeAI = None
    logging.warning("google.generativeai or langchain_google_genai not installed. Gemini functionality will be limited.")

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY_O3_mini")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT_O3_mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION_O3_mini")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_O3_mini")

AZURE_OPENAI_API_KEY_ALT = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT_ALT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION_ALT = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_ALT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY2")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php"

HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.8"))
MEDIUM_CONFIDENCE_THRESHOLD = float(os.getenv("MEDIUM_CONFIDENCE_THRESHOLD", "0.5"))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.3"))


if GEMINI_API_KEY and genai: # Check if genai was imported successfully
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        logging.warning(f"Failed to configure google.genai: {e}")
else:
    logging.warning("GEMINI_API_KEY not set or google.generativeai not installed. Cannot configure Gemini client.")

def get_primary_llm(streaming: bool = False):
    """Get the primary LLM client."""
    if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            deployment_name=AZURE_OPENAI_DEPLOYMENT,
            openai_api_key=AZURE_OPENAI_API_KEY,
        )
    else:
        raise ValueError("AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set. Cannot create primary LLM client.")


def get_tavily_client():
    """Get the Tavily search client."""
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY not set. Cannot create Tavily client.")
    return TavilyClient(api_key=TAVILY_API_KEY)

def get_duckduckgo_client():
    """Get the DuckDuckGo search client."""
    from duckduckgo_search import DDGS
    return DDGS()

def get_news_api_client():
    """Get the NewsAPI client."""
    if not NEWS_API_KEY:
        raise ValueError("NEWS_API_KEY not set. Cannot create NewsAPI client.")
    
    from newsapi import NewsApiClient
    return NewsApiClient(api_key=NEWS_API_KEY)

def get_wikidata_client():
    """Get a function to query the Wikidata API."""
    
    def query_wikidata(search_term: str, language: str = "en", limit: int = 5):
        """Query the Wikidata API for entities."""
        params = {
            'action': 'wbsearchentities',
            'format': 'json',
            'language': language,
            'search': search_term,
            'limit': limit
        }
        
        response = requests.get(WIKIDATA_API_ENDPOINT, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Wikidata API request failed with status code {response.status_code}")
    
    return query_wikidata

def get_entity_details(entity_id: str, language: str = "en"):
    """Get detailed information about a Wikidata entity."""
    params = {
        'action': 'wbgetentities',
        'format': 'json',
        'ids': entity_id,
        'languages': language
    }
    
    response = requests.get(WIKIDATA_API_ENDPOINT, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Wikidata API request failed with status code {response.status_code}")


def get_azure_openai_parser_llm():
    """Get the Azure OpenAI LLM client using alternate credentials for parsing tasks."""
    if AZURE_OPENAI_API_KEY_ALT and AZURE_OPENAI_ENDPOINT_ALT:
        from langchain_openai import AzureChatOpenAI
        logging.info(f"Initializing Azure OpenAI parser LLM with endpoint: {AZURE_OPENAI_ENDPOINT_ALT} and deployment: {AZURE_OPENAI_DEPLOYMENT_ALT}")
        return AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT_ALT,
            openai_api_version=AZURE_OPENAI_API_VERSION_ALT,
            deployment_name=AZURE_OPENAI_DEPLOYMENT_ALT,
            openai_api_key=AZURE_OPENAI_API_KEY_ALT,
            temperature=0.1, 
            max_retries=3, 
        )
    else:
        logging.error("Alternate Azure OpenAI credentials (API Key, Endpoint, Version, Deployment) not fully set. Cannot create parser LLM client.")
        raise ValueError("Alternate Azure OpenAI credentials not set.") 

import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash-preview-04-17"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    tools = [
        types.Tool(google_search=types.GoogleSearch())
    ]
    generate_content_config = types.GenerateContentConfig(
        tools=tools,
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()

def perform_gemini_google_search(claim: str) -> str:
    """
    Performs a search using the Gemini model (specified by GEMINI_MODEL env var)
    with Google Search tool enabled.

    Args:
        claim: The claim or query to search for.

    Returns:
        The text response from the Gemini model.
    """
    if not genai:
        logging.error("google.generativeai is not installed or configured.")
        return "Error: Gemini library not available."
    if not GEMINI_API_KEY:
        logging.error("GEMINI_API_KEY not set.")
        return "Error: GEMINI_API_KEY not configured."
    if not GEMINI_MODEL:
        logging.error("GEMINI_MODEL environment variable not set.")
        return "Error: GEMINI_MODEL not configured."

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=claim),
                ],
            ),
        ]
        
        tools = [
            types.Tool(google_search=types.GoogleSearch())
        ]
        
        generate_content_config = types.GenerateContentConfig(
            tools=tools,
            temperature=0.1,
            response_mime_type="text/plain",
        )
        
        logging.info(f"Sending request to Gemini model '{GEMINI_MODEL}' with Google Search enabled.")
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=generate_content_config,
        )
        
        # Extract and return the text response
        if response.candidates and response.candidates[0].content.parts:
            search_response = response.candidates[0].content.parts[0].text
            logging.info(f"Received response from Gemini: {search_response[:100]}...")
            return search_response
        else:
            logging.warning("Gemini response did not contain expected content parts.")
            return "Error: No valid response content from Gemini."

    except Exception as e:
        logging.error(f"Error during Gemini search for '{claim}': {e}")
        return f"Error: {e}"