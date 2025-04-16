"""
Configuration for the fact-checking system.
"""

import os
import requests
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv
from tavily import TavilyClient

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# Load environment variables
load_dotenv()

# API keys and configurations
# Azure OpenAI
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

# Tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

# News API
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Wikidata API settings
WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php"

# Confidence score thresholds
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.8"))
MEDIUM_CONFIDENCE_THRESHOLD = float(os.getenv("MEDIUM_CONFIDENCE_THRESHOLD", "0.5"))
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.3"))

# Initialize Google Gemini API if key is available
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Factory functions for clients
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

def get_secondary_llm(streaming: bool = False):
    """Get the secondary LLM client."""
    if GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.2,
            top_p=0.95,
            convert_system_message_to_human=True,
            streaming=streaming
        )
    elif AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            deployment_name=AZURE_OPENAI_DEPLOYMENT,
            openai_api_key=AZURE_OPENAI_API_KEY,
        )
    else:
        raise ValueError("No API keys available for secondary LLM client.")

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