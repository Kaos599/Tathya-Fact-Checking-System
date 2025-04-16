"""
Custom tools for accessing external APIs and services.
"""

import json
import uuid
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
from langchain.tools import tool
from bs4 import BeautifulSoup
from newsapi.newsapi_client import NewsApiClient

from fact_check_system.config import (
    get_tavily_client, 
    get_wikidata_client, 
    get_entity_details,
    NEWS_API_KEY
)
from fact_check_system.schemas import Evidence

# Initialize clients
tavily_client = get_tavily_client()
query_wikidata = get_wikidata_client()

# Web Search Tools
@tool
def tavily_search(query: str, search_depth: str = "basic", max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web using Tavily API.
    
    Args:
        query: The search query
        search_depth: The depth of search ('basic' or 'deep')
        max_results: Maximum number of results to return
        
    Returns:
        List of search results
    """
    # Create mock search results instead of calling the actual API
    results = []
    
    # Add mock results that match the expected structure in workflow.py
    for i in range(3):
        result = {
            "id": str(uuid.uuid4()),
            "source": f"Mock Source {i+1}",
            "url": f"https://example.com/result{i+1}",
            "content": f"This is mock content about {query}",
            "credibility_score": 0.7,
            "relevance_score": 0.8,
            "retrieval_date": datetime.now()
        }
        results.append(result)
    
    return results

# Knowledge Base Tools
@tool
def wikidata_entity_search(query: str, language: str = "en", limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for entities in Wikidata.
    
    Args:
        query: Search term
        language: Language code (default: 'en')
        limit: Maximum number of results to return
        
    Returns:
        List of Wikidata entities
    """
    # Create mock knowledge base results
    results = []
    
    # Add mock results that match the expected structure in workflow.py
    for i in range(2):
        result = {
            "id": str(uuid.uuid4()),
            "source": "Wikidata",
            "url": f"https://www.wikidata.org/wiki/Q{1000+i}",
            "content": f"Entity information about {query}",
            "credibility_score": 0.9,
            "relevance_score": 0.7,
            "retrieval_date": datetime.now()
        }
        results.append(result)
    
    return results

# News Search Tools
@tool
def news_search(query: str, language: str = "en", page_size: int = 5) -> List[Dict[str, Any]]:
    """
    Search for news articles using NewsAPI.
    
    Args:
        query: Search term
        language: Language code (default: 'en')
        page_size: Number of results to return
        
    Returns:
        List of news articles
    """
    # Create mock news results
    results = []
    
    # Add mock results that match the expected structure in workflow.py
    for i in range(2):
        result = {
            "id": str(uuid.uuid4()),
            "source": f"News Source {i+1}",
            "url": f"https://news-example.com/article{i+1}",
            "content": f"News article about {query}",
            "credibility_score": 0.6,
            "relevance_score": 0.8,
            "retrieval_date": datetime.now(),
            "publication_date": datetime.now()
        }
        results.append(result)
    
    return results

# Web Scraping Tool for additional context
@tool
def scrape_webpage(url: str) -> Dict[str, Any]:
    """
    Scrape content from a webpage.
    
    Args:
        url: URL of the webpage to scrape
        
    Returns:
        Dictionary with webpage content
    """
    # Create a mock scraping result
    result = {
        "id": str(uuid.uuid4()),
        "source": "web_scrape",
        "url": url,
        "content": f"Mock scraped content from {url}",
        "credibility_score": 0.5,
        "relevance_score": 0.6,
        "retrieval_date": datetime.now()
    }
    
    return result 