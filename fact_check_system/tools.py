"""
Custom tools for accessing external APIs and services.
"""

import json
import uuid
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import time
import logging
from langchain.tools import tool
from bs4 import BeautifulSoup
from newsapi.newsapi_client import NewsApiClient

from fact_check_system.config import (
    get_tavily_client, 
    get_wikidata_client, 
    get_entity_details,
    get_duckduckgo_client,
    get_news_api_client,
    NEWS_API_KEY,
    TAVILY_API_KEY
)
from fact_check_system.schemas import Evidence

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
try:
    tavily_client = get_tavily_client() if TAVILY_API_KEY else None
except Exception as e:
    logger.warning(f"Failed to initialize Tavily client: {str(e)}")
    tavily_client = None

try:
    ddg_client = get_duckduckgo_client()
except Exception as e:
    logger.warning(f"Failed to initialize DuckDuckGo client: {str(e)}")
    ddg_client = None

try:
    news_api = get_news_api_client() if NEWS_API_KEY else None
except Exception as e:
    logger.warning(f"Failed to initialize NewsAPI client: {str(e)}")
    news_api = None

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
    if not tavily_client:
        logger.error("Tavily client not initialized")
        return []
    
    try:
        search_results = tavily_client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results
        )
        
        results = []
        for i, result in enumerate(search_results.get("results", [])):
            evidence = {
                "id": str(uuid.uuid4()),
                "source": "Tavily",
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "credibility_score": result.get("score", 0.7),
                "relevance_score": result.get("score", 0.7),
                "retrieval_date": datetime.now()
            }
            results.append(evidence)
        
        return results
    except Exception as e:
        logger.error(f"Error in Tavily search: {str(e)}")
        return []

@tool
def duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web using DuckDuckGo.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List of search results
    """
    if not ddg_client:
        logger.error("DuckDuckGo client not initialized")
        return []
    
    try:
        search_results = ddg_client.text(query, max_results=max_results)
        
        results = []
        for i, result in enumerate(search_results):
            evidence = {
                "id": str(uuid.uuid4()),
                "source": "DuckDuckGo",
                "url": result.get("href", ""),
                "content": result.get("body", ""),
                "credibility_score": 0.7,  # Default credibility for DuckDuckGo
                "relevance_score": 0.7,    # Default relevance for DuckDuckGo
                "retrieval_date": datetime.now()
            }
            results.append(evidence)
        
        return results
    except Exception as e:
        logger.error(f"Error in DuckDuckGo search: {str(e)}")
        return []

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
    try:
        wikidata_results = query_wikidata(search_term=query, language=language, limit=limit)
        
        results = []
        for entity in wikidata_results.get("search", []):
            entity_id = entity.get("id", "")
            
            # Get more details about the entity
            try:
                entity_details = get_entity_details(entity_id, language)
                entity_data = entity_details.get("entities", {}).get(entity_id, {})
                
                # Extract descriptions and other data
                description = entity_data.get("descriptions", {}).get(language, {}).get("value", "")
                labels = entity_data.get("labels", {}).get(language, {}).get("value", "")
                
                content = f"Entity ID: {entity_id}, Label: {labels}, Description: {description}"
                
                # Add claims/properties if available
                if "claims" in entity_data:
                    content += "\nProperties:"
                    for prop_id, prop_values in list(entity_data["claims"].items())[:5]:  # Limit to 5 properties
                        prop_value = prop_values[0].get("mainsnak", {}).get("datavalue", {}).get("value", "")
                        content += f"\n- {prop_id}: {prop_value}"
                
            except Exception as e:
                # If details retrieval fails, use basic information
                description = entity.get("description", "")
                labels = entity.get("label", "")
                content = f"Entity ID: {entity_id}, Label: {labels}, Description: {description}"
            
            evidence = {
                "id": str(uuid.uuid4()),
                "source": "Wikidata",
                "url": f"https://www.wikidata.org/wiki/{entity_id}",
                "content": content,
                "credibility_score": 0.9,  # Default credibility for Wikidata
                "relevance_score": 0.8,    # Default relevance
                "retrieval_date": datetime.now()
            }
            
            results.append(evidence)
        
        return results
    except Exception as e:
        logger.error(f"Error in Wikidata search: {str(e)}")
        return []

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
    if not news_api:
        logger.error("NewsAPI client not initialized")
        return []
    
    try:
        # Search for news articles
        news_results = news_api.get_everything(
            q=query,
            language=language,
            sort_by="relevancy",
            page_size=page_size
        )
        
        results = []
        for article in news_results.get("articles", []):
            evidence = {
                "id": str(uuid.uuid4()),
                "source": "NewsAPI",
                "url": article.get("url", ""),
                "title": article.get("title", ""),
                "content": article.get("description", "") + "\n" + (article.get("content", "") or ""),
                "author": article.get("author", ""),
                "published_at": article.get("publishedAt", ""),
                "credibility_score": 0.75,  # Default credibility for news
                "relevance_score": 0.8,     # Default relevance
                "retrieval_date": datetime.now()
            }
            results.append(evidence)
        
        return results
    except Exception as e:
        logger.error(f"Error in NewsAPI search: {str(e)}")
        return []

# Webpage Processing Tools
@tool
def scrape_webpage(url: str) -> Dict[str, Any]:
    """
    Scrape content from a webpage.
    
    Args:
        url: The URL to scrape
        
    Returns:
        Dictionary containing scraped content
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract title
            title = soup.title.text.strip() if soup.title else ""
            
            # Extract main content (this is a simplified approach)
            # For production, use a more sophisticated content extraction
            main_content = ""
            
            # Try to find main content containers
            content_elements = soup.select("article, .article, .content, #content, .post, .entry")
            
            if content_elements:
                for element in content_elements:
                    paragraphs = element.select("p")
                    for p in paragraphs:
                        main_content += p.text.strip() + "\n\n"
            else:
                # Fallback to all paragraphs
                paragraphs = soup.select("p")
                for p in paragraphs:
                    main_content += p.text.strip() + "\n\n"
            
            return {
                "url": url,
                "title": title,
                "content": main_content,
                "retrieved_at": datetime.now()
            }
        else:
            logger.error(f"Failed to scrape webpage: {url}, status code: {response.status_code}")
            return {"url": url, "error": f"Failed with status code: {response.status_code}"}
    except Exception as e:
        logger.error(f"Error scraping webpage {url}: {str(e)}")
        return {"url": url, "error": str(e)}

# Entity and Metadata Extraction Tools
@tool
def entity_extraction(text: str) -> List[str]:
    """
    Extract named entities from text.
    
    Args:
        text: The input text
        
    Returns:
        List of extracted entities
    """
    # This is a simplified implementation
    # For production, use a proper NER model or service
    try:
        # Simple extraction based on capitalization patterns
        words = text.split()
        capitalized_words = []
        
        for i in range(len(words)):
            word = words[i].strip(".,;:!?()[]{}")
            
            # Check if word starts with capital letter (not at the beginning of a sentence)
            if word and word[0].isupper() and (i > 0 or len(word) > 1):
                # Check for multi-word entities
                entity = word
                j = i + 1
                while j < len(words) and words[j].strip(".,;:!?()[]{}")[0:1].isupper():
                    entity += " " + words[j].strip(".,;:!?()[]{}")
                    j += 1
                
                if entity not in capitalized_words:
                    capitalized_words.append(entity)
        
        return capitalized_words
    except Exception as e:
        logger.error(f"Error in entity extraction: {str(e)}")
        return []

@tool
def date_extraction(text: str) -> List[str]:
    """
    Extract dates from text.
    
    Args:
        text: The input text
        
    Returns:
        List of extracted dates
    """
    # This is a simplified implementation
    # For production, use a dedicated date extraction library
    try:
        import re
        
        # Define regex patterns for common date formats
        patterns = [
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',  # MM-DD-YYYY or DD-MM-YYYY
            r'\b\d{4}/\d{1,2}/\d{1,2}\b',    # YYYY/MM/DD
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',    # YYYY-MM-DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
            r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',    # DD Month YYYY
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b',  # Full month
            r'\b\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b',    # DD Full month YYYY
            r'\b\d{4}\b'  # Just year (less precise)
        ]
        
        dates = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates.extend(matches)
        
        return dates
    except Exception as e:
        logger.error(f"Error in date extraction: {str(e)}")
        return []

@tool
def url_extraction(text: str) -> List[str]:
    """
    Extract URLs from text.
    
    Args:
        text: The input text
        
    Returns:
        List of extracted URLs
    """
    try:
        import re
        
        # URL regex pattern
        pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[-\w%!./?=&]*)?'
        
        urls = re.findall(pattern, text)
        return urls
    except Exception as e:
        logger.error(f"Error in URL extraction: {str(e)}")
        return []

# Add wrapper functions with the names expected by workflow.py
def tavily_search_tool(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Wrapper for tavily_search function."""
    search_depth = "basic" if isinstance(max_results, int) else max_results
    max_results = max_results if isinstance(max_results, int) else 5
    return tavily_search.invoke({"query": query, "search_depth": search_depth, "max_results": max_results})

def duckduckgo_search_tool(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Wrapper for duckduckgo_search function."""
    return duckduckgo_search.invoke({"query": query, "max_results": max_results})

def news_api_search_tool(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Wrapper for news_search function."""
    return news_search.invoke({"query": query, "language": "en", "page_size": max_results})

def wikidata_tool(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Wrapper for wikidata_entity_search function."""
    return wikidata_entity_search.invoke({"query": query, "limit": max_results})

def entity_extraction_tool(text: str) -> List[str]:
    """Wrapper for entity_extraction function."""
    return entity_extraction.invoke({"text": text})

def date_extraction_tool(text: str) -> List[str]:
    """Wrapper for date_extraction function."""
    return date_extraction.invoke({"text": text})

def url_extraction_tool(text: str) -> List[str]:
    """Wrapper for url_extraction function."""
    return url_extraction.invoke({"text": text}) 