"""
Configuration settings for the fact-checking system.
"""

import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult
from tavily import TavilyClient
import requests
import json
import google.generativeai as genai

# Load environment variables
load_dotenv()

# API Keys and Settings
# Azure OpenAI
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

# Tavily
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

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

# Mock LLM implementation to avoid API errors
class MockLLM(BaseChatModel):
    """Mock LLM implementation for testing without real API access."""
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a mock response based on input messages."""
        # Concatenate all messages to create a prompt
        prompt = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
        
        # Create a simple response based on the prompt
        if "fact-checking" in prompt.lower():
            response = "This claim is TRUE with a confidence score of 0.9. The evidence strongly supports this statement."
        elif "claim decomposition" in prompt.lower():
            response = """I'll break this claim down into its atomic components:
            
            {
              "claims": [
                {
                  "id": "1234-5678-abcd",
                  "statement": "The Earth orbits the Sun",
                  "entities": ["Earth", "Sun"],
                  "time_references": [],
                  "location_references": [],
                  "numeric_values": [],
                  "importance": 1.0
                }
              ]
            }
            """
        else:
            response = "I'm a mock LLM response for testing purposes."
            
        # Create and return chat result
        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    @property
    def _llm_type(self) -> str:
        return "mock-llm"

# Factory functions for clients
def get_primary_llm(streaming: bool = False):
    """Get the primary LLM client."""
    if GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.1,
            top_p=0.95,
            convert_system_message_to_human=True,
            streaming=streaming
        )
    else:
        print("Warning: Using MockLLM because GEMINI_API_KEY is not set")
        return MockLLM()

def get_secondary_llm(streaming: bool = False):
    """Get the secondary LLM client."""
    if GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.2,  # Slightly higher temperature for diversity
            top_p=0.95,
            convert_system_message_to_human=True,
            streaming=streaming
        )
    else:
        print("Warning: Using MockLLM because GEMINI_API_KEY is not set")
        return MockLLM()

def get_tavily_client():
    """Get the Tavily search client."""
    return TavilyClient(api_key=TAVILY_API_KEY)

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