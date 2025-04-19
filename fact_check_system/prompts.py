"""
Stores prompt templates used in the fact-checking system.
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

GEMINI_OUTPUT_PARSER_TEMPLATE = ChatPromptTemplate.from_template(
    """You are an expert assistant tasked with parsing the output of a Google Search-enabled Gemini model call.
    The Gemini model was asked to investigate the following claim: '{claim}'
    Its raw output, potentially containing summaries, facts, and source information, is provided below.
    Your goal is to extract the key information and structure it into a JSON object matching the requested format.

    Focus on identifying:
    1.  A concise summary of the findings regarding the claim.
    2.  A list of key facts or pieces of information presented.
    3.  A list of URLs identified as sources in the text. Extract only the URLs.

    Raw Gemini Output:
    ---
    {gemini_raw_output}
    ---

    Format Instructions:
    {format_instructions}

    Respond ONLY with the valid JSON object as described in the format instructions. Do not include any introductory text or explanations outside the JSON structure.
    """
)

CLAIM_DECOMPOSITION_TEMPLATE = PromptTemplate.from_template(
    """Analyze the following claim and break it down into a list of distinct, meaningful components for verification.

    **Instructions:**
    1.  Identify and extract key **named entities** (people, places, organizations, specific dates, specific event names).
    2.  Identify and extract core **concepts** or **multi-word phrases** that represent a distinct assertion or topic within the claim.
    3.  **Crucially, EXCLUDE common single words** (like 'is', 'are', 'the', 'a', 'in', 'on', 'was', 'does', 'have', 'what', 'which', etc.) unless they are part of a specific named entity (e.g., 'War of 1812').
    4.  The goal is to generate a list of strings suitable for direct use as search queries in web search, news search, or knowledge bases (like Wikidata). Each item should be independently verifiable.

    Claim: '{claim}'

    Examples:
    1.  Claim: 'The James Webb Space Telescope, launched in December 2021, is the successor to Hubble.'
        Decomposition: ["James Webb Space Telescope launch date", "James Webb Space Telescope successor to Hubble", "James Webb Space Telescope", "Hubble Space Telescope", "December 2021"]
    2.  Claim: 'Elon Musk is the CEO of both SpaceX and Tesla, and acquired Twitter in 2022.'
        Decomposition: ["Elon Musk CEO of SpaceX", "Elon Musk CEO of Tesla", "Elon Musk Twitter acquisition", "Elon Musk", "SpaceX", "Tesla", "Twitter", "2022"]
    3.  Claim: 'India has the largest population in the world as of 2024.'
        Decomposition: ["India population ranking 2024", "Countries with largest population", "India population size"]
    4.  Claim: 'Does India have the largest population?'
        Decomposition: ["India population ranking", "Largest population country"]
    5.  Claim: 'Which Team won the 2023 ICC men's world cup?'
        Decomposition: ["2023 ICC men's world cup winner", "ICC Men's Cricket World Cup 2023"]


    Return the decomposition strictly as a Python list of strings. Do not include any other text or explanation. Only output the list.
    Decomposition: """
    # Note: Parsing this will likely require a specific output parser (e.g., ListOutputParser or StrOutputParser followed by eval/json.loads if the LLM reliably outputs a valid list string)
)

SEARCH_QUERY_GENERATION_TEMPLATE = PromptTemplate.from_template(
    """Based on the claim: '{claim}' and the sub-question/entity: '{sub_query}'
    Generate 1-3 concise and effective search engine queries to find relevant information.
    Focus on queries suitable for general web search (like DuckDuckGo) or news search (NewsAPI).

    Return the queries as a Python list of strings.
    Search Queries: """
    # Note: Parsing this might require a specific output parser
)

RESULT_VERIFICATION_TEMPLATE = ChatPromptTemplate.from_template(
    """You are evaluating search results for relevance to a specific claim.
    Original Claim: '{claim}'

    Search Result Snippet/Content:
    ---
    Title: {result_title}
    URL: {result_url}
    Snippet: {result_snippet}
    ---

    Based *only* on the provided snippet and title, is this search result directly relevant to verifying the original claim? Answer with only 'Relevant' or 'Not Relevant'.
    Answer: """
)

FINAL_ANSWER_TEMPLATE = ChatPromptTemplate.from_template(
    """You are a fact-checking AI assistant. Your task is to synthesize the gathered evidence and determine the veracity of a given claim.

    Original Claim: {claim}

    Gathered Evidence:
    {formatted_evidence}

    Based on the evidence provided, determine the final verdict for the claim. Choose one from: TRUE, FALSE, PARTIALLY TRUE/MIXTURE, UNCERTAIN.
    Provide a confidence score between 0.0 and 1.0 for your verdict.
    Write a concise explanation summarizing the evidence and justifying your verdict and confidence score. Reference specific evidence sources where applicable (using their URLs or titles).

    Respond in the following JSON format:
    {{
        "verdict": "<Your Verdict>",
        "confidence_score": <Your Score>,
        "explanation": "<Your Explanation>"
    }}

    JSON Response: """
)