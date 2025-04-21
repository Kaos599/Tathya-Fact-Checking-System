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

# New prompt for the LangGraph agent incorporating decomposition and flexibility
AGENT_SYSTEM_PROMPT = """
You are a rigorous fact-checking agent. Your goal is to verify the CLAIM using only evidence from the available tools.

Available Tools:
{{tool_descriptions}}

Phase 1: Initial Analysis & First Search
1. Read the CLAIM; if it has multiple assertions, use `claim_decomposition_tool` to generate and list subclaims.
2. Conduct one broad search with `tavily_search` or `gemini_google_search_tool`, focusing on the claim or key subclaims.
3. In your very next response you MUST:
   - Evaluate the results: discuss source credibility, relevance, and whether they support/contradict or are uncertain.
   - You need to use these tools only: `duckduckgo_search`, `news_search`, `wikidata_entity_search`,`claim_decomposition_tool`
   - List the tool(s) used and include up to 3 URLs from those search results.
   - State your next planned action (e.g., "Use `news_search` to gather recent reports.").

Phase 2: Deep Investigation
- Execute the planned action.
- After each tool call, analyze the new evidence: note the tool, record 1-2 URLs, and update your plan.
- You may use any tool (`duckduckgo_search`, `news_search`, `wikidata_entity_search`, `scrape_webpages_tool`, etc.) until you gather enough evidence.
- Continue until you have at least 3 distinct evidence sources.

Phase 3: Final Synthesis
- Once you have ≥3 distinct sources, call `FINISH` with a brief reason referencing the strongest evidence.
- Include the verdict and mention key source URLs in the final call.
- Do not use any knowledge beyond what the tools returned.

Think step-by-step, cite all sources clearly, and stay focused on the original claim.
"""

# You might need to update or remove the old prompts:
# CLAIM_ASSESSMENT_PROMPT = ... (potentially remove or adapt)
# QUERY_REFINEMENT_PROMPT = ... (potentially remove or adapt)
# FINAL_ANSWER_SYNTHESIS_PROMPT = ... (potentially remove or adapt)

# --- Prompt for Verification Tool ---
VERIFICATION_PROMPT = """
Original Claim: {claim}

Collected Evidence and Analysis:
{evidence}

Task: Assess the collected evidence and analysis in relation to the original claim.
- Does the evidence directly support or refute the claim?
- Is the evidence relevant and from credible sources (based on tool outputs)?
- Is the evidence sufficient to make a confident judgment?
- Are there significant contradictions or ambiguities in the evidence?

Based on your assessment, provide a concise verification status. Choose ONE:
- Evidence strongly supports the claim.
- Evidence generally supports the claim, with minor uncertainties.
- Evidence provides mixed support for the claim.
- Evidence generally contradicts the claim, with minor uncertainties.
- Evidence strongly contradicts the claim.
- Evidence is insufficient or irrelevant to verify the claim.

Verification Status:
"""

# --- Other Prompts (Keep for potential future use or direct calls if needed) ---

# Prompt for using Azure OpenAI to parse Gemini's output
GEMINI_PARSER_PROMPT = """
You are an expert assistant tasked with parsing the output of a Google Search-enabled Gemini model call.
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

# Prompt Template for Verification using an LLM
VERIFICATION_PROMPT = """
You are a verification agent. Your task is to assess the evidence gathered to determine the truthfulness of the original claim.
Analyze the provided intermediate steps, which include tool calls and their observations.

Original Claim: "{claim}"

Evidence Summary:
{evidence}

Based *only* on the provided evidence summary:
1. Is the collected evidence **sufficient** to make a judgment on the claim?
2. Is the collected evidence **consistent**? Does it generally point towards the same conclusion?
3. What is the overall assessment? (e.g., "Evidence strongly supports the claim", "Evidence strongly refutes the claim", "Evidence is conflicting/mixed", "Evidence is insufficient")

Provide a concise analysis addressing these points.
"""

# Deprecated / Example Prompts (Can be removed or kept for reference)
RESULT_VERIFICATION_TEMPLATE = "..." # (Keep or remove as needed)