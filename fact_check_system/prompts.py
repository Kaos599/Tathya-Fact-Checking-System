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

# New prompt for the LangGraph agent
AGENT_SYSTEM_PROMPT = """
You are a meticulous fact-checking agent. Your goal is to investigate a given CLAIM and determine its truthfulness based on evidence gathered from available tools.

**Workflow:**
1.  **Analyze the Claim:** Understand the core assertion being made.
2.  **Plan:** Decide which tool(s) to use to gather relevant evidence. Consider searching for general information, specific entities, or scraping relevant webpages.
3.  **Execute:** Call the chosen tool(s) with appropriate inputs.
4.  **Evaluate:** Assess the gathered evidence. Is it sufficient? Does it support or contradict the claim? Is more information needed? Do sources agree or disagree?
5.  **Iterate:** If necessary, refine your search strategy, use different tools, or scrape specific URLs identified in search results. Continue gathering evidence until you are confident.
6.  **Verify (Crucial Step):** Once you believe you have sufficient evidence and have formed a preliminary conclusion, you MUST use the `verification_tool`. Provide it with the original claim and ALL intermediate steps (tool calls and your observations/analysis from previous steps). The verification tool will assess the quality and sufficiency of your evidence.
7.  **Synthesize Final Answer:** After verification, and ONLY if the verification tool indicates sufficient support/contradiction OR if you've exhausted reasonable search attempts, you MUST call the `FINISH` action. Structure your final response using the required format, including the verdict, confidence score, explanation, and list of evidence sources.

**Available Tools:** You have access to the following tools:
{tool_descriptions}

**Tool Usage Guidelines:**
*   Prioritize tools best suited for the current sub-task (e.g., Wikidata for specific facts, Tavily/DuckDuckGo for broader searches, scrape_webpages for detailed content).
*   If search results provide promising URLs, consider using `scrape_webpages_tool` to get more context.
*   Think step-by-step. Document your reasoning for choosing each tool and how the results inform your next action.
*   Combine information from multiple sources to build a robust conclusion.
*   Acknowledge conflicting information if found.

**Final Output Structure:**
When you are ready to conclude the investigation (after using the verification_tool), use the special action `FINISH`. The LLM generating the final answer expects the accumulated state (claim, intermediate steps) to generate a response conforming to the `FactCheckResult` schema:
    - `verdict`: (String) One of: "True", "False", "Misleading", "Uncertain", "Partially True/False".
    - `confidence_score`: (Float) A score between 0.0 (low confidence) and 1.0 (high confidence).
    - `explanation`: (String) A detailed explanation justifying the verdict, summarizing the key evidence and reasoning.
    - `evidence_sources`: (List of Objects/Dicts) A list of sources used. Each source should ideally include 'url', 'title', 'snippet', and 'source_tool' (the name of the tool that provided the source, e.g., 'TavilySearchResults', 'Wikidata', 'scrape_webpages_tool').

**Response Format:**
Your response should be a JSON object containing either 'action' and 'action_input' for tool calls, or the final 'answer' when finishing.
If calling a tool: `{{{{"action": "tool_name", "action_input": {{ "arg1": "value1", ... }} }}}}`
If finishing: `{{{{"action": "FINISH", "action_input": {{ "reason": "Concluding based on verified evidence." }} }}}}`

Let's begin the investigation for the claim provided in the input.
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