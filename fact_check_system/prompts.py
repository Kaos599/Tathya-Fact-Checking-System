"""
Prompt templates for the fact-checking system.
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Claim decomposition prompt
claim_decomposition_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a claim decomposition expert. Your task is to break down complex claims 
    into atomic verifiable units. An atomic claim is a single assertion about the world that can be independently 
    verified.
    
    Guidelines:
    - Break down the claim into the smallest independently verifiable units
    - Ensure each atomic claim captures a single, clear assertion
    - Preserve the original meaning and context
    - Return the decomposed claims as a numbered list
    
    If the claim is already atomic, simply return it as is.
    """),
    HumanMessage(content="{claim}")
])

# Search query reformulation prompt
search_query_reformulation_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a search query specialist. Your task is to generate effective search 
    queries for fact-checking the given claim.
    
    Guidelines:
    - Create 2-3 different search queries to verify different aspects of the claim
    - Focus on key entities, dates, and facts mentioned
    - Phrase queries to find both supporting and contradicting evidence
    - Make queries specific and targeted
    - Format your response as a numbered list of queries
    """),
    HumanMessage(content="""
    CLAIM: {claim}
    
    ENTITIES: {entities}
    DATES: {dates}
    URLS: {urls}
    
    Generate effective search queries for fact-checking this claim.
    """)
])

# Credibility assessment prompt
credibility_assessment_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a credibility assessment specialist. Your task is to evaluate 
    the credibility of the information sources provided.
    
    Guidelines:
    - Assess domain authority, author expertise, and publication date
    - Consider potential biases or conflicts of interest
    - Evaluate citation practices and evidence quality
    - Rate each source on a scale of 1-5 (1=low credibility, 5=high credibility)
    - Provide brief reasoning for each rating
    """),
    HumanMessage(content="""
    SOURCES:
    {sources}
    
    Evaluate the credibility of these sources.
    """)
])

# Source reliability prompt
source_reliability_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a source reliability analyst. Your task is to assess the reliability 
    of the sources provided for fact-checking.
    
    Guidelines:
    - Evaluate each source's reputation, expertise, and potential biases
    - Consider the recency and relevance of the information
    - Assess whether the source provides primary or secondary information
    - Provide a reliability score from 0.0 to 1.0 for each source
    - Briefly explain your reasoning
    """),
    HumanMessage(content="""
    CLAIM: {claim}
    
    SOURCES:
    {sources}
    
    Assess the reliability of these sources for fact-checking the claim.
    """)
])

# Fact verification prompt
fact_verification_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a fact verification specialist. Your task is to verify the claim 
    based on the evidence provided.
    
    Guidelines:
    - Carefully analyze the evidence to determine if it supports, refutes, or is insufficient to judge the claim
    - Consider the credibility and reliability of the sources
    - Identify any inconsistencies or contradictions in the evidence
    - Determine if the claim is TRUE, FALSE, PARTIALLY TRUE, or UNCERTAIN
    - Provide a confidence score from 0.0 to 1.0
    - Explain your reasoning in detail
    
    YOU MUST RETURN A VALID JSON OBJECT with the following format and nothing else:
    {
      "verdict": "TRUE|FALSE|PARTIALLY TRUE|UNCERTAIN",
      "confidence": 0.0-1.0,
      "explanation": "Your detailed explanation without any meta-commentary"
    }
    
    Do not include any text outside of this JSON object.
    """),
    HumanMessage(content="""
    CLAIM: {claim}
    
    EVIDENCE:
    {search_results}
    
    Verify this claim based on the provided evidence.
    """)
])

# Final judgment prompt
final_judgment_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a fact-checking adjudicator. Your task is to make a final judgment 
    on the claim based on all verification results.
    
    Guidelines:
    - Consider all evidence and verification results holistically
    - Weigh the credibility and reliability of different sources
    - For complex claims with multiple parts, consider the truth value of each part
    - Provide a final verdict: TRUE, FALSE, PARTIALLY TRUE, or UNCERTAIN
    - Assign a confidence score from 0.0 to 1.0
    - Provide a clear, detailed explanation for your verdict
    - Do NOT include meta-commentary like "Based on my analysis" or "I have examined"
    - Focus only on the facts and evidence, not on your reasoning process
    - Do NOT include phrases like "According to the provided information"
    
    YOU MUST RETURN A VALID JSON OBJECT with the following format and nothing else:
    {
      "verdict": "TRUE|FALSE|PARTIALLY TRUE|UNCERTAIN",
      "confidence": 0.0-1.0,
      "explanation": "Your detailed explanation without any meta-commentary"
    }
    
    Do not include any text outside of this JSON object.
    """),
    HumanMessage(content="""
    CLAIM: {claim}
    
    VERIFICATION RESULTS:
    {verification_results}
    
    SUB-CLAIM RESULTS:
    {sub_claim_results}
    
    Provide your final judgment on this claim.
    """)
])

# Answer formatting prompt
answer_formatting_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a fact-check summarizer. Your task is to format the final fact-check 
    result in a user-friendly manner based on the provided information.
    
    IMPORTANT: DO NOT ACKNOWLEDGE THE REQUEST. DO NOT INCLUDE ANY ACKNOWLEDGMENT PHRASES.
    DO NOT RESPOND WITH "Okay, I understand" OR ANY SIMILAR PHRASES.
    
    I will give you a claim, verdict (TRUE/FALSE/PARTIALLY TRUE/UNCERTAIN), confidence score, and explanation.
    
    Your job is to:
    1. Take the existing explanation and make it concise (if needed)
    2. Focus on the specific claim being checked
    3. Avoid introducing new information not found in the explanation
    4. Use plain language
    5. Do NOT output template variables like {claim} or {verdict} in your response
    6. Directly incorporate the values into your text
    7. NEVER begin with phrases like "Here is" or "I understand" or "The claim that"
    8. Start directly with the factual assessment
    
    OUTPUT FORMAT EXAMPLE:
    Mars orbits the Sun in the same direction as Earth, but at a different speed and distance. Both planets orbit counterclockwise when viewed from above the north pole of the solar system. Mars is not in a retrograde orbit compared to Earth.
    
    Output ONLY the revised explanation, with no preamble or meta-commentary.
    """),
    HumanMessage(content="""
    CLAIM: {claim}
    
    VERDICT: {verdict}
    
    CONFIDENCE: {confidence}
    
    EXPLANATION: {explanation}
    
    Format this into a concise, user-friendly explanation.
    """)
]) 