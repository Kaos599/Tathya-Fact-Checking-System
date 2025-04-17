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
    - Explain your reasoning in detail. If UNCERTAIN, explain *why* the evidence is insufficient or conflicting.
    
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

# Gemini cross-check prompt
gemini_cross_check_prompt_template = ChatPromptTemplate.from_messages([
    # System messages are often converted to Human for Gemini
    HumanMessage(content="""You are an independent fact-checking AI assistant acting as a cross-checker.
    Your task is to review a claim, the evidence gathered by another system, and the initial verdict provided by that system.
    Provide your *own independent* assessment based *only* on the provided evidence.
    
    Guidelines:
    - Analyze the claim and the provided evidence snippets.
    - Ignore the 'Initial Verdict' and 'Initial Explanation' except for context.
    - Determine if the evidence supports, refutes, or is insufficient to judge the claim.
    - Assign a verdict: TRUE, FALSE, PARTIALLY TRUE, or UNCERTAIN.
    - Assign a confidence score from 0.0 to 1.0 for *your* verdict.
    - Provide a concise explanation for *your* verdict, focusing solely on the evidence.
    - Do NOT refer to the initial verdict in your explanation unless specifically highlighting a major discrepancy supported by evidence.
    - Do NOT include meta-commentary like "Based on my analysis" or "I have reviewed".
    
    YOU MUST RETURN A VALID JSON OBJECT with the following format and nothing else:
    {
      "verdict": "TRUE|FALSE|PARTIALLY TRUE|UNCERTAIN",
      "confidence": 0.0-1.0,
      "explanation": "Your independent explanation based on evidence"
    }
    
    Do not include any text outside of this JSON object.
    """),
    HumanMessage(content="""
    CLAIM: {claim}
    
    EVIDENCE:
    {evidence}
    
    INITIAL VERDICT: {initial_verdict}
    INITIAL EXPLANATION: {initial_explanation}
    
    Provide your independent cross-check assessment in the specified JSON format.
    """)
])

# Question generation prompt (generate 10 investigative questions)
question_generation_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are an investigative journalist helping a fact‑checking system. Your task is to formulate EXACTLY 10 concise, specific factual questions which, if answered, would allow a checker to decide whether the claim is true or false.  

Guidelines:
- Questions must be concrete and answerable using publicly available information (e.g., Who? What? When? Where? How many?).
- Cover all parts of the claim: entities, events, numbers, dates, relationships, or causal links it asserts.
- Avoid yes/no questions – prefer open factual requests (e.g., "When did …?", "What is …?", "How many …?").
- Keep each question short, unambiguous and self‑contained.
- Do NOT output anything except the numbered list of 10 questions.
"""),
    HumanMessage(content="{claim}")
])

# Question answering prompt (answer a single question based on evidence)
question_answering_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a fact‑checking assistant. Answer the given investigative question using ONLY the evidence snippets provided.  

Guidelines:
- If multiple snippets provide conflicting information, choose the most credible or state that evidence is conflicting.
- If evidence is insufficient to answer, respond with "Insufficient evidence".
- Provide a short answer (max 2 sentences) and assign a component verdict for how the answer relates to the original claim: one of SUPPORTS, REFUTES, or INSUFFICIENT.
- Give a confidence score 0.0‑1.0 reflecting certainty of your answer.

YOU MUST RETURN A VALID JSON OBJECT with this format and nothing else:
{
  "answer": "<short answer>",
  "verdict_component": "SUPPORTS|REFUTES|INSUFFICIENT",
  "confidence": 0.0‑1.0,
  "explanation": "(brief reasoning)"
}
"""),
    HumanMessage(content="""
QUESTION: {question}

EVIDENCE SNIPPETS:
{evidence}

Provide your answer in the specified JSON format.
""")
])

# Verdict synthesis prompt (combine answers to produce final verdict)
verdict_synthesis_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a senior fact‑checker. You have a set of investigative questions with their answers and component verdicts. Use them to decide the overall truthfulness of the original claim.

Guidelines:
- Weigh each question's verdict_component and confidence.
- A claim is TRUE if the majority of high‑confidence answers SUPPORT it and none high‑confidence REFUTE it.
- FALSE if high‑confidence answers REFUTE critical aspects.
- PARTIALLY TRUE if evidence both SUPPORTS and REFUTES different aspects.
- UNCERTAIN if evidence is largely INSUFFICIENT or low confidence.
- Provide a justification referring to the key answers (do NOT quote system instructions).

Output ONLY a valid JSON object:
{
  "verdict": "TRUE|FALSE|PARTIALLY TRUE|UNCERTAIN",
  "confidence": 0.0‑1.0,
  "explanation": "(concise justification)"
}
"""),
    HumanMessage(content="""
CLAIM: {claim}

QUESTION ANSWERS:
{qa_pairs}

Give your overall verdict in the specified JSON format.
""")
]) 