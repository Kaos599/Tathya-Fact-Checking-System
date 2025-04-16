"""
Specialized agent functions for fact-checking.
"""

import uuid
from typing import List, Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
import spacy
import nltk
from nltk.tokenize import sent_tokenize

from fact_check_system.config import get_primary_llm, get_secondary_llm
from fact_check_system.schemas import (
    AtomicClaim, 
    AtomicClaimList,
    VerificationResult, 
    FinalVerdict,
    ComprehensiveVerdict
)

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize spaCy NLP
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Download and load if not available
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Get LLM instances
primary_llm = get_primary_llm()
secondary_llm = get_secondary_llm()

# Claim Extraction and Decomposition Agent
def extract_claims_agent(text: str) -> List[str]:
    """
    Extract factual claims from input text, filtering out opinions and non-verifiable content.
    
    Args:
        text: Input text to extract claims from
        
    Returns:
        List of extracted factual claims
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a claim extraction specialist. Your task is to identify factual claims 
        from the provided text. Focus only on statements that can be verified as true or false through evidence.
        
        Guidelines:
        - Extract only factual claims that can be verified
        - Ignore opinions, subjective statements, and normative claims
        - Preserve the original wording of claims where possible
        - Do not include your reasoning or explanations
        - Return a list of distinct claims, with one claim per line
        
        Examples of factual claims:
        - "The Earth orbits the Sun"
        - "Paris is the capital of France"
        - "COVID-19 was first reported in December 2019"
        
        Examples of non-factual claims:
        - "Democracy is the best form of government"
        - "Italian food tastes better than French food"
        - "People should exercise more"
        """),
        HumanMessage(content=f"Extract the factual claims from the following text:\n\n{text}")
    ])
    
    output_parser = StrOutputParser()
    chain = prompt | primary_llm | output_parser
    
    result = chain.invoke({})
    
    # Split into individual claims
    claims = [claim.strip() for claim in result.split("\n") if claim.strip()]
    
    return claims

def decompose_claims_agent(claim: str) -> List[AtomicClaim]:
    """
    Break down complex claims into atomic verifiable units.
    
    Args:
        claim: The complex claim to decompose
        
    Returns:
        List of atomic claims
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a claim decomposition expert. Your task is to break down complex claims 
        into atomic verifiable units. An atomic claim is a single assertion about the world that can be independently 
        verified.
        
        Guidelines:
        - Break down the claim into the smallest independently verifiable units
        - Ensure each atomic claim captures a single, clear assertion
        - Identify entities, time references, locations, and numeric values
        - Preserve the original meaning and context
        - Assign an importance score (0.0-1.0) to each atomic claim
        """),
        HumanMessage(content=f"Decompose the following claim into atomic verifiable units:\n\n{claim}")
    ])
    
    # Use PydanticOutputParser to get structured output
    parser = PydanticOutputParser(pydantic_object=AtomicClaimList)
    format_instructions = parser.get_format_instructions()
    
    # Add format instructions to the prompt
    prompt_with_format = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a claim decomposition expert. Your task is to break down complex claims 
        into atomic verifiable units. An atomic claim is a single assertion about the world that can be independently 
        verified.
        
        Guidelines:
        - Break down the claim into the smallest independently verifiable units
        - Ensure each atomic claim captures a single, clear assertion
        - Identify entities, time references, locations, and numeric values
        - Preserve the original meaning and context
        - Assign an importance score (0.0-1.0) to each atomic claim
        
        """ + format_instructions),
        HumanMessage(content=f"Decompose the following claim into atomic verifiable units:\n\n{claim}")
    ])
    
    chain = prompt_with_format | primary_llm | parser
    
    try:
        # Try parsing the output
        atomic_claims_list = chain.invoke({})
        atomic_claims = atomic_claims_list.claims
        
        # Ensure each claim has an ID
        for i, claim in enumerate(atomic_claims):
            if not claim.id:
                claim.id = str(uuid.uuid4())
        
        return atomic_claims
    except Exception as e:
        # Fallback to manual parsing if the structured output fails
        chain = prompt | primary_llm | StrOutputParser()
        result = chain.invoke({})
        
        # Manual parsing logic
        lines = result.strip().split('\n')
        atomic_claims = []
        
        current_claim = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('- '):
                # New claim
                if current_claim and 'statement' in current_claim:
                    atomic_claims.append(AtomicClaim(
                        id=str(uuid.uuid4()),
                        statement=current_claim.get('statement', ''),
                        entities=current_claim.get('entities', []),
                        time_references=current_claim.get('time_references', []),
                        location_references=current_claim.get('location_references', []),
                        numeric_values=current_claim.get('numeric_values', []),
                        importance=current_claim.get('importance', 1.0)
                    ))
                current_claim = {'statement': line[2:].strip()}
            elif ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == 'entities':
                    current_claim['entities'] = [e.strip() for e in value.split(',')]
                elif key == 'time references':
                    current_claim['time_references'] = [t.strip() for t in value.split(',')]
                elif key == 'location references':
                    current_claim['location_references'] = [l.strip() for l in value.split(',')]
                elif key == 'numeric values':
                    current_claim['numeric_values'] = [n.strip() for n in value.split(',')]
                elif key == 'importance':
                    try:
                        current_claim['importance'] = float(value)
                    except:
                        current_claim['importance'] = 1.0
        
        # Don't forget the last claim
        if current_claim and 'statement' in current_claim:
            atomic_claims.append(AtomicClaim(
                id=str(uuid.uuid4()),
                statement=current_claim.get('statement', ''),
                entities=current_claim.get('entities', []),
                time_references=current_claim.get('time_references', []),
                location_references=current_claim.get('location_references', []),
                numeric_values=current_claim.get('numeric_values', []),
                importance=current_claim.get('importance', 1.0)
            ))
        
        return atomic_claims

# Verification Agents
def primary_verification_agent(claim: AtomicClaim, evidence_collection: Dict[str, List[Dict]]) -> VerificationResult:
    """
    Primary agent for verifying claims against evidence.
    
    Args:
        claim: The atomic claim to verify
        evidence_collection: Collection of evidence from different sources
        
    Returns:
        Verification result with reasoning and confidence
    """
    # Flatten evidence for easier processing
    all_evidence = []
    for source_type, evidence_list in evidence_collection.items():
        all_evidence.extend(evidence_list)
    
    # Convert evidence to a readable format
    evidence_text = "\n\n".join([
        f"EVIDENCE {i+1} (Source: {get_evidence_attr(e, 'source', 'unknown')}, URL: {get_evidence_attr(e, 'url', 'N/A')}):\n{get_evidence_attr(e, 'content', 'No content')}"
        for i, e in enumerate(all_evidence)
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a primary fact-checking agent. Your task is to verify 
        a claim based on the evidence provided. You must reason step-by-step and provide a clear 
        verdict with confidence score.
        
        Possible verdicts:
        - TRUE: The claim is fully supported by the evidence
        - FALSE: The evidence contradicts the claim
        - PARTIALLY_TRUE: The evidence supports some aspects of the claim but not others
        - UNVERIFIABLE: There is insufficient evidence to verify the claim
        - NEEDS_CONTEXT: The claim is misleading without additional context
        
        For each piece of evidence, assess its relevance and credibility in relation to the claim.
        """),
        HumanMessage(content=f"""
        CLAIM: {claim.statement}
        
        ENTITIES: {', '.join(claim.entities) if claim.entities else 'None specified'}
        TIME REFERENCES: {', '.join(claim.time_references) if claim.time_references else 'None specified'}
        LOCATION REFERENCES: {', '.join(claim.location_references) if claim.location_references else 'None specified'}
        NUMERIC VALUES: {', '.join(claim.numeric_values) if claim.numeric_values else 'None specified'}
        
        EVIDENCE:
        {evidence_text}
        
        Verify this claim based on the evidence provided. Reason step-by-step, then provide your verdict 
        and confidence score.
        """)
    ])
    
    # Define parser
    parser = PydanticOutputParser(pydantic_object=VerificationResult)
    format_instructions = parser.get_format_instructions()
    
    # Add format instructions
    prompt_with_format = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a primary fact-checking agent. Your task is to verify 
        a claim based on the evidence provided. You must reason step-by-step and provide a clear 
        verdict with confidence score.
        
        Possible verdicts:
        - TRUE: The claim is fully supported by the evidence
        - FALSE: The evidence contradicts the claim
        - PARTIALLY_TRUE: The evidence supports some aspects of the claim but not others
        - UNVERIFIABLE: There is insufficient evidence to verify the claim
        - NEEDS_CONTEXT: The claim is misleading without additional context
        
        For each piece of evidence, assess its relevance and credibility in relation to the claim.
        
        """ + format_instructions),
        HumanMessage(content=f"""
        CLAIM: {claim.statement}
        
        ENTITIES: {', '.join(claim.entities) if claim.entities else 'None specified'}
        TIME REFERENCES: {', '.join(claim.time_references) if claim.time_references else 'None specified'}
        LOCATION REFERENCES: {', '.join(claim.location_references) if claim.location_references else 'None specified'}
        NUMERIC VALUES: {', '.join(claim.numeric_values) if claim.numeric_values else 'None specified'}
        
        EVIDENCE:
        {evidence_text}
        
        Verify this claim based on the evidence provided. Reason step-by-step, then provide your verdict 
        and confidence score.
        """)
    ])
    
    chain = prompt_with_format | primary_llm | parser
    
    try:
        result = chain.invoke({})
        # Make sure agent_id is set
        if not result.agent_id:
            result.agent_id = "primary"
        return result
    except Exception as e:
        # Fallback to free-form response if structured output fails
        print(f"Error parsing structured output: {e}")
        chain = prompt | primary_llm | StrOutputParser()
        result_text = chain.invoke({})
        
        # Extract verdict and confidence from text
        verdict = "UNVERIFIABLE"  # Default
        confidence = 0.5  # Default
        reasoning = result_text
        
        # Simple extraction logic
        if "TRUE" in result_text.upper():
            verdict = "TRUE"
        elif "FALSE" in result_text.upper():
            verdict = "FALSE"
        elif "PARTIALLY TRUE" in result_text.upper() or "PARTIALLY_TRUE" in result_text.upper():
            verdict = "PARTIALLY_TRUE"
        elif "NEEDS CONTEXT" in result_text.upper() or "NEEDS_CONTEXT" in result_text.upper():
            verdict = "NEEDS_CONTEXT"
        
        # Try to extract confidence score
        import re
        confidence_pattern = r"confidence(?:\s+score)?(?:\s*:)?\s*(?:is)?\s*(\d+(?:\.\d+)?)"
        confidence_match = re.search(confidence_pattern, result_text, re.IGNORECASE)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                # Normalize to 0-1 if necessary
                if confidence > 1:
                    confidence /= 100
            except:
                pass
        
        return VerificationResult(
            agent_id="primary",
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            evidence_assessment={},  # No detailed assessment in fallback
            reasoning_difficulty=0.5,
            claim_ambiguity=0.5
        )

# Helper function to get attributes from either dict or object
def get_evidence_attr(evidence, attr, default):
    """Get an attribute from evidence, whether it's a dict or object."""
    if isinstance(evidence, dict):
        return evidence.get(attr, default)
    else:
        return getattr(evidence, attr, default)

def cross_verification_agent(claim: AtomicClaim, evidence_collection: Dict[str, List[Dict]]) -> VerificationResult:
    """
    Cross-verification agent using a different LLM to independently verify the claim.
    
    Args:
        claim: The atomic claim to verify
        evidence_collection: Collection of evidence from different sources
        
    Returns:
        Verification result with reasoning and confidence
    """
    # Similar to primary_verification_agent but using secondary_llm
    # Flatten evidence for easier processing
    all_evidence = []
    for source_type, evidence_list in evidence_collection.items():
        all_evidence.extend(evidence_list)
    
    # Convert evidence to a readable format
    evidence_text = "\n\n".join([
        f"EVIDENCE {i+1} (Source: {get_evidence_attr(e, 'source', 'unknown')}, URL: {get_evidence_attr(e, 'url', 'N/A')}):\n{get_evidence_attr(e, 'content', 'No content')}"
        for i, e in enumerate(all_evidence)
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a cross-verification fact-checking agent. Your task is to independently verify 
        a claim based on the evidence provided, focusing on identifying potential biases or limitations in the 
        evidence. You must reason step-by-step and provide a clear verdict with confidence score.
        
        Possible verdicts:
        - TRUE: The claim is fully supported by the evidence
        - FALSE: The evidence contradicts the claim
        - PARTIALLY_TRUE: The evidence supports some aspects of the claim but not others
        - UNVERIFIABLE: There is insufficient evidence to verify the claim
        - NEEDS_CONTEXT: The claim is misleading without additional context
        
        For each piece of evidence, assess its relevance and credibility in relation to the claim.
        """),
        HumanMessage(content=f"""
        CLAIM: {claim.statement}
        
        ENTITIES: {', '.join(claim.entities) if claim.entities else 'None specified'}
        TIME REFERENCES: {', '.join(claim.time_references) if claim.time_references else 'None specified'}
        LOCATION REFERENCES: {', '.join(claim.location_references) if claim.location_references else 'None specified'}
        NUMERIC VALUES: {', '.join(claim.numeric_values) if claim.numeric_values else 'None specified'}
        
        EVIDENCE:
        {evidence_text}
        
        Verify this claim based on the evidence provided. Reason step-by-step, then provide your verdict 
        and confidence score.
        """)
    ])
    
    # Define parser for structured output
    parser = PydanticOutputParser(pydantic_object=VerificationResult)
    format_instructions = parser.get_format_instructions()
    
    # Add format instructions to the prompt
    prompt_with_format = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a cross-verification fact-checking agent. Your task is to independently verify 
        a claim based on the evidence provided, focusing on identifying potential biases or limitations in the 
        evidence. You must reason step-by-step and provide a clear verdict with confidence score.
        
        Possible verdicts:
        - TRUE: The claim is fully supported by the evidence
        - FALSE: The evidence contradicts the claim
        - PARTIALLY_TRUE: The evidence supports some aspects of the claim but not others
        - UNVERIFIABLE: There is insufficient evidence to verify the claim
        - NEEDS_CONTEXT: The claim is misleading without additional context
        
        For each piece of evidence, assess its relevance and credibility in relation to the claim.
        
        """ + format_instructions),
        HumanMessage(content=f"""
        CLAIM: {claim.statement}
        
        ENTITIES: {', '.join(claim.entities) if claim.entities else 'None specified'}
        TIME REFERENCES: {', '.join(claim.time_references) if claim.time_references else 'None specified'}
        LOCATION REFERENCES: {', '.join(claim.location_references) if claim.location_references else 'None specified'}
        NUMERIC VALUES: {', '.join(claim.numeric_values) if claim.numeric_values else 'None specified'}
        
        EVIDENCE:
        {evidence_text}
        
        Verify this claim based on the evidence provided. Reason step-by-step, then provide your verdict 
        and confidence score.
        """)
    ])
    
    # Try structured output first, with fallback
    try:
        # Use the secondary LLM here
        chain = prompt_with_format | secondary_llm | parser
        result = chain.invoke({})
        
        # Make sure agent_id is set
        if not result.agent_id:
            result.agent_id = "cross"
        
        return result
    except Exception as e:
        print(f"Error with secondary LLM structured output: {e}")
        
        # Fallback to primary LLM if secondary fails
        try:
            chain = prompt_with_format | primary_llm | parser
            result = chain.invoke({})
            
            # Make sure agent_id is set
            if not result.agent_id:
                result.agent_id = "cross"
            
            return result
        except Exception as e2:
            print(f"Error with primary LLM structured output too: {e2}")
            
            # Final fallback to free-form output
            chain = prompt | primary_llm | StrOutputParser()
            result_text = chain.invoke({})
            
            # Extract verdict and confidence from text
            verdict = "UNVERIFIABLE"  # Default
            confidence = 0.5  # Default
            reasoning = result_text
            
            # Simple extraction logic
            if "TRUE" in result_text.upper():
                verdict = "TRUE"
            elif "FALSE" in result_text.upper():
                verdict = "FALSE"
            elif "PARTIALLY TRUE" in result_text.upper() or "PARTIALLY_TRUE" in result_text.upper():
                verdict = "PARTIALLY_TRUE"
            elif "NEEDS CONTEXT" in result_text.upper() or "NEEDS_CONTEXT" in result_text.upper():
                verdict = "NEEDS_CONTEXT"
            
            # Try to extract confidence score
            import re
            confidence_pattern = r"confidence(?:\s+score)?(?:\s*:)?\s*(?:is)?\s*(\d+(?:\.\d+)?)"
            confidence_match = re.search(confidence_pattern, result_text, re.IGNORECASE)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    # Normalize to 0-1 if necessary
                    if confidence > 1:
                        confidence /= 100
                except:
                    pass
            
            return VerificationResult(
                agent_id="cross",
                verdict=verdict,
                confidence=confidence,
                reasoning=reasoning,
                evidence_assessment={},  # No detailed assessment in fallback
                reasoning_difficulty=0.5,
                claim_ambiguity=0.5
            )

# Adjudication and Synthesis
def adjudication_agent(claim: AtomicClaim, verification_results: Dict[str, VerificationResult]) -> FinalVerdict:
    """
    Adjudicate between different verification results to produce a final verdict.
    
    Args:
        claim: The atomic claim being verified
        verification_results: Results from different verification agents
        
    Returns:
        Final verdict with reasoning and confidence score
    """
    # Format verification results for prompt
    results_text = ""
    for agent_id, result in verification_results.items():
        results_text += f"\n\nAGENT: {agent_id}\n"
        results_text += f"VERDICT: {result.verdict}\n"
        results_text += f"CONFIDENCE: {result.confidence}\n"
        results_text += f"REASONING: {result.reasoning}\n"
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a fact-checking adjudicator. Your task is to synthesize results 
        from multiple verification agents into a final verdict. You should consider the reasoning, 
        confidence, and potential biases of each agent.
        
        If there are contradictions between agents, you should carefully reconcile them by:
        1. Identifying the source of disagreement
        2. Assessing the strength of reasoning from each agent
        3. Making a determination based on the most compelling evidence and reasoning
        
        Your verdict should be one of:
        - TRUE: The claim is fully supported by the evidence
        - FALSE: The evidence contradicts the claim
        - PARTIALLY_TRUE: The evidence supports some aspects of the claim but not others
        - UNVERIFIABLE: There is insufficient evidence to verify the claim
        - NEEDS_CONTEXT: The claim is misleading without additional context
        """),
        HumanMessage(content=f"""
        CLAIM: {claim.statement}
        
        VERIFICATION RESULTS:{results_text}
        
        Synthesize these verification results into a final verdict with reasoning and confidence score.
        """)
    ])
    
    # Define parser
    parser = PydanticOutputParser(pydantic_object=FinalVerdict)
    format_instructions = parser.get_format_instructions()
    
    # Add format instructions
    prompt_with_format = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a fact-checking adjudicator. Your task is to synthesize results 
        from multiple verification agents into a final verdict. You should consider the reasoning, 
        confidence, and potential biases of each agent.
        
        If there are contradictions between agents, you should carefully reconcile them by:
        1. Identifying the source of disagreement
        2. Assessing the strength of reasoning from each agent
        3. Making a determination based on the most compelling evidence and reasoning
        
        Your verdict should be one of:
        - TRUE: The claim is fully supported by the evidence
        - FALSE: The evidence contradicts the claim
        - PARTIALLY_TRUE: The evidence supports some aspects of the claim but not others
        - UNVERIFIABLE: There is insufficient evidence to verify the claim
        - NEEDS_CONTEXT: The claim is misleading without additional context
        
        """ + format_instructions),
        HumanMessage(content=f"""
        CLAIM: {claim.statement}
        
        VERIFICATION RESULTS:{results_text}
        
        Synthesize these verification results into a final verdict with reasoning and confidence score.
        """)
    ])
    
    chain = prompt_with_format | primary_llm | parser
    
    try:
        result = chain.invoke({})
        # Make sure claim_id and claim_text are set
        if not result.claim_id:
            result.claim_id = claim.id
        result.claim_text = claim.statement
        return result
    except Exception as e:
        # Fallback to free-form response
        chain = prompt | primary_llm | StrOutputParser()
        result_text = chain.invoke({})
        
        # Extract verdict and confidence from text
        verdict = "UNVERIFIABLE"  # Default
        confidence = 0.5  # Default
        reasoning = result_text
        evidence_summary = "No structured evidence summary available."
        
        # Simple extraction logic
        if "TRUE" in result_text.upper() and not "PARTIALLY" in result_text.upper():
            verdict = "TRUE"
        elif "FALSE" in result_text.upper() and not "PARTIALLY" in result_text.upper():
            verdict = "FALSE"
        elif "PARTIALLY TRUE" in result_text.upper() or "PARTIALLY_TRUE" in result_text.upper():
            verdict = "PARTIALLY_TRUE"
        elif "NEEDS CONTEXT" in result_text.upper() or "NEEDS_CONTEXT" in result_text.upper():
            verdict = "NEEDS_CONTEXT"
        
        # Try to extract confidence score
        import re
        confidence_pattern = r"confidence(?:\s+score)?(?:\s*:)?\s*(?:is)?\s*(\d+(?:\.\d+)?)"
        confidence_match = re.search(confidence_pattern, result_text, re.IGNORECASE)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                # Normalize to 0-1 if necessary
                if confidence > 1:
                    confidence /= 100
            except:
                pass
        
        # Try to extract evidence summary
        summary_pattern = r"evidence\s+summary\s*:\s*(.*?)(?:\n\n|\Z)"
        summary_match = re.search(summary_pattern, result_text, re.IGNORECASE | re.DOTALL)
        if summary_match:
            evidence_summary = summary_match.group(1).strip()
        
        return FinalVerdict(
            claim_id=claim.id,
            claim_text=claim.statement,
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            evidence_summary=evidence_summary
        )

def synthesis_agent(original_claim: str, atomic_verdicts: Dict[str, FinalVerdict]) -> ComprehensiveVerdict:
    """
    Synthesize verdicts from atomic claims into a comprehensive verdict for the original claim.
    
    Args:
        original_claim: The original complex claim
        atomic_verdicts: Dictionary of atomic claim verdicts
        
    Returns:
        Comprehensive verdict for the original claim
    """
    # Format atomic verdicts for prompt
    verdicts_text = ""
    for claim_id, verdict in atomic_verdicts.items():
        verdicts_text += f"\n\nCLAIM: {verdict.claim_text}\n"
        verdicts_text += f"VERDICT: {verdict.verdict}\n"
        verdicts_text += f"CONFIDENCE: {verdict.confidence}\n"
        verdicts_text += f"REASONING: {verdict.reasoning}\n"
        verdicts_text += f"EVIDENCE SUMMARY: {verdict.evidence_summary}\n"
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a fact-checking synthesis expert. Your task is to synthesize 
        verdicts from multiple atomic claims into a comprehensive verdict for the original complex claim.
        
        You should consider:
        1. The verdicts and confidence scores of individual atomic claims
        2. The relative importance of each atomic claim to the overall claim
        3. The logical relationships between the atomic claims
        
        Your verdict should be one of:
        - TRUE: The claim is fully supported by the evidence
        - FALSE: The evidence contradicts the claim
        - PARTIALLY_TRUE: The evidence supports some aspects of the claim but not others
        - UNVERIFIABLE: There is insufficient evidence to verify the claim
        - NEEDS_CONTEXT: The claim is misleading without additional context
        
        You should also provide:
        - An overall confidence score
        - A comprehensive reasoning that explains how the atomic verdicts combine
        - A summary that explains the verdict in plain language
        """),
        HumanMessage(content=f"""
        ORIGINAL CLAIM: {original_claim}
        
        ATOMIC CLAIM VERDICTS:{verdicts_text}
        
        Synthesize these atomic verdicts into a comprehensive verdict for the original claim.
        """)
    ])
    
    # Define parser
    parser = PydanticOutputParser(pydantic_object=ComprehensiveVerdict)
    format_instructions = parser.get_format_instructions()
    
    # Add format instructions
    prompt_with_format = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a fact-checking synthesis expert. Your task is to synthesize 
        verdicts from multiple atomic claims into a comprehensive verdict for the original complex claim.
        
        You should consider:
        1. The verdicts and confidence scores of individual atomic claims
        2. The relative importance of each atomic claim to the overall claim
        3. The logical relationships between the atomic claims
        
        Your verdict should be one of:
        - TRUE: The claim is fully supported by the evidence
        - FALSE: The evidence contradicts the claim
        - PARTIALLY_TRUE: The evidence supports some aspects of the claim but not others
        - UNVERIFIABLE: There is insufficient evidence to verify the claim
        - NEEDS_CONTEXT: The claim is misleading without additional context
        
        You should also provide:
        - An overall confidence score
        - A comprehensive reasoning that explains how the atomic verdicts combine
        - A summary that explains the verdict in plain language
        
        """ + format_instructions),
        HumanMessage(content=f"""
        ORIGINAL CLAIM: {original_claim}
        
        ATOMIC CLAIM VERDICTS:{verdicts_text}
        
        Synthesize these atomic verdicts into a comprehensive verdict for the original claim.
        """)
    ])
    
    chain = prompt_with_format | primary_llm | parser
    
    try:
        result = chain.invoke({})
        # Make sure original_claim and atomic_verdicts are set
        result.original_claim = original_claim
        result.atomic_verdicts = atomic_verdicts
        return result
    except Exception as e:
        # Fallback to free-form response
        chain = prompt | primary_llm | StrOutputParser()
        result_text = chain.invoke({})
        
        # Extract verdict and confidence from text
        verdict = "UNVERIFIABLE"  # Default
        confidence = 0.5  # Default
        reasoning = result_text
        summary = "No structured summary available."
        
        # Simple extraction logic
        if "TRUE" in result_text.upper() and not "PARTIALLY" in result_text.upper():
            verdict = "TRUE"
        elif "FALSE" in result_text.upper() and not "PARTIALLY" in result_text.upper():
            verdict = "FALSE"
        elif "PARTIALLY TRUE" in result_text.upper() or "PARTIALLY_TRUE" in result_text.upper():
            verdict = "PARTIALLY_TRUE"
        elif "NEEDS CONTEXT" in result_text.upper() or "NEEDS_CONTEXT" in result_text.upper():
            verdict = "NEEDS_CONTEXT"
        
        # Try to extract confidence score
        import re
        confidence_pattern = r"confidence(?:\s+score)?(?:\s*:)?\s*(?:is)?\s*(\d+(?:\.\d+)?)"
        confidence_match = re.search(confidence_pattern, result_text, re.IGNORECASE)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                # Normalize to 0-1 if necessary
                if confidence > 1:
                    confidence /= 100
            except:
                pass
        
        # Try to extract summary
        summary_pattern = r"summary\s*:\s*(.*?)(?:\n\n|\Z)"
        summary_match = re.search(summary_pattern, result_text, re.IGNORECASE | re.DOTALL)
        if summary_match:
            summary = summary_match.group(1).strip()
        
        return ComprehensiveVerdict(
            original_claim=original_claim,
            atomic_verdicts=atomic_verdicts,
            overall_verdict=verdict,
            overall_confidence=confidence,
            reasoning=reasoning,
            summary=summary
        ) 