"""
Example script to demonstrate the usage of the fact-checking system.
"""

import json
import time
from dotenv import load_dotenv

from fact_check_system.workflow import process_claim

# Load environment variables
load_dotenv()

def format_verdict(verdict):
    """Format the verdict for display."""
    if not verdict:
        return "No verdict available."
    
    # Convert from dict to readable format
    result = []
    result.append(f"ORIGINAL CLAIM: {verdict['original_claim']}")
    result.append(f"OVERALL VERDICT: {verdict['overall_verdict']} (Confidence: {verdict['overall_confidence']:.2f})")
    result.append(f"SUMMARY: {verdict['summary']}")
    
    # Display intermediate steps
    if 'process_steps' in verdict and verdict['process_steps']:
        result.append("\nFACT-CHECKING PROCESS:")
        
        for step in verdict['process_steps']:
            step_name = step['step'].replace('_', ' ').upper()
            result.append(f"\n  STEP: {step_name} ({step['timestamp']})")
            
            if step['step'] == 'decompose_claims' and 'extracted_claims' in step:
                result.append("  EXTRACTED CLAIMS:")
                for i, claim in enumerate(step['extracted_claims']):
                    result.append(f"    {i+1}. {claim}")
            
            if step['step'] == 'verify_claims' and 'verification_details' in step:
                result.append("  VERIFICATION PROCESS:")
                for claim_id, details in step['verification_details'].items():
                    result.append(f"    CLAIM: {details['claim']}")
                    result.append(f"    PRIMARY AGENT: {details['primary_result']['verdict']} (Confidence: {details['primary_result']['confidence']:.2f})")
                    result.append(f"    CROSS AGENT: {details['cross_result']['verdict']} (Confidence: {details['cross_result']['confidence']:.2f})")
                    result.append(f"    COMBINED CONFIDENCE: {details['combined_confidence']:.2f}")
                    result.append("")
            
            if step['step'] == 'adjudicate_results' and 'overall_verdict' in step:
                result.append("  FINAL ASSESSMENT:")
                result.append(f"    VERDICT: {step['overall_verdict']['verdict']}")
                result.append(f"    CONFIDENCE: {step['overall_verdict']['confidence']:.2f}")
                result.append(f"    REASONING: {step['overall_verdict']['reasoning']}")
    elif 'intermediate_steps' in verdict and verdict['intermediate_steps']:
        # Backward compatibility with older format
        result.append("\nFACT-CHECKING PROCESS:")
        
        for step in verdict['intermediate_steps']:
            step_name = step['step'].replace('_', ' ').upper()
            result.append(f"\n  STEP: {step_name} ({step['timestamp']})")
            
            if step['step'] == 'decompose_claims' and 'extracted_claims' in step:
                result.append("  EXTRACTED CLAIMS:")
                for i, claim in enumerate(step['extracted_claims']):
                    result.append(f"    {i+1}. {claim}")
            
            if step['step'] == 'verify_claims' and 'verification_details' in step:
                result.append("  VERIFICATION PROCESS:")
                for claim_id, details in step['verification_details'].items():
                    result.append(f"    CLAIM: {details['claim']}")
                    result.append(f"    PRIMARY AGENT: {details['primary_result']['verdict']} (Confidence: {details['primary_result']['confidence']:.2f})")
                    result.append(f"    CROSS AGENT: {details['cross_result']['verdict']} (Confidence: {details['cross_result']['confidence']:.2f})")
                    result.append(f"    COMBINED CONFIDENCE: {details['combined_confidence']:.2f}")
                    result.append("")
            
            if step['step'] == 'adjudicate_results' and 'overall_verdict' in step:
                result.append("  FINAL ASSESSMENT:")
                result.append(f"    VERDICT: {step['overall_verdict']['verdict']}")
                result.append(f"    CONFIDENCE: {step['overall_verdict']['confidence']:.2f}")
                result.append(f"    REASONING: {step['overall_verdict']['reasoning']}")
    
    # Display sources used
    if 'sources_used' in verdict and verdict['sources_used']:
        result.append("\nSOURCES USED:")
        
        for claim_id, sources in verdict['sources_used'].items():
            # Find the corresponding claim text
            claim_text = None
            
            # First check in atomic_verdicts
            if 'atomic_verdicts' in verdict:
                for atomic_id, atomic_verdict in verdict['atomic_verdicts'].items():
                    if atomic_id == claim_id:
                        claim_text = atomic_verdict.get('claim_text', None)
                        break
            
            # If not found, check in process steps
            if not claim_text:
                for step in verdict.get('process_steps', verdict.get('intermediate_steps', [])):
                    if step['step'] == 'decompose_claims':
                        for claim in step.get('atomic_claims', []):
                            if claim.get('id') == claim_id:
                                claim_text = claim.get('statement')
                                break
            
            if claim_text:
                result.append(f"\n  CLAIM: {claim_text}")
                
                # Group sources by type
                source_types = {}
                for source in sources:
                    source_type = source.get('source_type', 'unknown')
                    if source_type not in source_types:
                        source_types[source_type] = []
                    source_types[source_type].append(source)
                
                # Display sources by type
                for source_type, type_sources in source_types.items():
                    result.append(f"    {source_type.upper()} SOURCES:")
                    for i, source in enumerate(type_sources[:5]):  # Limit to 5 sources per type
                        if 'url' in source and source['url']:
                            result.append(f"      {i+1}. {source.get('source', 'Unknown source')}: {source['url']}")
                        else:
                            result.append(f"      {i+1}. {source.get('source', 'Unknown source')}")
                    
                    if len(type_sources) > 5:
                        result.append(f"      ... and {len(type_sources) - 5} more {source_type} sources")
    
    result.append("\nATOMIC CLAIMS:")
    
    for claim_id, atomic_verdict in verdict['atomic_verdicts'].items():
        result.append(f"\n  CLAIM: {atomic_verdict['claim_text']}")
        result.append(f"  VERDICT: {atomic_verdict['verdict']} (Confidence: {atomic_verdict['confidence']:.2f})")
        result.append(f"  EVIDENCE: {atomic_verdict['evidence_summary']}")
        
        # Display sources for this atomic claim if available in the new schema
        if 'sources_used' in atomic_verdict and atomic_verdict['sources_used']:
            result.append(f"  SOURCES:")
            source_types = {}
            for source in atomic_verdict['sources_used']:
                source_type = source.get('source_type', 'unknown')
                if source_type not in source_types:
                    source_types[source_type] = []
                source_types[source_type].append(source)
            
            for source_type, type_sources in source_types.items():
                result.append(f"    {source_type.upper()}:")
                for i, source in enumerate(type_sources[:3]):  # Limit to 3 sources per type
                    if 'url' in source and source['url']:
                        result.append(f"      {i+1}. {source.get('source', 'Unknown')}: {source['url']}")
                    else:
                        result.append(f"      {i+1}. {source.get('source', 'Unknown')}")
                
                if len(type_sources) > 3:
                    result.append(f"      ... and {len(type_sources) - 3} more {source_type} sources")
    
    return "\n".join(result)

def run_example():
    """Run a fact-checking example."""
    # Simple claim example
    simple_claim = "The Earth orbits the Sun."
    
    print("=" * 80)
    print(f"CHECKING SIMPLE CLAIM: {simple_claim}")
    print("=" * 80)
    
    start_time = time.time()
    verdict = process_claim(simple_claim)
    processing_time = time.time() - start_time
    
    print(format_verdict(verdict))
    print(f"\nProcessing time: {processing_time:.2f} seconds")
    
    # Complex claim example
    complex_claim = """
    COVID-19 vaccines have been linked to thousands of deaths, and the mRNA technology 
    used in the Pfizer and Moderna vaccines can alter human DNA.
    """
    
    print("\n\n" + "=" * 80)
    print(f"CHECKING COMPLEX CLAIM: {complex_claim.strip()}")
    print("=" * 80)
    
    start_time = time.time()
    verdict = process_claim(complex_claim)
    processing_time = time.time() - start_time
    
    print(format_verdict(verdict))
    print(f"\nProcessing time: {processing_time:.2f} seconds")

if __name__ == "__main__":
    run_example() 