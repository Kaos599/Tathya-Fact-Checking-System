"""
Main orchestrator for the fact-checking process using a LangGraph agent.
"""

import logging
import uuid
import json # Needed to parse final JSON output
from typing import List, Dict, Any, Optional
import os # Import os to read env var for recursion limit

# Keep Pydantic for the final result structure if desired
from pydantic import BaseModel, Field, ValidationError

# Removed old imports like specific prompts, parsers, runnables

# Import the agent builder and state definition
# Removed unused imports: build_agent_executor, AgentState
# Import the final result models (keep or adapt)
from .models import FactCheckResult, EvidenceSource # Keep EvidenceSource for structure
# Import message types for checking final state
from langchain_core.messages import AIMessage, HumanMessage

# Import the new agent graph and config
from .agent import create_fact_checking_agent_graph
# Import the FactCheckResult schema from schemas if it's defined there,
# otherwise rely on the models import. Let's assume models for now.
# from .schemas import FactCheckResult # Or from .models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Removed DecompositionOutput model
# Removed format_evidence_for_prompt function

# --- Compile the agent graph once on module load --- #
try:
    logger.info("Initializing and compiling the fact-checking agent graph...")
    fact_checking_agent = create_fact_checking_agent_graph()
    logger.info("Fact-checking agent graph compiled successfully.")
except Exception as e:
    logger.exception("Failed to compile the fact-checking agent graph on startup!")
    fact_checking_agent = None # Set to None to indicate failure
# ------------------------------------------------- #

def run_fact_check(claim: str, config_override: Dict[str, Any] = None) -> FactCheckResult:
    """
    Runs the fact-checking agent for a given claim.

    Args:
        claim: The claim string to verify.
        config_override: Optional dictionary to override parts of the graph config for this run.

    Returns:
        A FactCheckResult object containing the verdict, confidence, explanation, and sources.
    """
    claim_id = str(uuid.uuid4())
    if fact_checking_agent is None:
        logger.error("Fact-checking agent graph failed to compile. Cannot run fact check.")
        return FactCheckResult(claim=claim, claim_id=claim_id, verdict="Error", confidence_score=0.0, explanation="Agent graph compilation failed.", evidence_sources=[])

    logger.info(f"Starting fact check run for claim: '{claim}'")

    # Prepare initial state for the graph
    initial_state = {
        "claim": claim,
        "claim_id": claim_id,
        "messages": [HumanMessage(content=f"Fact check this claim: {claim}")]
        # intermediate_steps and final_result start empty
    }

    # Configuration for the graph invocation (e.g., recursion limit)
    # Get recursion limit from env var or use default
    try:
        recursion_limit = int(os.getenv("LANGGRAPH_RECURSION_LIMIT", "15"))
    except ValueError:
        recursion_limit = 15
        logger.warning(f"Invalid LANGGRAPH_RECURSION_LIMIT env var. Using default: {recursion_limit}")

    # Apply overrides if provided
    runtime_config = {"recursion_limit": recursion_limit}
    if config_override:
        if "recursion_limit" in config_override:
            try:
                 runtime_config["recursion_limit"] = int(config_override["recursion_limit"])
            except ValueError:
                 logger.warning(f"Invalid recursion_limit in config_override. Using default/env var value: {runtime_config['recursion_limit']}")
        # Add other potential runtime overrides here
        # runtime_config.update(config_override) # Or just update blindly?

    logger.info(f"Invoking agent graph with config: {runtime_config}")

    try:
        # Invoke the graph
        final_state = fact_checking_agent.invoke(initial_state, config=runtime_config)
        logger.info("Agent graph invocation finished.")

        # Extract the final result from the state
        if final_state and isinstance(final_state, dict) and final_state.get('final_result'):
            result: FactCheckResult = final_state['final_result']
            logger.info(f"Successfully extracted FactCheckResult: Verdict='{result.verdict}'")
            # Ensure the claim field in the result matches the input claim
            if result.claim != claim:
                logger.warning(f"Final result claim '{result.claim}' differs from input '{claim}'. Overwriting.")
                result.claim = claim
            # Ensure claim_id is set
            if not getattr(result, 'claim_id', None):
                result.claim_id = claim_id
            return result
        else:
            logger.error("Agent graph finished but no 'final_result' found in the final state.")
            # Try to get some info from intermediate steps if available
            explanation = "Agent execution finished unexpectedly without a final result." 
            steps = final_state.get("intermediate_steps", []) if isinstance(final_state, dict) else []
            if steps:
                explanation += f" Last observation: {steps[-1][1] if steps else 'N/A'}"
            return FactCheckResult(claim=claim, claim_id=claim_id, verdict="Error", confidence_score=0.0, explanation=explanation, evidence_sources=[])

    except Exception as e:
        logger.exception(f"An error occurred during agent graph execution for claim '{claim}': {e}")
        return FactCheckResult(claim=claim, claim_id=claim_id, verdict="Error", confidence_score=0.0, explanation=f"Runtime error during fact check: {e}", evidence_sources=[])

# --- Example Usage (Directly) ---
if __name__ == "__main__":
    test_claim_1 = "The Great Wall of China is visible from the Moon with the naked eye."
    print(f"\n--- Testing Claim 1: {test_claim_1} ---")
    result_1 = run_fact_check(test_claim_1)
    print("\n--- Result 1 ---")
    print(f"Verdict: {result_1.verdict}")
    print(f"Confidence: {result_1.confidence_score}")
    print(f"Explanation: {result_1.explanation}")
    print("Sources:")
    if result_1.evidence_sources:
        for src in result_1.evidence_sources:
            print(f"  - URL: {src.url}, Title: {src.title}, Tool: {src.source_tool}")
    else:
        print("  No sources provided.")

    # test_claim_2 = "Elon Musk founded Microsoft."
    # print(f"\n--- Testing Claim 2: {test_claim_2} ---")
    # result_2 = run_fact_check(test_claim_2)
    # print("\n--- Result 2 ---")
    # print(f"Verdict: {result_2.verdict}")
    # print(f"Confidence: {result_2.confidence_score}")
    # print(f"Explanation: {result_2.explanation}")
    # print("Sources:")
    # if result_2.evidence_sources:
    #     for src in result_2.evidence_sources:
    #          print(f"  - URL: {src.url}, Title: {src.title}, Tool: {src.source_tool}")
    # else:
    #      print("  No sources provided.") 