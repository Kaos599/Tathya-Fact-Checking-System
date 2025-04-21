import operator
import logging
import uuid
import json
from typing import TypedDict, Annotated, Sequence, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from .config import get_primary_llm
from .tools import create_agent_tools
from .prompts import AGENT_SYSTEM_PROMPT # Use the updated prompt
from .models import FactCheckResult, EvidenceSource # Import schema for structured output

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Represents the state of the fact-checking agent."""
    claim: str
    claim_id: str  # Unique identifier for this fact-check request
    messages: Annotated[Sequence[BaseMessage], operator.add]
    intermediate_steps: Annotated[List[tuple[ToolInvocation, str]], operator.add] = []
    # Add a field to hold the final structured result
    final_result: Optional[FactCheckResult] = None


# --- Agent Nodes ---

def agent_node(state: AgentState, agent, tools, name: str):
    """Node that calls the agent model to decide the next action."""
    logger.info(f"[{name}] Agent node executing.")
    # Prepare messages, including system prompt with tool descriptions
    system_prompt = AGENT_SYSTEM_PROMPT.format(
        tool_descriptions="\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    )
    messages_with_system = [HumanMessage(content=system_prompt)] + list(state["messages"])

    # Add intermediate steps to the prompt history for context
    for tool_call, observation in state.get("intermediate_steps", []):
        messages_with_system.append(AIMessage(content=f"Tool Call: {tool_call.tool}({tool_call.tool_input}) was executed."))
        messages_with_system.append(HumanMessage(content=f"Observation from {tool_call.tool}: {observation}"))

    logger.info(f"[{name}] Invoking agent LLM...")
    # Ensure the input to the agent is just the list of messages
    # The agent runnable internally handles converting this list to the expected input format
    result = agent.invoke(messages_with_system)

    logger.info(f"[{name}] Agent LLM response received: {result}")
    # We reuse the same state attribute for simplicity
    return {"messages": [result]}


def tool_node(state: AgentState, tool_executor, name: str):
    """Node that executes the tool chosen by the agent."""
    logger.info(f"[{name}] Tool node executing.")
    messages = state["messages"]
    last_message = messages[-1]

    # Ensure the last message contains tool calls
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        logger.warning(f"[{name}] No tool calls found in the last message. Returning to agent.")
        # If the last message isn't an AIMessage or has no tool calls, 
        # we should probably not proceed with tool execution. Return empty.
        # The `should_continue` logic might route this to the end or back to the agent.
        return {"messages": []} # Return no new messages

    tool_invocation_list = []
    tool_messages = [] # Store tool messages to return
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]
        logger.info(f"[{name}] Preparing to execute tool: {tool_name} with input: {tool_input}")

        # Handle the verification tool needing intermediate steps
        if tool_name == "verification_tool":
            # Serialize intermediate steps before adding them
            serializable_steps = []
            # Adjust unpacking based on whether tool_call_id was added to the tuple
            # Assuming original structure: tuple[ToolInvocation, str]
            for inv, obs in state.get("intermediate_steps", []):
                 serializable_steps.append({
                     "tool": inv.tool,
                     "tool_input": inv.tool_input, # Assumes tool_input itself is serializable
                     "observation": obs
                 })
            tool_input["intermediate_steps"] = serializable_steps # Pass serializable list
            tool_input["original_claim"] = state["claim"]
            logger.info(f"[{name}] Added *serialized* state (claim, intermediate_steps) to verification_tool input.")

        tool_invocation = ToolInvocation(tool=tool_name, tool_input=tool_input)
        tool_call_id = tool_call['id'] # Get the ID for the response message

        # Execute the tool invocation
        try:
            # The response is a list of ToolMessage objects
            # Use tool_executor.invoke for single tool call for simplicity here
            # If batching is truly needed, loop through responses
            response = tool_executor.invoke(tool_invocation)
            logger.info(f"[{name}] Tool execution completed. Responses: {response}")

            # Check response type and extract content
            if isinstance(response, ToolMessage):
                observation = response.content
            elif isinstance(response, str):
                observation = response
            else:
                observation = str(response) # Fallback
            
            # Add the invocation and observation to intermediate_steps
            current_intermediate = state.get("intermediate_steps", [])
            current_intermediate.append((tool_invocation, observation))

            # Create the ToolMessage for the history
            tool_messages.append(ToolMessage(content=observation, tool_call_id=tool_call_id))
            
        except Exception as e:
            # Use the specific tool_invocation from this iteration for logging
            tool_name_for_error = tool_invocation.tool if tool_invocation else "Unknown Tool"
            logger.error(f"[{name}] Error executing tool {tool_name_for_error}: {e}", exc_info=True)
            # Return error message as observation
            error_message = f"Error executing tool {tool_name_for_error}: {e}"
            # Append the failed invocation and error message to intermediate steps
            current_intermediate = state.get("intermediate_steps", [])
            # Ensure tool_invocation is defined before appending
            if tool_invocation:
                current_intermediate.append((tool_invocation, error_message))

            # IMPORTANT: Still return a ToolMessage, but with the error content
            # Use the tool_call_id from the current iteration
            tool_messages.append(ToolMessage(content=error_message, tool_call_id=tool_call_id))

    # Return the collected tool messages and updated intermediate steps
    # Only update intermediate_steps if current_intermediate was potentially modified
    if tool_messages: # Check if any tool was processed (successfully or with error)
        return {"messages": tool_messages, "intermediate_steps": current_intermediate}
    else:
        # If the loop didn't run (e.g., no tool calls), return empty
        return {"messages": []}


# --- NEW Node for Final Answer Generation ---
def generate_final_answer_node(state: AgentState, name: str):
    """Generates the final structured answer using an LLM bound to FactCheckResult."""
    logger.info(f"[{name}] Generating final structured answer.")
    synthesis_model = get_primary_llm()
    if not synthesis_model:
        logger.error(f"[{name}] Primary LLM (for synthesis) not configured. Cannot generate final answer.")
        # Fallback: Create a basic FactCheckResult indicating the error
        return {"final_result": FactCheckResult(claim=state['claim'], claim_id=state['claim_id'], verdict="Error", confidence_score=0.0, explanation="Synthesis LLM not configured.", evidence_sources=[])}

    # Use the LLM with structured output bound to the FactCheckResult schema
    try:
        structured_llm = synthesis_model.with_structured_output(FactCheckResult)
    except Exception as e:
        logger.error(f"[{name}] Error binding FactCheckResult schema to the synthesis model: {e}", exc_info=True)
        return {"final_result": FactCheckResult(
            claim=state['claim'], claim_id=state['claim_id'],
            verdict="Error", confidence_score=0.0,
            explanation=f"Failed to configure structured output: {e}", evidence_sources=[]
        )}

    # Prepare the simplified synthesis prompt (focus on context and task)
    formatted_evidence = "\n\n".join([
        f"Tool Used: {action.tool}\nInput: {action.tool_input}\nObservation: {obs}"
        for action, obs in state["intermediate_steps"]
    ])
    synthesis_prompt_content = f"""
Original Claim: {state['claim']}

Collected Evidence and Agent Analysis:
{formatted_evidence}

Task: Based *only* on the provided claim and evidence, generate a final fact-check result. Determine the verdict, confidence score, provide a concise explanation, and list the relevant evidence sources used.
""" # Removed explicit JSON structure instruction

    try:
        logger.info(f"[{name}] Invoking structured LLM for final synthesis.")
        # Invoke the LLM configured for structured output
        # The result should directly be a FactCheckResult object
        final_result: FactCheckResult = structured_llm.invoke([HumanMessage(content=synthesis_prompt_content)])

        # Log success before post-processing
        logger.info(f"[{name}] Successfully received structured output from LLM.")

    except Exception as e:
        logger.error(f"[{name}] Error during final answer synthesis with structured output: {e}", exc_info=True)
        final_result = FactCheckResult(
            claim=state['claim'], claim_id=state['claim_id'],
            verdict="Error", confidence_score=0.0,
            explanation=f"Runtime error during structured synthesis: {e}",
            evidence_sources=[]
        )

    # --- Post-processing: Add source_tool if missing and ensure claim_id ---
    processed_sources = []
    if final_result and final_result.evidence_sources:
        for source in final_result.evidence_sources:
            # Ensure source is an EvidenceSource object or dict before proceeding
            if isinstance(source, (EvidenceSource, dict)):
                # If it's a dict, try converting first
                if isinstance(source, dict):
                    try:
                        source = EvidenceSource(**source)
                    except Exception as pydantic_e:
                        logger.warning(f"[{name}] Could not convert source dict to EvidenceSource: {pydantic_e}. Skipping source: {source}")
                        continue

                # Now it should be an EvidenceSource object
                if not getattr(source, 'source_tool', None):
                    # Try to guess based on URL or context - IMPROVEMENT NEEDED
                    source.source_tool = "Agent Synthesized" # Placeholder
                processed_sources.append(source)
            else:
                logger.warning(f"[{name}] Skipping unexpected item in evidence_sources (expected EvidenceSource or dict): {type(source)} - {source}")

    # Update the final_result object only if it exists
    if final_result:
        final_result.evidence_sources = processed_sources
        # Ensure claim_id is present
        final_result.claim_id = state['claim_id']
    # -------------------------------------------------------- #

    return {"final_result": final_result}


# --- Conditional Edge Logic ---
def should_continue(state: AgentState) -> str:
    """Determines whether to continue iteration or finish."""
    messages = state["messages"]
    last_message = messages[-1]
    # If the agent explicitly calls the FINISH action
    if hasattr(last_message, 'tool_calls') and any(tc.get("name") == "FINISH" for tc in last_message.tool_calls):
         logger.info("Agent decided to FINISH. Routing to final answer generation.")
         return "generate_final_answer"
    # If the agent produced a response without tool calls (should ideally call FINISH)
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        logger.warning("Agent did not request tool usage or FINISH. Assuming finish.")
        return "generate_final_answer" # Or handle potential errors/loops
    # Otherwise, continue with the tool node
    else:
        logger.info("Agent requested tool usage. Routing to tool node.")
        return "continue"

# --- Graph Definition ---
def create_fact_checking_agent_graph():
    """Builds the LangGraph agent for fact-checking."""
    # Define default tool config here instead of loading from load_config
    tool_config = {
        "enable_tavily": True,
        "tavily_max_results": 5,
        "enable_wikidata": False,
        "enable_duckduckgo": True,
        "duckduckgo_max_results": 5,
        "enable_google_search": False, # Requires SerpAPI setup
        "enable_web_scraper": True, # Assuming scrape_webpages_tool is defined in tools.py
        "enable_verification_tool": True
    }
    # Get required LLM using specific getter
    agent_llm = get_primary_llm()
    if not agent_llm:
        raise ValueError("Primary LLM (for agent) configuration is missing or invalid.")

    # Create tools using the defined config
    tools = create_agent_tools(tool_config)
    if not tools:
        logger.warning("No tools configured for the agent. Functionality will be limited.")
        # Consider adding a default tool or raising an error if tools are essential
    
    # Ensure the agent LLM supports tool calling
    try:
        agent_runnable = agent_llm.bind_tools(tools)
        logger.info(f"Agent LLM bound with {len(tools)} tools.")
    except AttributeError:
        logger.error("The configured agent LLM does not seem to support the `.bind_tools` method. Ensure it's a tool-calling compatible model (e.g., from OpenAI, Anthropic, Gemini).")
        raise ValueError("Agent LLM does not support tool binding.")
    except Exception as e:
        logger.error(f"Error binding tools to agent LLM: {e}", exc_info=True)
        raise

    tool_executor = ToolExecutor(tools)

    # Define the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", lambda state: agent_node(state, agent=agent_runnable, tools=tools, name="Agent"))
    workflow.add_node("tools", lambda state: tool_node(state, tool_executor=tool_executor, name="ToolExecutor"))
    workflow.add_node("final_answer_generator", lambda state: generate_final_answer_node(state, name="FinalAnswerGenerator"))

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "generate_final_answer": "final_answer_generator",
        },
    )

    # Add edges
    workflow.add_edge("tools", "agent")
    workflow.add_edge("final_answer_generator", END)

    # Compile the graph
    app = workflow.compile()
    logger.info("Fact-checking agent graph compiled successfully.")
    return app

# --- Main Execution Logic (Example) ---
if __name__ == "__main__":
    logger.info("Compiling and running fact-checking agent graph...")
    agent_graph = create_fact_checking_agent_graph()

    # Example usage:
    claim_to_check = "Paris is the capital of Spain."
    logger.info(f"Starting agent run for claim: '{claim_to_check}'")
    # Include a unique claim_id for the example initial state
    initial_state = {"claim": claim_to_check, "claim_id": str(uuid.uuid4()), "messages": [HumanMessage(content=f"Fact check this claim: {claim_to_check}")]}

    final_state = agent_graph.invoke(initial_state)

    logger.info("\n--- Agent Run Finished ---")
    logger.info(f"Original Claim: {final_state.get('claim')}")

    if final_state.get('final_result'):
        result: FactCheckResult = final_state['final_result']
        logger.info(f"Final Verdict: {result.verdict}")
        logger.info(f"Confidence: {result.confidence_score}")
        logger.info(f"Explanation: {result.explanation}")
        logger.info("Evidence Sources:")
        if result.evidence_sources:
            for i, source in enumerate(result.evidence_sources):
                source_info = f"  {i+1}. Title: {getattr(source, 'title', 'N/A')}, URL: {getattr(source, 'url', 'N/A')}, Tool: {getattr(source, 'source_tool', 'N/A')}"
                logger.info(source_info)
        else:
            logger.info("  No evidence sources provided in the final result.")
    else:
        logger.warning("No final_result found in the agent's final state.")
        logger.info("Intermediate Steps Dump:")
        for i, (action, obs) in enumerate(final_state.get("intermediate_steps", [])):
            logger.info(f"  Step {i+1}: Tool={action.tool}, Input={action.tool_input}, Obs={obs[:200]}...") # Truncate observation

