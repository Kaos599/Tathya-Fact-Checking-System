import operator
import logging
import uuid
import json
from typing import TypedDict, Annotated, Sequence, List, Optional, Dict, Any
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_core.output_parsers.json import JsonOutputParser
from pydantic import BaseModel, Field

from .config import get_primary_llm
from .tools import create_agent_tools
from .prompts import AGENT_SYSTEM_PROMPT
from .models import FactCheckResult, EvidenceSource

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Represents the state of the fact-checking agent."""
    claim: str
    claim_id: str  # Unique identifier for this fact-check request
    messages: Annotated[Sequence[BaseMessage], operator.add]
    intermediate_steps: Annotated[List[tuple[ToolInvocation, Any]], operator.add] = []
    # Add a field to hold the final structured result
    final_result: Optional[FactCheckResult] = None


# --- Agent Nodes ---

def agent_node(state: AgentState, agent, tools, name: str):
    """Node that calls the agent model to decide the next action."""
    logger.info(f"[{name}] Agent node executing.")
    # Prepare system prompt
    system_prompt_content = AGENT_SYSTEM_PROMPT.format(
        tool_descriptions="\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    )

    # *** Simplified History Handling ***
    # Prepend the system message to the current message history from the state
    messages_for_llm = [SystemMessage(content=system_prompt_content)] + list(state["messages"])

    logger.info(f"[{name}] Invoking agent LLM with {len(messages_for_llm)} total messages (system + history)...")
    # Log the last few messages for debugging context if needed
    # logger.debug(f"[{name}] Last messages sent to LLM: {messages_for_llm[-3:]}")

    # Invoke the agent - expecting the input format to be a list of messages
    result: BaseMessage = agent.invoke(messages_for_llm)

    # Log the raw result
    logger.info(f"[{name}] Agent LLM raw response: {result}")

    # Ensure the result is always a list for operator.add compatibility
    # The graph will add this message to the state's 'messages' list
    output_messages = [result] if isinstance(result, BaseMessage) else []
    if not output_messages:
         logger.warning(f"[{name}] Agent LLM did not return a valid BaseMessage. Result: {result}")


    # Return the message(s) to be added to the state
    return {"messages": output_messages}


def tool_node(state: AgentState, tool_executor, name: str):
    """Node that executes the tool chosen by the agent."""
    logger.info(f"[{name}] Tool node executing.")
    messages = state["messages"]
    if not messages:
        logger.warning(f"[{name}] No messages found in state. Cannot execute tool.")
        return {"messages": [], "intermediate_steps": state.get("intermediate_steps", [])}

    last_message = messages[-1]

    if not isinstance(last_message, AIMessage) or not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        logger.warning(f"[{name}] Last message is not an AIMessage with tool calls. Type: {type(last_message)}. Content: {last_message}")
        return {"messages": [], "intermediate_steps": state.get("intermediate_steps", [])}

    tool_invocation_list = []
    tool_messages = []
    intermediate_steps_updates = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_input = tool_call.get("args", {})
        tool_call_id = tool_call.get('id')

        if not tool_name or not tool_call_id:
            logger.warning(f"[{name}] Skipping invalid tool call in message: {tool_call}")
            continue

        logger.info(f"[{name}] Preparing to execute tool: {tool_name} (Call ID: {tool_call_id}) with input: {tool_input}")

        if tool_name == "verification_tool":
            serializable_steps = []
            for inv, obs in state.get("intermediate_steps", []):
                 serializable_steps.append({
                     "tool": inv.tool,
                     "tool_input": inv.tool_input,
                     "observation": str(obs)
                 })
            tool_input["intermediate_steps"] = serializable_steps
            tool_input["original_claim"] = state["claim"]
            logger.info(f"[{name}] Added serialized state (claim, intermediate_steps) to verification_tool input.")

        tool_invocation = ToolInvocation(tool=tool_name, tool_input=tool_input)
        tool_invocation_list.append(tool_invocation)

        try:
            response = tool_executor.invoke(tool_invocation)
            logger.info(f"[{name}] Tool '{tool_name}' execution completed. Response type: {type(response)}")

            if isinstance(response, ToolMessage):
                observation = response.content
            elif isinstance(response, (str, dict, list)):
                 observation = response
            else:
                 observation = str(response)

            intermediate_steps_updates.append((tool_invocation, observation))

            tool_messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

        except Exception as e:
            logger.error(f"[{name}] Error executing tool {tool_name} (Call ID: {tool_call_id}): {e}", exc_info=True)
            error_message = f"Error executing tool {tool_name}: {str(e)}"

            intermediate_steps_updates.append((tool_invocation, error_message))

            tool_messages.append(ToolMessage(content=error_message, tool_call_id=tool_call_id))

    new_intermediate_steps = state.get("intermediate_steps", []) + intermediate_steps_updates

    return {"messages": tool_messages, "intermediate_steps": new_intermediate_steps}


# --- Structure for the final synthesis LLM call ---
class SynthesisResult(BaseModel):
     verdict: str = Field(..., description="The final verdict (e.g., TRUE, FALSE, PARTIALLY TRUE/MIXTURE, UNCERTAIN).")
     confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the verdict (0.0 to 1.0).")
     explanation: str = Field(..., description="Detailed explanation justifying the verdict, citing evidence using [Step N] or [Source URL] notation where N is the step number from the provided evidence summary.")


# --- Synthesis Prompt ---
SYNTHESIS_PROMPT_TEMPLATE = """
You are a neutral fact-checking analyst. Your task is to synthesize the findings from the research process and provide a final verdict on the original claim.

Original Claim:
"{claim}"

Evidence Summary:
This summary includes the outputs of various research tools called during the investigation. Each step represents a tool call and its observation. Use this information to form your judgment.
---
{evidence_summary}
---

Based *only* on the evidence summary provided above:
1.  Determine the most accurate verdict for the claim (e.g., TRUE, FALSE, PARTIALLY TRUE/MIXTURE, UNCERTAIN).
2.  Assign a confidence score (0.0 to 1.0) reflecting the certainty of your verdict based on the evidence's strength and consistency.
3.  Write a detailed explanation justifying your verdict. **Crucially, you MUST cite the specific evidence that supports each part of your explanation.** Use the format `[Step N]` where N is the step number from the evidence summary (e.g., "[Step 1]", "[Step 3]"). You can also cite specific URLs if they are highly relevant using `[Source URL: <url>]`. Be objective and stick strictly to the provided evidence. Do not introduce external knowledge.

Provide your response as a JSON object matching the following schema:
{format_instructions}
"""


# --- NEW Node for Final Answer Generation ---
def generate_final_answer_node(state: AgentState, name: str) -> Dict[str, FactCheckResult]:
    """Generates the final structured answer, extracting sources programmatically and calling LLM for synthesis."""
    logger.info(f"[{name}] Generating final structured answer.")
    synthesis_model = get_primary_llm()
    if not synthesis_model:
        logger.error(f"[{name}] Primary LLM (for synthesis) not configured. Cannot generate final answer.")
        return {"final_result": FactCheckResult(claim=state['claim'], claim_id=state['claim_id'], verdict="Error", confidence_score=0.0, explanation="Synthesis LLM not configured.", evidence_sources=[])}

    extracted_sources: List[EvidenceSource] = []
    evidence_summary_for_llm = ""
    unique_urls = set()
    intermediate_steps = state.get("intermediate_steps", [])

    logger.info(f"[{name}] Processing {len(intermediate_steps)} intermediate steps for source extraction and summary.")

    for step_index, (tool_invocation, observation) in enumerate(intermediate_steps):
        step_number = step_index + 1
        tool_name = tool_invocation.tool
        tool_input_str = str(tool_invocation.tool_input)
        source_details = {}

        evidence_summary_for_llm += f"\n--- Step {step_number}: Tool: {tool_name}, Input: {tool_input_str[:100]}... ---\n"

        obs_data = None
        obs_text_summary = ""

        if isinstance(observation, str):
            try:
                obs_data = json.loads(observation)
                obs_text_summary = f"Observation (parsed JSON): {str(obs_data)[:500]}..."
            except json.JSONDecodeError:
                obs_data = None
                obs_text_summary = f"Observation (text): {observation[:500]}..."
        elif isinstance(observation, (dict, list)):
            obs_data = observation
            obs_text_summary = f"Observation (structured data): {str(obs_data)[:500]}..."
        else:
            logger.warning(f"[{name}] Step {step_number}: Unknown or None observation type for tool {tool_name}: {type(observation)}. Skipping source extraction.")
            obs_text_summary = f"Observation: [Could not parse source data - type: {type(observation)}]"

        evidence_summary_for_llm += obs_text_summary + "\n"

        if obs_data is not None:
            results_list = None

            if tool_name == "tavily_search_results_json" and isinstance(obs_data, list):
                results_list = obs_data
            elif tool_name == "duckduckgo_search" and isinstance(obs_data, list):
                 results_list = obs_data
            elif tool_name == "news_search" and isinstance(obs_data, dict) and isinstance(obs_data.get('articles'), list):
                results_list = obs_data['articles']
            elif tool_name == "wikidata_entity_search" and isinstance(obs_data, dict) and isinstance(obs_data.get('search'), list):
                 results_list = obs_data['search']
            elif tool_name == "gemini_google_search_and_parse" and isinstance(obs_data, dict):
                 gemini_sources = obs_data.get('sources', [])
                 logger.info(f"[{name}] Step {step_number}: Extracting sources from Gemini output ({len(gemini_sources)} found).")
                 for idx, url in enumerate(gemini_sources):
                     if isinstance(url, str) and url and url not in unique_urls:
                         try:
                              extracted_sources.append(EvidenceSource(
                                  id=f"step-{step_number}-{tool_name}-{idx}",
                                  source_tool=tool_name,
                                  url=url,
                                  title=f"Gemini Source {idx+1}",
                                  snippet=obs_data.get('summary'),
                                  raw_content={'url': url, 'summary': obs_data.get('summary'), 'key_facts': obs_data.get('key_facts')},
                                  retrieval_date=datetime.now().isoformat()
                              ))
                              unique_urls.add(url)
                         except Exception as p_err:
                              logger.warning(f"[{name}] Failed to create EvidenceSource for Gemini source {idx} from step {step_number}: {p_err}")
                 continue

            if results_list is not None:
                logger.info(f"[{name}] Step {step_number}: Processing {len(results_list)} items from results list for tool {tool_name}.")
                for i, item in enumerate(results_list):
                    if not isinstance(item, dict): 
                        logger.warning(f"[{name}] Step {step_number}: Skipping non-dict item in results_list: {item}")
                        continue # Skip non-dict items
                    
                    logger.debug(f"[{name}] Step {step_number}: Processing item {i}: {item}") # DEBUG ITEM

                    # Extract common fields, checking multiple possible keys
                    title = item.get('title', item.get('label', f"Source {i+1} from {tool_name}"))
                    snippet = item.get('body', item.get('snippet', item.get('description', item.get('content', 'No Snippet'))))
                    url = item.get('href', item.get('url', item.get('concepturi'))) # Check common URL keys

                    logger.debug(f"[{name}] Step {step_number}: Extracted - URL: {url}, Title: {title}, Snippet: {snippet[:50]}...") # DEBUG EXTRACTION

                    # Ensure URL is a valid string before proceeding
                    if isinstance(url, str) and url and url not in unique_urls:
                         logger.debug(f"[{name}] Step {step_number}: URL '{url}' is valid and unique. Attempting to create EvidenceSource.") # DEBUG APPEND ATTEMPT
                         try:
                              source_details_item = {
                                  'id': f"step-{step_number}-{tool_name}-{i}",
                                  'source_tool': tool_name,
                                  'url': url,
                                  'title': title,
                                  'snippet': snippet,
                                  'raw_content': item, # Store the original item dict
                                  'retrieval_date': datetime.now().isoformat()
                              }
                              extracted_sources.append(EvidenceSource(**source_details_item))
                              unique_urls.add(url)
                              logger.debug(f"[{name}] Step {step_number}: Successfully appended EvidenceSource for URL: {url}") # DEBUG APPEND SUCCESS
                         except Exception as p_err:
                              # This warning was already here, ensure it's effective
                              logger.warning(f"[{name}] Failed to create EvidenceSource for item {i} from step {step_number} ({tool_name}): {p_err} - Data: {source_details_item}")
                    elif not isinstance(url, str):
                         logger.warning(f"[{name}] Step {step_number}: Skipping item {i} due to non-string URL: {url} (Type: {type(url)})") # DEBUG SKIP
                    elif not url:
                         logger.warning(f"[{name}] Step {step_number}: Skipping item {i} due to empty URL.") # DEBUG SKIP
                    elif url in unique_urls:
                         logger.debug(f"[{name}] Step {step_number}: Skipping duplicate URL: {url}") # DEBUG SKIP

    logger.info(f"[{name}] Extracted {len(extracted_sources)} unique sources programmatically.")

    parser = JsonOutputParser(pydantic_object=SynthesisResult)
    synthesis_prompt = PromptTemplate(
        template=SYNTHESIS_PROMPT_TEMPLATE,
        input_variables=["claim", "evidence_summary"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    synthesis_chain = synthesis_prompt | synthesis_model | parser

    logger.info(f"[{name}] Invoking synthesis LLM...")
    try:
        synthesis_output: SynthesisResult = synthesis_chain.invoke({
            "claim": state['claim'],
            "evidence_summary": evidence_summary_for_llm
        })

        logger.info(f"[{name}] Synthesis LLM call successful. Verdict: {synthesis_output.get('verdict')}")

        final_structured_result = FactCheckResult(
            claim=state['claim'],
            claim_id=state['claim_id'],
            verdict=synthesis_output.get('verdict', 'Error'),
            confidence_score=synthesis_output.get('confidence_score', 0.0),
            explanation=synthesis_output.get('explanation', 'Error during synthesis.'),
            evidence_sources=extracted_sources
        )

    except Exception as e:
        logger.error(f"[{name}] Error during synthesis LLM call or parsing: {e}", exc_info=True)
        final_structured_result = FactCheckResult(
            claim=state['claim'],
            claim_id=state['claim_id'],
            verdict="Error",
            confidence_score=0.0,
            explanation=f"Error during final synthesis: {e}",
            evidence_sources=extracted_sources
        )

    return {"final_result": final_structured_result}


# --- Graph Edges ---

def should_continue(state: AgentState) -> str:
    """Determines the next step after the agent speaks."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
         if any(tool_call.get("name") == "FINISH" for tool_call in last_message.tool_calls):
              logger.info("Agent decided to FINISH. Proceeding to final answer generation.")
              return "generate_final_answer"

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info("Agent requested tool execution. Proceeding to tool node.")
        return "tools"

    logger.info("Agent did not request tools or FINISH. Ending process and attempting final answer generation.")
    return "generate_final_answer"


# --- Build the Graph ---

def create_fact_checking_agent_graph(config: Optional[Dict] = None):
    """Creates and compiles the LangGraph for the fact-checking agent."""
    if config is None:
        config = {}

    logger.info("Creating fact-checking agent graph...")
    agent_llm = get_primary_llm()
    if not agent_llm:
         raise ValueError("Primary LLM is not configured. Cannot create agent.")

    tools = create_agent_tools(config)
    tool_executor = ToolExecutor(tools)

    try:
         agent_runnable = agent_llm.bind_tools(tools)
         logger.info("Tools bound to agent LLM.")
    except AttributeError:
        logger.warning("LLM does not support .bind_tools(). Tool usage might be less reliable.")
        agent_runnable = agent_llm
    except Exception as e:
        logger.error(f"Error binding tools to LLM: {e}. Proceeding without binding.", exc_info=True)
        agent_runnable = agent_llm

    bound_agent_node = lambda state: agent_node(state, agent=agent_runnable, tools=tools, name="Agent")
    bound_tool_node = lambda state: tool_node(state, tool_executor=tool_executor, name="Action")
    bound_generate_final_answer_node = lambda state: generate_final_answer_node(state, name="Synthesizer")

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", bound_agent_node)
    workflow.add_node("tools", bound_tool_node)
    workflow.add_node("generate_final_answer", bound_generate_final_answer_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "generate_final_answer": "generate_final_answer"
        }
    )

    workflow.add_edge("tools", "agent")

    workflow.add_edge("generate_final_answer", END)

    graph = workflow.compile()
    logger.info("Fact-checking agent graph compiled successfully.")
    return graph


# --- Example Usage (Optional) ---
if __name__ == '__main__':
    run_config = {
        "enable_tavily": True,
        "enable_duckduckgo": False,
        "enable_wikidata": False,
        "enable_web_scraper": True,
        "enable_verification_tool": False,
    }
    fact_checker_graph = create_fact_checking_agent_graph(config=run_config)

    claim_to_check = "The Eiffel Tower is located in Berlin."
    unique_id = str(uuid.uuid4())
    inputs = {
        "claim": claim_to_check,
        "claim_id": unique_id,
        "messages": [HumanMessage(content=f"Please investigate this claim: {claim_to_check}")]
    }

    logger.info(f"--- Starting Fact Check Run (ID: {unique_id}) ---")
    final_state = fact_checker_graph.invoke(inputs)
    logger.info(f"--- Fact Check Run Finished (ID: {unique_id}) ---")

    final_result = final_state.get("final_result")

    if final_result:
         print("\n--- Final Result ---")
         print(final_result.json(indent=2))
    else:
        print("\n--- Error: No final result found in state ---")
        print("Final State:", final_state)

