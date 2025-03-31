# mlb_agent_graph_refined.py
# --- Imports (combine necessary imports from previous agent script and ingestion script) ---
import pandas as pd
import json
from datetime import datetime, UTC
import requests
import logging
import time
from typing import Dict, List, Tuple, Any, Optional, TypedDict
import os
import re 
from ratelimit import limits, sleep_and_retry

# LangGraph and LangChain specific
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_vertexai import ChatVertexAI
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Google Cloud specific
from google.cloud import bigquery
from google.api_core.exceptions import BadRequest, NotFound
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

# --- Configuration (Ensure these match ingestion script) ---
GCP_PROJECT_ID = "silver-455021"
GCP_LOCATION = "us-central1"
BQ_DATASET_ID = "mlb_rag_data_2024"
BQ_RAG_TABLE_ID = "rag_documents"      # For summaries & play snippets + embeddings
BQ_PLAYS_TABLE_ID = "plays"            # For structured play-by-play data
BQ_FULL_RAG_TABLE_ID = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_RAG_TABLE_ID}"
BQ_FULL_PLAYS_TABLE_ID = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_PLAYS_TABLE_ID}"
BQ_INDEX_NAME = "rag_docs_embedding_idx"

VERTEX_LLM_MODEL = "gemini-1.5-flash-001"
VERTEX_EMB_MODEL = "text-embedding-004"
EMBEDDING_TASK_TYPE = "RETRIEVAL_QUERY" # Use RETRIEVAL_QUERY for search queries
EMBEDDING_DIMENSIONALITY = 768
VERTEX_EMB_RPM = 1400 # Adjust
MLB_API_CALLS = 9
MLB_API_RATE_LIMIT = 60

# --- Logging and Clients (Initialize as before) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    if not vertexai.global_config.project:
         vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)
    # Use ChatVertexAI for LangChain compatibility
    model = ChatVertexAI(model_name=VERTEX_LLM_MODEL, project=GCP_PROJECT_ID, location=GCP_LOCATION, temperature=0.2)
    # Separate model instance maybe for structured output/tool use if needed
    structured_output_model = ChatVertexAI(model_name=VERTEX_LLM_MODEL, project=GCP_PROJECT_ID, location=GCP_LOCATION, temperature=0.0)
    # Embedding model instance (LangChain wrapper isn't strictly needed here, but could be used)
    emb_model = TextEmbeddingModel.from_pretrained(VERTEX_EMB_MODEL)
    logger.info(f"Initialized Google Cloud clients and LangChain Model for project {GCP_PROJECT_ID}")
except Exception as e:
    logger.critical(f"Failed to initialize Google Cloud clients: {e}", exc_info=True)
    raise RuntimeError("Critical initialization failed.") from e

# --- Agent State Definition (Added critique, revision tracking) ---
class AgentState(TypedDict):
    task: str
    game_pk: Optional[int]
    plan: str
    structured_data: Optional[Any] # Can be Dict or List[Dict] now
    narrative_context: Optional[List[str]]
    draft: Optional[str]       # The current draft being worked on
    critique: Optional[str]    # Feedback from the reflection node
    generated_content: str     # Final output
    revision_number: int       # Start at 0, increment with each generation attempt
    max_revisions: int         # Max refinement loops
    error: Optional[str]

# --- BQ Query Functions (Keep from ingestion, slightly adapted for agent use) ---
def execute_bq_query(query: str) -> Optional[pd.DataFrame]:
    # (Same as in ingestion script)
    try:
        logger.info(f"Executing BQ Query: {query[:200]}...")
        query_job = bq_client.query(query)
        results = query_job.to_dataframe()
        logger.info(f"BQ Query returned {len(results)} rows.")
        return results
    except Exception as e:
        logger.error(f"Error executing BQ query: {query[:200]}... Error: {e}", exc_info=True)
        return None

def get_structured_game_metadata(game_pk: int) -> Optional[Dict[str, Any]]:
    """Fetches just the game summary doc metadata from BQ RAG table."""
    if not game_pk: return None
    query = f"""
    SELECT game_id, doc_type, metadata, last_updated
    FROM `{BQ_FULL_RAG_TABLE_ID}`
    WHERE game_id = {game_pk} AND doc_type = 'game_summary' LIMIT 1"""
    df = execute_bq_query(query)
    if df is not None and not df.empty:
        data_dict = df.iloc[0].to_dict()
        # Parse metadata JSON
        if isinstance(data_dict.get('metadata'), str):
            try: data_dict['metadata'] = json.loads(data_dict['metadata'])
            except json.JSONDecodeError: logger.warning(f"Could not parse metadata JSON (game {game_pk})")
        return data_dict['metadata'] # Return just the metadata dict
    return None

def get_structured_play_data(game_pk: int, play_filter_criteria: str = "1=1") -> Optional[List[Dict]]:
    """Fetches structured play data from the BQ Plays table, applying filters."""
    if not game_pk: return None
    # Basic safety check on criteria - more robust validation needed for production
    safe_criteria = re.sub(r"[^a-zA-Z0-9_=\s\<\>\'\.\-\(\),%]", "", play_filter_criteria) if play_filter_criteria else "1=1"

    query = f"""
    SELECT * EXCEPT(pitch_data, hit_data, runners_before, runners_after), -- Select core fields
           -- Select nested JSON data if needed, or process later
           pitch_data, hit_data, runners_after
    FROM `{BQ_FULL_PLAYS_TABLE_ID}`
    WHERE game_pk = {game_pk} AND {safe_criteria}
    ORDER BY play_index
    """
    df = execute_bq_query(query)
    if df is not None and not df.empty:
         # Convert DataFrame to list of dicts, parsing JSON strings back
        records = df.to_dict('records')
        for record in records:
            for col in ['pitch_data', 'hit_data', 'runners_after', 'runners_before']: # Add runners_before if stored
                 if col in record and isinstance(record[col], str):
                     try: record[col] = json.loads(record[col])
                     except json.JSONDecodeError: record[col] = None # Handle bad JSON
        logger.info(f"Retrieved {len(records)} structured plays for game {game_pk} with criteria '{safe_criteria}'")
        return records
    logger.warning(f"No structured plays found for game {game_pk} with criteria '{safe_criteria}'")
    return None

# Keep call_vertex_embedding from ingestion script
@sleep_and_retry
@limits(calls=VERTEX_EMB_RPM, period=60)
def call_vertex_embedding_agent(text_inputs: List[str]) -> List[Optional[List[float]]]:
    """Embedding specifically for agent queries/retrieval."""
    results = []
    batch_size = 200 # Adjust if needed
    try:
        all_embeddings = []
        for i in range(0, len(text_inputs), batch_size):
            batch = text_inputs[i:i + batch_size]
            # Use RETRIEVAL_QUERY type for the search query itself
            instances = [TextEmbeddingInput(text=text, task_type="RETRIEVAL_QUERY") for text in batch]
            kwargs = {"output_dimensionality": EMBEDDING_DIMENSIONALITY} # Only specify if not default
            embeddings_batch = emb_model.get_embeddings(instances, **kwargs)
            all_embeddings.extend([emb.values for emb in embeddings_batch])
            if len(text_inputs) > batch_size: time.sleep(1)
        return all_embeddings
    except Exception as e:
        logger.error(f"Error calling Vertex AI Embedding API: {e}", exc_info=True)
        return [None] * len(text_inputs)


def get_narrative_context_vector_search(query_text: str, game_pk: Optional[int] = None, top_n: int = 5) -> List[str]:
    """Performs vector search on the BQ RAG table."""
    # (Similar to ingestion, but use call_vertex_embedding_agent with RETRIEVAL_QUERY)
    if not query_text: return []
    try:
        query_embedding_response = call_vertex_embedding_agent([query_text])
        if not query_embedding_response or not query_embedding_response[0]: return []
        query_embedding = query_embedding_response[0]

        filter_clause = f"AND game_id = {game_pk}" if game_pk else ""
        # Ensure the table name includes project and dataset
        query = f"""
        SELECT base.content, distance
        FROM VECTOR_SEARCH(
            TABLE `{BQ_FULL_RAG_TABLE_ID}`,
            'embedding',
            (SELECT {query_embedding} AS embedding),
            top_k => {top_n},
            distance_type => 'COSINE'
        ) AS base
        WHERE TRUE {filter_clause}
        ORDER BY distance ASC
        """ # Note: Added alias 'base' which might be needed by BQ

        logger.info(f"Performing vector search: '{query_text[:50]}...' (Game PK: {game_pk})")
        df = execute_bq_query(query)
        if df is not None and not df.empty:
            results = df['content'].tolist()
            logger.info(f"Vector search returned {len(results)} snippets.")
            return results
        logger.warning(f"Vector search returned no results.")
        return []
    except Exception as e:
        logger.error(f"Error during vector search: {e}", exc_info=True)
        return []


# --- Refined Retriever Logic ---
# Define Pydantic models for structured LLM output for planning retrieval
class BQQuery(BaseModel):
    """A BigQuery SQL query designed to retrieve specific structured data."""
    query: str = Field(..., description="The SQL query to execute against BigQuery (use full table names like project.dataset.table). Filter by game_pk if relevant.")

class VectorSearch(BaseModel):
    """A query for semantic vector search."""
    query_text: str = Field(..., description="The natural language query to embed and search for in the vector store.")
    filter_by_game: bool = Field(True, description="Whether to restrict the search to the current game_pk.")

class RetrievalPlan(BaseModel):
    """Specifies which data retrieval methods to use."""
    structured_queries: Optional[List[BQQuery]] = Field(default_factory=list, description="List of BigQuery SQL queries for structured data.")
    vector_searches: Optional[List[VectorSearch]] = Field(default_factory=list, description="List of queries for vector search.")

RETRIEVER_PLANNER_PROMPT = """
You are a data retrieval expert for an MLB analysis system. Your goal is to decide HOW to fetch the data needed based on the user's task and the overall plan.

Available Data Sources:
1.  **BQ Structured Metadata (`{rag_table}`):** Contains game-level metadata (teams, score, date, venue) accessible via SQL. Use `WHERE game_id = <game_pk> AND doc_type = 'game_summary'`.
2.  **BQ Structured Plays (`{plays_table}`):** Contains detailed play-by-play data (inning, description, pitch data, hit data, rbi, scores) accessible via SQL. Use `WHERE game_pk = <game_pk>`. Filter further based on the need (e.g., `AND is_scoring_play = TRUE`, `AND batter_id = <id>`).
3.  **BQ Vector Search (`{rag_table}` column `embedding`):** Contains embeddings for game summaries and key play narrative snippets. Use for semantic search (similarity, context, "feeling"). Can be filtered by game_pk.

User Request: {task}
Overall Plan: {plan}
Game ID (if applicable): {game_pk}

Determine the best retrieval actions. Generate a list of specific BQ SQL queries and/or vector search queries needed.

- Use BQ Structured Queries for precise facts, stats, scores, lists of specific plays.
- Use Vector Search for narrative context, summaries, finding 'similar' items, or understanding subjective elements.
- If a game_pk is provided, filter queries appropriately unless the request explicitly asks for cross-game comparison.
- Be specific in your SQL queries (e.g., select needed columns, filter effectively).
- Formulate concise, targeted vector search queries based on the task/plan.
"""

def retrieve_data_node_refined(state: AgentState) -> Dict[str, Any]:
    """Parses the plan and executes structured BQ queries and/or vector searches."""
    logger.info("--- Refined Data Retrieval Node ---")
    task = state.get('task')
    plan = state.get('plan')
    game_pk = state.get('game_pk')

    if not plan: return {"error": "Plan is missing for retrieval."}

    # Use LLM to determine retrieval strategy
    prompt = RETRIEVER_PLANNER_PROMPT.format(
        task=task,
        plan=plan,
        game_pk=game_pk if game_pk else "Not Specified",
        rag_table=BQ_FULL_RAG_TABLE_ID,
        plays_table=BQ_FULL_PLAYS_TABLE_ID
    )
    retrieved_structured_data = []
    retrieved_narrative_context = []

    try:
        logger.info("Generating retrieval action plan...")
        retrieval_actions = structured_output_model.with_structured_output(RetrievalPlan).invoke(prompt)
        logger.info(f"Retrieval Actions: {retrieval_actions}")

        # Execute Structured Queries
        if retrieval_actions.structured_queries:
            for bq_query in retrieval_actions.structured_queries:
                # Simple safety check: ensure game_pk filter if game_pk provided
                query_to_run = bq_query.query
                if game_pk and f"game_pk = {game_pk}" not in query_to_run.lower() and f"game_id = {game_pk}" not in query_to_run.lower():
                     # Attempt to add a WHERE clause intelligently (basic example)
                     if "where" in query_to_run.lower():
                         query_to_run = query_to_run.replace("WHERE", f"WHERE game_pk = {game_pk} AND ") # Assumes game_pk exists in the target table
                     else:
                         # Find table name to append WHERE clause
                         match = re.search(r"FROM\s+`([\w.-]+)`\.`([\w.-]+)`\.`(\w+)`", query_to_run, re.IGNORECASE)
                         if match:
                             query_to_run += f" WHERE game_pk = {game_pk}" # Assumes game_pk is the column name
                         else:
                              logger.warning(f"Could not automatically add game_pk filter to: {query_to_run}")
                     logger.info(f"Modified query for game_pk filter: {query_to_run[:200]}...")


                df_result = execute_bq_query(query_to_run)
                if df_result is not None and not df_result.empty:
                    retrieved_structured_data.extend(df_result.to_dict('records'))
                time.sleep(0.5) # Small delay between queries

        # Execute Vector Searches
        if retrieval_actions.vector_searches:
            for vec_search in retrieval_actions.vector_searches:
                search_game_pk = game_pk if vec_search.filter_by_game else None
                snippets = get_narrative_context_vector_search(vec_search.query_text, search_game_pk)
                retrieved_narrative_context.extend(snippets)
                time.sleep(0.5) # Small delay

    except Exception as e:
        logger.error(f"Error during refined data retrieval: {e}", exc_info=True)
        # Fallback: try fetching basic game metadata if specific retrieval failed
        if game_pk and not retrieved_structured_data:
             logger.warning("Falling back to basic game metadata retrieval.")
             basic_meta = get_structured_game_metadata(game_pk)
             if basic_meta: retrieved_structured_data.append(basic_meta) # Append as list for consistency
        if not retrieved_narrative_context:
             logger.warning("No narrative context retrieved.")
        # Decide if this should be a hard error for the graph
        # return {"error": f"Data retrieval failed: {e}"}

    # Deduplicate narrative context
    unique_narratives = list(dict.fromkeys(retrieved_narrative_context))

    logger.info(f"Retrieved {len(retrieved_structured_data)} structured data records.")
    logger.info(f"Retrieved {len(unique_narratives)} unique narrative snippets.")

    return {
        "structured_data": retrieved_structured_data if retrieved_structured_data else None, # Store results
        "narrative_context": unique_narratives if unique_narratives else None
    }

# --- Reflection and Critique Nodes ---
REFLECTION_PROMPT = """
You are an expert MLB analyst acting as a writing critic. Review the generated draft based on the original request and plan.

Original Request: {task}
Plan: {plan}
Draft:
{draft}

Provide constructive criticism and specific recommendations for improvement. Focus on:
- Accuracy: Does the draft accurately reflect the data? Point out any factual errors.
- Completeness: Does the draft fully address the user's request and the plan? What's missing?
- Storytelling/Engagement: Is the narrative compelling? Does it connect stats to the game flow well? Is it interesting for an MLB fan? Suggest ways to make it more engaging (e.g., add context, highlight drama, use stronger verbs).
- Clarity and Conciseness: Is the writing clear and easy to understand? Is it too verbose or too brief?
- Specific Data Usage: Could specific stats or play details (if available in context, though not explicitly shown here) be integrated better?

If the draft is excellent and requires no changes, respond with "The draft looks excellent and fully addresses the request." Otherwise, provide specific, actionable feedback.
"""

RESEARCH_CRITIQUE_PROMPT = """
You are a research assistant. Based on the critique of the previous draft, generate specific search queries (max 3) to find information needed for the revision.

Critique:
{critique}

Focus on queries that will find facts, stats, context, or narrative examples to address the critique's points (e.g., specific player stats for that game, details about a key play mentioned, historical context). Use BQ table names `{rag_table}` and `{plays_table}` if suggesting SQL.
"""

def reflection_node(state: AgentState) -> Dict[str, str]:
    """Generates critique on the draft."""
    logger.info("--- Reflection Node ---")
    if not state.get('draft'): return {"error": "Draft missing for reflection."}
    prompt = REFLECTION_PROMPT.format(
        task=state['task'],
        plan=state['plan'],
        draft=state['draft']
    )
    try:
        response = model.invoke(prompt)
        critique = response.content
        logger.info(f"Critique: {critique}")
        return {"critique": critique}
    except Exception as e:
        logger.error(f"Error in reflection_node: {e}", exc_info=True)
        return {"error": f"Failed to generate critique: {e}", "critique": "Error generating critique."}

def research_critique_node(state: AgentState) -> Dict[str, Any]:
    """Generates research queries based on critique and fetches data."""
    logger.info("--- Research Critique Node ---")
    if not state.get('critique') or "excellent" in state.get('critique', '').lower():
         logger.info("Critique is positive or missing, skipping research.")
         # Return existing content so generate node can reuse it
         return {"content": state.get("content")} # Changed this: Pass previous content

    prompt = RESEARCH_CRITIQUE_PROMPT.format(
        critique=state['critique'],
        rag_table=BQ_FULL_RAG_TABLE_ID,
        plays_table=BQ_FULL_PLAYS_TABLE_ID
    )
    # Use the same retrieval logic, potentially informed by the critique
    # For simplicity, we'll just re-run a vector search based on the critique text
    # A more advanced system might parse the critique to generate specific BQ queries too
    new_narrative_context = []
    try:
        # Optionally, generate specific queries from critique using LLM
        # queries = structured_output_model.with_structured_output(Queries).invoke(prompt)
        # For now, just search based on the critique text itself
        logger.info("Performing vector search based on critique...")
        new_narrative_context = get_narrative_context_vector_search(state['critique'], state.get('game_pk'))

    except Exception as e:
         logger.error(f"Error generating/executing research queries from critique: {e}", exc_info=True)

    # Combine new context with previous context (optional, can replace)
    # For now, let's add it, generate node needs to handle potential redundancy
    combined_narrative = state.get('narrative_context', []) + new_narrative_context
    unique_combined_narrative = list(dict.fromkeys(combined_narrative)) # Deduplicate

    logger.info(f"Added {len(new_narrative_context)} new snippets based on critique.")

    # Keep structured data from previous retrieval
    return {
        "narrative_context": unique_combined_narrative,
        "structured_data": state.get("structured_data") # Carry over structured data
        }

# --- Add this function definition with the other nodes ---

PLANNER_PROMPT_TEMPLATE = """
You are an expert MLB analyst and content planner. Your goal is to create a plan for fulfilling the user's request regarding an MLB game.
The available data includes structured game metadata (scores, teams, date, venue) and potentially narrative game summaries stored in a database.
Structured play-by-play data might also be queryable. Vector search can find relevant narrative context (like summaries or similar plays).

User Request: {task}
Game ID (if specified): {game_pk}

Based on the request, create a step-by-step plan focusing on:
1.  **Identifying Necessary Data:** Specify *what* data is needed (e.g., final score, specific player stats, summary text, similar historical plays).
2.  **Identifying Retrieval Methods:** Indicate *how* to get the data (e.g., 'BQ Query for metadata', 'Vector Search for narrative summary', 'BQ Query for play-by-play if needed').
3.  **Content Focus:** Briefly outline the key points the final content should cover to satisfy the user request, emphasizing storytelling using the data.

Output only the plan.

Plan:
"""

def planner_node(state: AgentState) -> Dict[str, Any]:
    """Generates a plan to fulfill the user's task."""
    logger.info("--- Planner Node ---")
    task = state.get('task')
    game_pk = state.get('game_pk') # game_pk might be None if not specified by user/task

    if not task:
        logger.error("Task is missing for planner.")
        # Return an error state or a default plan
        return {"error": "Task is missing.", "plan": "Error: Task missing."}

    prompt = PLANNER_PROMPT_TEMPLATE.format(
        task=task,
        game_pk=game_pk if game_pk else "Not Specified (Use latest or context if possible)"
    )
    try:
        # Ensure model is initialized correctly
        if 'model' not in globals() or not isinstance(model, ChatVertexAI):
             raise NameError("Global 'model' (ChatVertexAI) is not initialized.")

        response = model.invoke(prompt)
        plan = response.content
        logger.info(f"Generated Plan:\n{plan}")
        # Ensure plan is not empty
        if not plan or not plan.strip():
            logger.warning("Planner returned an empty plan, using default.")
            plan = "Default Plan: 1. Retrieve basic game info. 2. Generate summary."
            return {"plan": plan} # Return default plan but don't signal error yet

        return {"plan": plan}
    except Exception as e:
        logger.error(f"Error in planner_node: {e}", exc_info=True)
        # Return error and a default plan to potentially allow graceful failure
        return {"error": f"Failed to generate plan: {e}", "plan": "Default Plan due to error: Retrieve basic info."}

# --- Generate Node (Updated Prompt) ---
GENERATOR_PROMPT_REFINED_TEMPLATE = """
You are an expert MLB analyst and storyteller.
Original user request: "{task}"
Plan:
{plan}

You have already generated a draft, and received the following critique:
Critique:
{critique}

Based on the critique AND the available data, revise the draft or generate new content. Utilize all information below:

Structured Data:
```json
{structured_data_json}" \
"Narrative Context (Summaries, Play Snippets, Research based on Critique):
{narrative_context_str}

Instructions:

- Address the Critique: Explicitly incorporate the feedback from the critique.

- Synthesize ALL Data: Combine structured facts/stats with narrative context.

- Deep Storytelling: Connect stats to game flow, explain significance, highlight key moments/matchups. Use specific details if available (pitch types, speeds, hit data).

- Fulfill Original Task: Ensure the final output clearly answers the user's request: "{task}".

Output the improved content in Markdown format.
"""



def generate_node_refined(state: AgentState) -> Dict[str, Any]:
  """Generates or revises content based on data, plan, and critique."""
  logger.info(f"--- Content Generation/Revision Node (Revision: {state.get('revision_number', 0)}) ---")
  task = state.get('task')
  plan = state.get('plan')
  critique = state.get('critique') # Might be None on first pass
  structured_data = state.get('structured_data')
  narrative_context = state.get('narrative_context', [])

  if state.get("error"): return {"error": state.get("error")}
  if not plan: return {"error": "Plan missing."}

  structured_data_json = json.dumps(structured_data, indent=2, default=str) if structured_data else "{}"
  narrative_context_str = "\n---\n".join(narrative_context) if narrative_context else "No narrative context available."

  # Use different prompts for first draft vs revision
  if critique and "excellent" not in critique.lower(): # Revision prompt
    prompt = GENERATOR_PROMPT_REFINED_TEMPLATE.format(
        task=task,
        plan=plan,
        critique=critique,
        structured_data_json=structured_data_json,
        narrative_context_str=narrative_context_str
    )
  else: # First draft prompt (or if critique was positive)
     # Use a slightly simpler prompt if no critique needs addressing
     prompt = f"""
        You are an expert MLB analyst and storyteller.
        User request: "{task}"
        Plan:
        {plan}

        Available Data:
        Structured:
        ```json
        {structured_data_json}
        ```
        Narrative Context:
        {narrative_context_str}

        Generate the content following the plan, using deep data storytelling by synthesizing structured stats and narrative context. Output in Markdown.
        """

  try:
    response = model.invoke(prompt)
    new_draft = response.content
    logger.info(f"Generated/Revised Draft (first 100 chars): {new_draft[:100]}...")
    # Increment revision number *after* successful generation
    current_revision = state.get('revision_number', 0)
    return {
        "draft": new_draft,
        "revision_number": current_revision + 1 # Increment happens here
        }
  except Exception as e:
    logger.error(f"Error in generate_node_refined: {e}", exc_info=True)
    return {"error": f"Failed to generate content: {e}", "draft": state.get("draft") or "Error generating draft."}

#--- Conditional Edge Logic ---
def should_continue(state: AgentState) -> str:
  """Determines whether to reflect or end the process."""
  logger.info("--- Should Continue Node ---")
  revision_number = state.get("revision_number", 1) # Generation increments it, so check > max
  max_revisions = state.get("max_revisions", 2)
  critique = state.get("critique", "")

  if state.get("error"):
    logger.error(f"Error detected: {state['error']}")
    return "END" # Or a specific error end node

  if revision_number > max_revisions:
    logger.info(f"Reached max revisions ({max_revisions}). Ending.")
    return "END"

  # Check if the critique node ran and gave positive feedback
  if critique and "excellent" in critique.lower():
     logger.info("Critique was positive. Ending.")
     return "END"

  logger.info(f"Revision {revision_number} <= {max_revisions}. Continuing to reflection.")
  return "reflect" # Continue the refinement loop

#--- Build the Graph ---
workflow = StateGraph(AgentState)

#Add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retrieve_data", retrieve_data_node_refined) # Use refined retriever
workflow.add_node("generate", generate_node_refined) # Use refined generator
workflow.add_node("reflect", reflection_node)
workflow.add_node("research_critique", research_critique_node)

#Set entry point
workflow.set_entry_point("planner")

#Define edges
workflow.add_edge("planner", "retrieve_data")
workflow.add_edge("retrieve_data", "generate")

#Refinement loop
workflow.add_conditional_edges(
"generate", # Node to branch from
should_continue, # Function to decide the path
{
"reflect": "reflect", # If function returns "reflect", go to reflect node
"END": END # If function returns "END", finish execution
}
)
workflow.add_edge("reflect", "research_critique")
workflow.add_edge("research_critique", "generate") # Loop back to generate


memory = MemorySaver() # Optional: Add if chat history/memory is needed
app = workflow.compile(checkpointer=memory)
app = workflow.compile()


# --- Helper function (add this near the start of mlb_agent_graph_refined.py) ---
# Requires call_mlb_api to be defined in this script as well
@sleep_and_retry
@limits(calls=MLB_API_CALLS, period=MLB_API_RATE_LIMIT)
def call_mlb_api(url: str) -> Dict:
    # (Copy the definition from ingestion script here)
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        if 'application/json' in response.headers.get('Content-Type', ''):
            return response.json()
        else:
            logger.warning(f"Non-JSON response: {url}. Status: {response.status_code}.")
            return {}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling MLB API {url}: {e}")
        return {}

def get_latest_final_game_pk(team_id: int, season: int = 2024) -> Optional[int]:
    """Fetches the most recent *final* game ID for a specific team."""
    url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season}&teamId={team_id}&fields=dates,games,gamePk,officialDate,status,detailedState'
    logger.info(f"Fetching latest game for team {team_id}...")
    schedule_data = call_mlb_api(url)
    latest_game_pk = None
    latest_date = ""

    if schedule_data and 'dates' in schedule_data:
        all_final_games = []
        for date_entry in schedule_data.get('dates', []):
            for game in date_entry.get('games', []):
                 # Look specifically for 'Final' status
                 if game.get('status', {}).get('detailedState') == 'Final':
                    all_final_games.append({
                        'game_id': game.get('gamePk'),
                        'date': game.get('officialDate')
                    })

        if all_final_games:
            # Sort by date descending to get the most recent
            all_final_games.sort(key=lambda x: x['date'], reverse=True)
            latest_game_pk = all_final_games[0]['game_id']
            latest_date = all_final_games[0]['date']
            logger.info(f"Found latest final game for team {team_id}: PK {latest_game_pk} on {latest_date}")

    if not latest_game_pk:
         logger.warning(f"No recent *final* game ID found for team {team_id}, season {season}.")

    return latest_game_pk


# --- Updated Example Usage (at the end of mlb_agent_graph_refined.py) ---
if __name__ == "__main__":
    logger.info("\n--- Running Refined Agent Graph ---")

    # --- Dynamic Game PK ---
    # Choose a default team ID to find the latest game for (e.g., Rangers = 140)
    default_team_id_for_latest = 140
    latest_game_pk = get_latest_final_game_pk(default_team_id_for_latest)

    if not latest_game_pk:
        logger.error(f"Could not find the latest game PK for team {default_team_id_for_latest}. Exiting example.")
        exit() # Or handle differently, maybe fallback to a known good PK

    logger.info(f"Using latest game PK found: {latest_game_pk}")
    # -----------------------

    # Task can now reference the "latest game" implicitly or explicitly
    # task = f"Analyze the key moments and pitching duel in the latest game involving team ID {default_team_id_for_latest} (Game PK: {latest_game_pk})."
    task = f"Provide a detailed recap of game {latest_game_pk}, highlighting impactful plays and player performances." # Example task

    max_loops = 2

    initial_state = {
        "task": task,
        "game_pk": latest_game_pk, # Use the dynamically found PK
        "max_revisions": max_loops,
        "revision_number": 0,
        "plan": None,
        "structured_data": None,
        "narrative_context": [],
        "draft": None,
        "critique": None,
        "generated_content": None,
        "error": None,
    }

    logger.info(f"\nExecuting graph for Task: {task}")

    # Use invoke to get the final state directly
    try:
        # Make sure the graph is compiled correctly before invoking
        if 'app' not in globals():
             # Re-compile if needed (ensure workflow definition is complete above)
             logger.warning("Re-compiling graph 'app'...")
             app = workflow.compile()

        final_state = app.invoke(initial_state, {"recursion_limit": max_loops * 2 + 5})

        if final_state.get("error"):
             print("\n--- Execution Failed ---")
             print(f"Error: {final_state['error']}")
        elif final_state.get("draft"): # Check 'draft' as it holds the last generated content
             print("\n--- Final Generated Content ---")
             print(final_state["draft"])
        else:
            print("\n--- Execution Finished, but no final draft found in state. Check logs. ---")
            print("Final state snapshot:", {k: v for k,v in final_state.items() if k != 'structured_data' and k != 'narrative_context'}) # Avoid printing huge data

    except Exception as e:
         logger.error(f"Error invoking graph: {e}", exc_info=True)
         print(f"\n--- Graph Invocation Error ---")
         print(f"An exception occurred: {e}")