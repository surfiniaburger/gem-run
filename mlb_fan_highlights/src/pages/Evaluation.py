import pandas as pd
import random
import string
import uuid
import plotly.graph_objects as go
import vertexai
from vertexai.preview import reasoning_engines
from vertexai.preview.evaluation import EvalTask
from vertexai.preview.evaluation.metrics import (
    PointwiseMetric,
    PointwiseMetricPromptTemplate,
)
import time
from google.api_core.exceptions import ResourceExhausted
import streamlit as st  # Import streamlit
from genseng import MetricsStorage
import streamlit.components.v1 as components

# Add BigQuery Configuration
BQ_PROJECT_ID = "gem-rush-007"
BQ_DATASET_ID = "mlb_agent_evaluation"


# --- Streamlit Page Configuration ---
st.set_page_config(page_title="MLB Agent Evaluation", layout="wide")

# --- Initialize Vertex AI (replace placeholders) ---
PROJECT_ID = "gem-rush-007"  # YOUR PROJECT ID
LOCATION = "us-central1"
STAGING_BUCKET = "gs://gem-rush-007-reasoning-engine"
vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

# Initialize BigQuery metrics storage
@st.cache_resource
def get_metrics_storage():
    """Create or get the MetricsStorage instance."""
    return MetricsStorage(BQ_PROJECT_ID, BQ_DATASET_ID, max_retries=5)

metrics_storage = get_metrics_storage()


# --- Prepare Evaluation Datasets ---
single_tool_eval_data = {
    "prompt": [
        "What was the score of the last Rangers game?",
        "Who are the top 5 hitters on the Yankees by batting average?",
        "Show me all plays where Aaron Judge hit a home run against the Red Sox.",
        "List the games the Dodgers played on 2024-09-15",
        "Find the games where the Mets played against the Braves.",
        "Get me the player stats of games played on 2024-08-10",
    ],
    "reference_tool_name": [
        "fetch_team_games",
        "fetch_team_player_stats",
        "fetch_player_plays_by_opponent",
        "fetch_team_games",
        "fetch_team_games_by_opponent",
        "fetch_player_game_stats",
    ],
}
single_tool_eval_df = pd.DataFrame(single_tool_eval_data)

trajectory_eval_data = {
    "prompt": [
        "What were the results of the last two Rangers games, and who were the starting pitchers in those games?",
        "Did any Yankees players hit more than one home run against the Red Sox in their last series?  If so, who?",
        "Give me the results for Dodgers games on 2024-09-15",
        "Give me the stats for Rangers players against the Astros on 2024-09-15",
        "Did Adolis Garcia have more than 2 hits against the Angels this season?",
        "What was the final score of the rangers game against the dodgers on 2024-09-24?",
    ],
    "reference_trajectory": [
        [{"tool_name": "fetch_team_games", "tool_input": {"team_name": "rangers", "limit": 2}}],
        [{"tool_name": "fetch_team_plays_by_opponent", "tool_input": {"team_name": "yankees", "opponent_team": "red sox"}}],
        [{"tool_name": "fetch_team_games", "tool_input": {"team_name": "dodgers", "specific_date": "2024-09-15"}}],
        [{
            "tool_name": "fetch_team_player_stats_by_opponent",
            "tool_input": {"team_name": "rangers", "opponent_team": "astros", "specific_date": "2024-09-15"},
        }],
        [{"tool_name": "fetch_player_plays_by_opponent", "tool_input": {"player_name": "Adolis Garcia", "team_name": "rangers", "opponent_team": "angels"}}],
        [{"tool_name": "fetch_team_games_by_opponent", "tool_input": {"team_name": "rangers", "opponent_team": "dodgers", "specific_date": "2024-09-24"}}],
    ],
}
trajectory_eval_df = pd.DataFrame(trajectory_eval_data)

response_eval_data = {
    "prompt": [
        "Summarize the last Rangers game.",
        "How did Shohei Ohtani perform in his last 3 games?",
        "What is the yankees record against the red sox?",
        "What is the rangers record against the red sox on 2024-07-14?",
        "Give me the results for all games played by the rangers?"
    ],
    "reference_response": [
        "The Rangers played the [Opponent] on [Date]. The final score was [Score]. [Key Play Summary].",
        "In his last three games, Shohei Ohtani had [X] hits, [Y] home runs, and [Z] RBIs.",
        "The results of games between the yankees and the red sox are: ...",
        "The results of games between the rangers and the red sox on 2024-07-14 are: ...",
        "The results of all games played by the rangers are: ..."
    ],
    "reference_trajectory": [
        [{"tool_name": "fetch_team_games", "tool_input": {"team_name": "rangers", "limit": 1}}],
        [{"tool_name": "fetch_player_plays", "tool_input": {"player_name": "Shohei Ohtani", "team_name": "angels", "limit": 30}}],
        [{"tool_name": "fetch_team_games_by_opponent", "tool_input": {"team_name": "yankees", "opponent_team": "red sox"}}],
        [{"tool_name": "fetch_team_games_by_opponent", "tool_input": {"team_name": "rangers", "opponent_team": "red sox", "specific_date":"2024-07-14"}}],
        [{"tool_name": "fetch_team_games", "tool_input": {"team_name": "rangers"}}],
    ],

}
response_eval_df = pd.DataFrame(response_eval_data)

byod_eval_data = {
    "prompt": [
        "What was the score of the last Rangers game?",
        "Who pitched for the Yankees yesterday?",
    ],
    "reference_trajectory": [
        [{"tool_name": "fetch_team_games", "tool_input": {"team_name": "rangers", "limit": 1}}],
        [{"tool_name": "fetch_team_games", "tool_input": {"team_name": "yankees", "limit": 1}}],
    ],
    "predicted_trajectory": [
        [{"tool_name": "fetch_team_games", "tool_input": {"team_name": "rangers", "limit": 1}}],
        [{"tool_name": "fetch_team_player_stats", "tool_input": {"team_name": "yankees"}}],
    ],
    "response": [
        "The Rangers lost to the Astros 5-3.",
        "The starting pitcher was [Pitcher Name].  Relievers included...",
    ],
}
byod_eval_df = pd.DataFrame(byod_eval_data)

# --- Define Utility Functions ---
def get_id(length: int = 8) -> str:
    """Generate a unique id of a specified length."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

def display_eval_report(eval_result) -> None:
    """Display the evaluation results using Streamlit components."""
    metrics_df = pd.DataFrame.from_dict(eval_result.summary_metrics, orient="index").T
    st.markdown("### Summary Metrics")
    st.dataframe(metrics_df)
    st.markdown("### Row-wise Metrics")
    st.dataframe(eval_result.metrics_table)

def display_drilldown(row: pd.Series) -> None:
    """Displays drill-down view for trajectory data within a row."""
    if 'intermediate_steps' in row and isinstance(row['intermediate_steps'], list):
        predicted_trajectories = []
        for step in row['intermediate_steps']:
            if isinstance(step, tuple) and len(step) > 1 and isinstance(step[0], dict):
                tool_info = step[0]
                tool_name = tool_info.get('name')
                tool_input = tool_info.get('arguments', {})
                if tool_name:
                    predicted_trajectories.append({'tool_name': tool_name, 'tool_input': tool_input})
    else:
        return

    if 'reference_trajectory' not in row or not isinstance(row["reference_trajectory"], list):
        return
    if not isinstance(predicted_trajectories, list):
        return

    for predicted_trajectory, reference_trajectory in zip(predicted_trajectories, row["reference_trajectory"]):
        st.markdown(f"**Tool Names:** Predicted: `{predicted_trajectory.get('tool_name')}` | Reference: `{reference_trajectory.get('tool_name')}`")

        pred_input = predicted_trajectory.get("tool_input", {})
        ref_input = reference_trajectory.get("tool_input", {})

        for key in set(list(pred_input.keys()) + list(ref_input.keys())):  # Iterate over all keys
            pred_val = pred_input.get(key, "N/A")  # Get predicted value, default to "N/A"
            ref_val = ref_input.get(key, "N/A")  # Get reference value, default to "N/A"
            st.write(f"**{key}:** Predicted: `{pred_val}` | Reference: `{ref_val}`")
        st.markdown("---")


def display_dataframe_rows(df: pd.DataFrame, columns: list[str] | None = None, num_rows: int = 3, drilldown: bool = False) -> None:
    """Display a subset of DataFrame rows using Streamlit."""
    if columns:
        df = df[columns]
    for idx, row in df.head(num_rows).iterrows():
        st.markdown("#### Row " + str(idx + 1))
        for col in df.columns:
            st.markdown(f"**{col.replace('_', ' ').title()}**:")
            st.write(row[col])
        if drilldown and "predicted_trajectory" in df.columns and "reference_trajectory" in df.columns:
            display_drilldown(row)
        st.markdown("___")

def plot_bar_plot(eval_result, title: str, metrics: list[str] = None) -> None:
    """Plot a bar chart using Plotly and display it in Streamlit."""
    summary_metrics = eval_result.summary_metrics
    if metrics:
        summary_metrics = {k: v for k, v in summary_metrics.items() if any(m in k for m in metrics)}
    fig = go.Figure(data=[go.Bar(x=list(summary_metrics.keys()), y=list(summary_metrics.values()), name=title)])
    fig.update_layout(barmode="group", title=title)
    st.plotly_chart(fig, use_container_width=True)

def display_radar_plot(eval_result, title: str, metrics: list[str] = None) -> None:
    """Plot a radar chart using Plotly and display it in Streamlit."""
    summary_metrics = eval_result.summary_metrics
    if metrics:
        summary_metrics = {k: v for k, v in summary_metrics.items() if any(m in k for m in metrics)}
    min_val = min(summary_metrics.values())
    max_val = max(summary_metrics.values())
    fig = go.Figure(data=[go.Scatterpolar(r=list(summary_metrics.values()),
                                            theta=list(summary_metrics.keys()),
                                            fill="toself",
                                            name=title)])
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[min_val, max_val])),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Define Metrics and Evaluation Functions ---
# (Define your metrics as before using PointwiseMetric and PointwiseMetricPromptTemplate)
single_tool_usage_metrics = []
for tool_name in single_tool_eval_df["reference_tool_name"].unique():
    criteria = {
        "Correct Tool Selection": (
            f"Determine if the agent selected the correct tool: '{tool_name}'. "
            f"The correct tool should be able to directly answer the user's prompt. "
            f"Consider whether the chosen tool matches the expected tool for this type of query."
        )
    }
    rating_rubric = {"1": f"Correct tool: {tool_name}", "0": "Incorrect tool"}
    prompt_template = PointwiseMetricPromptTemplate(
        criteria=criteria,
        rating_rubric=rating_rubric,
        input_variables=["prompt", "predicted_trajectory"],
    )
    metric = PointwiseMetric(
        metric=f"single_tool_correct_{tool_name}",
        metric_prompt_template=prompt_template,
    )
    single_tool_usage_metrics.append(metric)

trajectory_metrics = [
    "trajectory_exact_match",
    "trajectory_in_order_match",
    "trajectory_any_order_match",
    "trajectory_precision",
    "trajectory_recall",
]

# --- Response Metrics (CORRECTED) ---
# Remove the string "helpfulness" and define it as a PointwiseMetric

# Custom metric: Helpfulness (NOW CORRECTLY DEFINED)
helpfulness_criteria = {
    "Helpfulness": (
        "Assess how helpful the agent's response is to the user. "
        "Consider if the response is relevant, useful, and provides "
        "the information the user likely needs, even if not explicitly stated."
    )
}
helpfulness_rating_rubric = {"1": "Helpful", "0": "Not Helpful"}
helpfulness_prompt_template = PointwiseMetricPromptTemplate(
    criteria=helpfulness_criteria,
    rating_rubric=helpfulness_rating_rubric,
    input_variables=["prompt", "response"],  # Correct input variables
)
helpfulness_metric = PointwiseMetric(
    metric="helpfulness",  # Now a proper metric name
    metric_prompt_template=helpfulness_prompt_template,
)


# Custom metric: Factual Accuracy
criteria_factual = {
    "Factual Accuracy": (
        "Evaluate whether the agent's response is factually accurate "
        "given the MLB data sources. Consider if scores, dates, player statistics, and names are correct."
    )
}
pointwise_rating_rubric = {"1": "Factually Accurate", "0": "Not Factually Accurate"}
factual_accuracy_prompt_template = PointwiseMetricPromptTemplate(
    criteria=criteria_factual,
    rating_rubric=pointwise_rating_rubric,
    input_variables=["prompt", "predicted_trajectory", "response"],
)
factual_accuracy_metric = PointwiseMetric(
    metric="mlb_factual_accuracy",
    metric_prompt_template=factual_accuracy_prompt_template,
)

# Custom metric: Completeness
completeness_criteria = {
    "Completeness": (
        "Assess whether the agent's response fully addresses all aspects of the prompt. "
        "Consider if multiple pieces of information are provided and no key details are missing."
    )
}
completeness_rating_rubric = {"1": "Complete", "0": "Incomplete"}
completeness_prompt_template = PointwiseMetricPromptTemplate(
    criteria=completeness_criteria,
    rating_rubric=completeness_rating_rubric,
    input_variables=["prompt", "response"],
)
completeness_metric = PointwiseMetric(
    metric="mlb_completeness",
    metric_prompt_template=completeness_prompt_template,
)

# Combine all response metrics
response_metrics_with_custom = [
    "safety",
    "coherence",
    "groundedness",
    helpfulness_metric,  # Use the PointwiseMetric object
    factual_accuracy_metric,
    completeness_metric,
]

response_metrics = [
    "safety",
    "coherence",
    "groundedness",
    helpfulness_metric
]

REASONING_ENGINE_RESOURCE_NAME = "projects/1011675918473/locations/us-central1/reasoningEngines/6357798444265897984"
remote_agent = reasoning_engines.ReasoningEngine(REASONING_ENGINE_RESOURCE_NAME)

def retry_with_backoff(func, max_attempts=5, initial_delay=1, backoff_factor=2):
    """Retries the provided function with exponential backoff on ResourceExhausted errors."""
    attempts = 0
    while attempts < max_attempts:
        try:
            return func()
        except ResourceExhausted as e:
            attempts += 1
            wait = initial_delay * (backoff_factor ** (attempts - 1))
            # Optionally, log a warning or display via Streamlit:
            st.warning(f"ResourceExhausted error encountered. Retrying in {wait} seconds... (Attempt {attempts} of {max_attempts})")
            time.sleep(wait)
    raise Exception("Maximum retry attempts reached.")


def run_evaluation(agent, eval_df, metrics, experiment_name, experiment_run_name_prefix):
    """Run an evaluation task and display the results in Streamlit."""
    experiment_run_name = f"{experiment_run_name_prefix}-{uuid.uuid4()}"
    eval_task = EvalTask(
        dataset=eval_df,
        metrics=metrics,
        experiment=experiment_name,
    )
    # Wrap the evaluate call with our retry logic
    def call_evaluate():
        return eval_task.evaluate(runnable=agent, experiment_run_name=experiment_run_name)

    eval_result = retry_with_backoff(call_evaluate)

    st.markdown("## Evaluation Report")
    display_eval_report(eval_result)
    st.markdown("## Detailed Metrics")
    display_dataframe_rows(eval_result.metrics_table, num_rows=5, drilldown=True)
    # If trajectory-related metrics are present, show a bar plot
    if any("trajectory" in str(metric).lower() for metric in metrics):
        plot_bar_plot(
            eval_result,
            title="Trajectory Metrics",
            metrics=[f"{metric}/mean" for metric in trajectory_metrics],
        )
    return eval_result


def modify_run_evaluation(agent, eval_df, metrics, experiment_name, 
                         experiment_run_name_prefix, evaluation_type: str,
                         metrics_storage: MetricsStorage):
    """Modified run_evaluation function that includes BigQuery storage."""
    experiment_run_name = f"{experiment_run_name_prefix}-{uuid.uuid4()}"
    eval_task = EvalTask(
        dataset=eval_df,
        metrics=metrics,
        experiment=experiment_name,
    )

    def call_evaluate():
        return eval_task.evaluate(runnable=agent, experiment_run_name=experiment_run_name)

    eval_result = retry_with_backoff(call_evaluate)

    # Save metrics to BigQuery with streaming inserts
    save_success = metrics_storage.save_evaluation_results(
        eval_result=eval_result,
        experiment_name=experiment_name,
        evaluation_type=evaluation_type,
        experiment_run_name=experiment_run_name 
    )

    if not save_success:
        st.warning("Some metrics may not have been saved to BigQuery. Check the logs for details.")

    # Original visualization code
    st.markdown("## Evaluation Report")
    display_eval_report(eval_result)
    st.markdown("## Detailed Metrics")
    display_dataframe_rows(eval_result.metrics_table, num_rows=5, drilldown=True)
    
    if any("trajectory" in str(metric).lower() for metric in metrics):
        plot_bar_plot(
            eval_result,
            title="Trajectory Metrics",
            metrics=[f"{metric}/mean" for metric in trajectory_metrics],
        )
    
    return eval_result

# --- Run Evaluations ---
EXPERIMENT_NAME = "mlb-agent-evaluation"

st.sidebar.header("Evaluation Options")
option = st.sidebar.selectbox("Choose Evaluation Type",
                              ["Single Tool", "Trajectory", "Response (General)", "Response (Custom)", "BYOD"])

if option == "Single Tool":
    st.header("Single Tool Evaluation")
    eval_result = modify_run_evaluation(
        remote_agent, 
        single_tool_eval_df, 
        single_tool_usage_metrics, 
        EXPERIMENT_NAME, 
        "single-tool",
        "Single Tool",
        metrics_storage
    )
elif option == "Trajectory":
    st.header("Trajectory Evaluation")
    eval_result = modify_run_evaluation(
        remote_agent, 
        trajectory_eval_df, 
        trajectory_metrics, 
        EXPERIMENT_NAME, 
        "trajectory",
        "Trajectory",
        metrics_storage
    )
elif option == "Response (General)":
    st.header("Response Evaluation (General)")
    eval_result = modify_run_evaluation(
        remote_agent, 
        response_eval_df, 
        response_metrics, 
        EXPERIMENT_NAME, 
        "response-general",
        "Response General",
        metrics_storage
    )
elif option == "Response (Custom)":
    st.header("Response Evaluation (with Custom Metrics)")
    eval_result = modify_run_evaluation(
        remote_agent, 
        response_eval_df, 
        response_metrics_with_custom, 
        EXPERIMENT_NAME, 
        "response-custom",
        "Response Custom",
        metrics_storage
    )
elif option == "BYOD":
    st.header("BYOD Evaluation")
    eval_result = modify_run_evaluation(
        remote_agent, 
        byod_eval_df, 
        response_metrics_with_custom, 
        EXPERIMENT_NAME, 
        "byod",
        "BYOD",
        metrics_storage
    )


LOOKER_STUDIO_REPORT_URL="https://lookerstudio.google.com/embed/reporting/ef4aa403-8f18-4cd9-a3e9-44f1f67f394a/page/X9KyE"
# Embed Looker Studio report (AFTER evaluation runs)
st.subheader("Looker Studio Report")
components.iframe(LOOKER_STUDIO_REPORT_URL, height=600)