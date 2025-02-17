import pandas as pd
import random
import string
from IPython.display import HTML, Markdown, display
import plotly.graph_objects as go
from vertexai.preview.evaluation import EvalTask
from vertexai.preview.evaluation.metrics import (
    PointwiseMetric,
    PointwiseMetricPromptTemplate,
    # TrajectorySingleToolUse,  <- No longer needed
)
import uuid
from vertexai.preview import reasoning_engines
import vertexai

# Initialize Vertex AI (replace placeholders)
PROJECT_ID = "gem-rush-007"  # YOUR PROJECT ID
LOCATION = "us-central1"
STAGING_BUCKET = "gs://gem-rush-007-reasoning-engine"

vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

# --- Single Tool Use Evaluation Dataset ---
single_tool_eval_data = {
    "prompt": [
        "What was the score of the last Rangers game?",
        "Who are the top 5 hitters on the Yankees by batting average?",
        "Show me all plays where Aaron Judge hit a home run against the Red Sox.",
        "List the games the Dodgers played on 2024-09-15",
        "Find the games where the Mets played against the Braves.",
        "Get me the player stats of games played on 2024-08-10"
    ],
    "reference_tool_name": [
        "fetch_team_games",
        "fetch_team_player_stats",
        "fetch_player_plays_by_opponent",
        "fetch_team_games",
        "fetch_team_games_by_opponent",
        "fetch_player_game_stats"
    ],
}
single_tool_eval_df = pd.DataFrame(single_tool_eval_data)
print("Single Tool Evaluation Dataset:")
print(single_tool_eval_df)



# --- Trajectory Evaluation Dataset ---
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
        [
            {"tool_name": "fetch_team_games", "tool_input": {"team_name": "rangers", "limit": 2}},
        ],
        [
            {"tool_name": "fetch_team_plays_by_opponent", "tool_input": {"team_name": "yankees", "opponent_team": "red sox"}},
        ],
        [
            {"tool_name": "fetch_team_games", "tool_input": {"team_name": "dodgers", "specific_date": "2024-09-15"}}
        ],
        [
            {
                "tool_name": "fetch_team_player_stats_by_opponent",
                "tool_input": {"team_name": "rangers", "opponent_team": "astros", "specific_date": "2024-09-15"},
            },
        ],
        [
           {"tool_name": "fetch_player_plays_by_opponent", "tool_input": {"player_name": "Adolis Garcia", "team_name": "rangers", "opponent_team": "angels"}},
        ],
        [
            {"tool_name": "fetch_team_games_by_opponent", "tool_input": {"team_name": "rangers", "opponent_team": "dodgers", "specific_date":"2024-09-24"}},
        ],

    ],
}
trajectory_eval_df = pd.DataFrame(trajectory_eval_data)
print("\nTrajectory Evaluation Dataset:")
print(trajectory_eval_df)


# --- Response Evaluation Dataset ---
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
        [{"tool_name": "fetch_team_games", "tool_input": {"team_name": "rangers", "limit": 1}}
        ],
        [{"tool_name": "fetch_player_plays", "tool_input": {"player_name": "Shohei Ohtani", "team_name": "angels", "limit": 30}}
        ],
        [{"tool_name": "fetch_team_games_by_opponent", "tool_input": {"team_name": "yankees", "opponent_team": "red sox"}}
        ],
        [{"tool_name": "fetch_team_games_by_opponent", "tool_input": {"team_name": "rangers", "opponent_team": "red sox", "specific_date":"2024-07-14"}}
        ],
        [{"tool_name": "fetch_team_games", "tool_input": {"team_name": "rangers"}}
        ],
    ],

}
response_eval_df = pd.DataFrame(response_eval_data)
print("\nResponse Evaluation Dataset:")
print(response_eval_df)

# --- BYOD (Bring Your Own Dataset) Example ---
byod_eval_data = {
    "prompt": [
        "What was the score of the last Rangers game?",
        "Who pitched for the Yankees yesterday?"
    ],
      "reference_trajectory": [
        [{"tool_name": "fetch_team_games", "tool_input": {"team_name": "rangers", "limit": 1}}
        ],
        [{"tool_name": "fetch_team_games", "tool_input": {"team_name": "yankees", "limit": 1}}
        ],
    ],
    "predicted_trajectory": [
        [{"tool_name": "fetch_team_games", "tool_input": {"team_name": "rangers", "limit": 1}}],
        [{"tool_name": "fetch_team_player_stats", "tool_input": {"team_name": "yankees"}}]
    ],
    "response": [
        "The Rangers lost to the Astros 5-3.",
        "The starting pitcher was [Pitcher Name].  Relievers included...",
    ],
}
byod_eval_df = pd.DataFrame(byod_eval_data)
print("\nBYOD Evaluation Dataset:")
print(byod_eval_df)



# --- Single Tool Usage Metrics ---
single_tool_usage_metrics = []
for tool_name in single_tool_eval_df["reference_tool_name"].unique():
    criteria = {
        "Correct Tool Selection": (
            f"Determine if the agent selected the correct tool: '{tool_name}'. "
            f"The correct tool should be able to directly answer the user's prompt. "
            f"Consider whether the chosen tool matches the expected tool for this type of query."
        )
    }
    rating_rubric = {
        "1": f"Correct tool: {tool_name}",
        "0": f"Incorrect tool",
    }
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

print("Single Tool Usage Metrics:", single_tool_usage_metrics)


# --- Trajectory Metrics ---
trajectory_metrics = [
    "trajectory_exact_match",
    "trajectory_in_order_match",
    "trajectory_any_order_match",
    "trajectory_precision",
    "trajectory_recall",
]
print("Trajectory Metrics:", trajectory_metrics)

# --- Response Metrics ---
response_metrics = ["safety", "coherence", "groundedness", "helpfulness"]
print("Response Metrics (General):", response_metrics)


# --- Custom Response Metric: Factual Accuracy ---
criteria = {
    "Factual Accuracy": (
        "Evaluate whether the agent's response is factually accurate "
        "given the information available from MLB data sources.  Consider:\n"
        "  - Are game scores and dates correct?\n"
        "  - Are player statistics (e.g., batting average, home runs) correct?\n"
        "  - Are team names and player names correctly used?\n"
        "  - Does the response avoid making claims that are not supported by the data?\n"
        "Provide specific examples from the response and the expected data to support your evaluation."
    )
}

pointwise_rating_rubric = {
    "1": "Factually Accurate",
    "0": "Not Factually Accurate",
}

factual_accuracy_prompt_template = PointwiseMetricPromptTemplate(
    criteria=criteria,
    rating_rubric=pointwise_rating_rubric,
    input_variables=["prompt", "predicted_trajectory", "response"],
)

factual_accuracy_metric = PointwiseMetric(
    metric="mlb_factual_accuracy",
    metric_prompt_template=factual_accuracy_prompt_template,
)

response_metrics_with_custom = response_metrics + [factual_accuracy_metric]
print("Response Metrics (with Custom):", response_metrics_with_custom)



# --- Custom Response Metric: Completeness ---
completeness_criteria = {
    "Completeness": (
        "Assess whether the agent's response fully addresses all aspects of the prompt. Consider:\n"
        "  - If the prompt asks for multiple pieces of information, are all provided?\n"
        "  - If the prompt has implicit requirements (e.g., 'last game' implies a date), are they met?\n"
        "  - Are there any missing details that a user would reasonably expect?\n"
    )
}

completeness_rating_rubric = {
    "1": "Complete",
    "0": "Incomplete",
}

completeness_prompt_template = PointwiseMetricPromptTemplate(
    criteria=completeness_criteria,
    rating_rubric=completeness_rating_rubric,
    input_variables=["prompt", "response"],
)

completeness_metric = PointwiseMetric(
    metric="mlb_completeness",
    metric_prompt_template=completeness_prompt_template,
)

response_metrics_with_custom = response_metrics_with_custom + [completeness_metric]
def get_id(length: int = 8) -> str:
    """Generate a uuid of a specified length (default=8)."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def display_eval_report(eval_result: pd.DataFrame) -> None:
    """Display the evaluation results."""
    metrics_df = pd.DataFrame.from_dict(eval_result.summary_metrics, orient="index").T
    display(Markdown("### Summary Metrics"))
    display(metrics_df)

    display(Markdown(f"### Row-wise Metrics"))
    display(eval_result.metrics_table)


def display_drilldown(row: pd.Series) -> None:
    """Displays a drill-down view for trajectory data within a row."""

    style = "white-space: pre-wrap; width: 800px; overflow-x: auto;"

    if not (
        isinstance(row["predicted_trajectory"], list)
        and isinstance(row["reference_trajectory"], list)
    ):
        return

    for predicted_trajectory, reference_trajectory in zip(
        row["predicted_trajectory"], row["reference_trajectory"]
    ):
        display(
            HTML(
                f"Tool Names:{predicted_trajectory['tool_name'], reference_trajectory['tool_name']}"
            )
        )

        if not (
            isinstance(predicted_trajectory.get("tool_input"), dict)
            and isinstance(reference_trajectory.get("tool_input"), dict)
        ):
            continue

        for tool_input_key in predicted_trajectory["tool_input"]:
            print("Tool Input Key: ", tool_input_key)

            if tool_input_key in reference_trajectory["tool_input"]:
                print(
                    "Tool Values: ",
                    predicted_trajectory["tool_input"][tool_input_key],
                    reference_trajectory["tool_input"][tool_input_key],
                )
            else:
                print(
                    "Tool Values: ",
                    predicted_trajectory["tool_input"][tool_input_key],
                    "N/A",
                )
        print("\n")
    display(HTML(""))


def display_dataframe_rows(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    num_rows: int = 3,
    display_drilldown: bool = False,  #  No change needed, parameter is correct
) -> None:
    """Displays a subset of rows from a DataFrame, optionally including a drill-down view."""

    if columns:
        df = df[columns]

    base_style = "font-family: monospace; font-size: 14px; white-space: pre-wrap; width: auto; overflow-x: auto;"
    header_style = base_style + "font-weight: bold;"

    for _, row in df.head(num_rows).iterrows():
        for column in df.columns:
            display(
                HTML(
                    f"{column.replace('_', ' ').title()}: "
                )
            )
            display(HTML(f"{row[column]}"))

        display(HTML(""))

        if (  # Corrected: Now *calls* display_drilldown
            display_drilldown
            and "predicted_trajectory" in df.columns
            and "reference_trajectory" in df.columns
        ):
            display_drilldown(row)  # Corrected line


def plot_bar_plot(
    eval_result: pd.DataFrame, title: str, metrics: list[str] = None
) -> None:
    fig = go.Figure()
    data = []

    summary_metrics = eval_result.summary_metrics
    if metrics:
        summary_metrics = {
            k: summary_metrics[k]
            for k, v in summary_metrics.items()
            if any(selected_metric in k for selected_metric in metrics)
        }

    data.append(
        go.Bar(
            x=list(summary_metrics.keys()),
            y=list(summary_metrics.values()),
            name=title,
        )
    )

    fig = go.Figure(data=data)
    fig.update_layout(barmode="group")
    fig.show()


def display_radar_plot(eval_results, title: str, metrics=None):
    """Plot the radar plot."""
    fig = go.Figure()
    summary_metrics = eval_results.summary_metrics
    if metrics:
        summary_metrics = {
            k: summary_metrics[k]
            for k, v in summary_metrics.items()
            if any(selected_metric in k for selected_metric in metrics)
        }

    min_val = min(summary_metrics.values())
    max_val = max(summary_metrics.values())

    fig.add_trace(
        go.Scatterpolar(
            r=list(summary_metrics.values()),
            theta=list(summary_metrics.keys()),
            fill="toself",
            name=title,
        )
    )
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[min_val, max_val])),
        showlegend=True,
    )
    fig.show()


REASONING_ENGINE_RESOURCE_NAME = "projects/1011675918473/locations/us-central1/reasoningEngines/2601796355038904320"
remote_agent = reasoning_engines.ReasoningEngine(REASONING_ENGINE_RESOURCE_NAME)

def run_evaluation(agent, eval_df, metrics, experiment_name, experiment_run_name_prefix):
    """Runs an evaluation task and displays the results."""

    experiment_run_name = f"{experiment_run_name_prefix}-{uuid.uuid4()}"

    eval_task = EvalTask(
        dataset=eval_df,
        metrics=metrics,
        experiment=experiment_name,
    )

    eval_result = eval_task.evaluate(
        runnable=agent,
        experiment_run_name=experiment_run_name
    )

    display_eval_report(eval_result)
    display_dataframe_rows(eval_result.metrics_table, num_rows=5, display_drilldown=True)
    if any("trajectory" in str(metric).lower() for metric in metrics):
      plot_bar_plot(
          eval_result,
          title="Trajectory Metrics",
          metrics=[f"{metric}/mean" for metric in trajectory_metrics],
        )

    return eval_result


# --- Run Single Tool Evaluation ---
EXPERIMENT_NAME = "mlb-agent-evaluation"
single_tool_eval_result = run_evaluation(
    remote_agent,
    single_tool_eval_df,
    single_tool_usage_metrics,
    EXPERIMENT_NAME,
    "single-tool",
)


# --- Run Trajectory Evaluation ---
trajectory_eval_result = run_evaluation(
    remote_agent,
    trajectory_eval_df,
    trajectory_metrics,
    EXPERIMENT_NAME,
    "trajectory",
)

# --- Run Response Evaluation (General) ---
response_eval_result_general = run_evaluation(
    remote_agent,
    response_eval_df,
    response_metrics,
    EXPERIMENT_NAME,
    "response-general",
)

# --- Run Response Evaluation (with Custom Metrics) ---
response_eval_result_custom = run_evaluation(
    remote_agent,
    response_eval_df,
    response_metrics_with_custom,
    EXPERIMENT_NAME,
    "response-custom",
)

# --- Run BYOD Evaluation ---
byod_eval_result = run_evaluation(
    remote_agent,
    byod_eval_df,
    response_metrics_with_custom,
    EXPERIMENT_NAME,
    "byod",
)