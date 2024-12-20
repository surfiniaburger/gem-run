# mlb_fan_highlights/src/data_processing/data_processor.py

from apache_beam.io import ReadFromText, WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms import (
    DoFn,
    ParDo,
    combiners,
)

# Replace with your Cloud Storage bucket names
raw_bucket = "your-raw-data-bucket"
processed_bucket = "your-processed-data-bucket"


class ExtractGameEvents(DoFn):
    """Extracts key game events (e.g., home runs, strikeouts) from Gumbo data."""

    def process(self, element):
        # Parse the JSON data
        data = json.loads(element)

        # Extract relevant game events (logic specific to Gumbo data format)
        game_events = []
        for play in data.get("plays", []):
            # Example logic: identify home runs
            if play.get("type") == "HOME_RUN":
                game_events.append(
                    {
                        "inning": play.get("inning"),
                        "batter": play.get("batter", {}).get("name"),
                        "team": play.get("batting_team", {}).get("abbreviation"),
                    }
                )

        yield game_events


class EnrichEventData(DoFn):
    """Enriches game events with player and team information (potentially using a lookup)."""

    def process(self, element):
        # Implement logic to enrich event data with additional information
        # (e.g., using a lookup table for player stats or team logos)

        # Example placeholder (replace with your enrichment logic)
        enriched_events = element

        yield enriched_events


class GroupByGameId(DoFn):
    """Groups events by game ID for efficient processing."""

    def process(self, element):
        game_id = element.get("game_id")
        events = element.get("events", [])
        yield (game_id, events)


def combine_events(events):
    """Combines events for a game into a single list."""
    return list(events)


def run_dataflow_pipeline():
    """Creates a Dataflow pipeline to process Gumbo data and extract game highlights."""

    pipeline_options = PipelineOptions(
        runner="DataflowRunner",
        project=project_id,
        region="your-region",  # Specify your preferred region
    )

    with beam.Pipeline(options=pipeline_options) as pipeline:
        # Read raw Gumbo data from Cloud Storage
        raw_data = pipeline | "ReadGumboData" >> ReadFromText(
            known_args={"path": f"gs://{raw_bucket}/gumbo/raw/*.json"}
        )

        # Extract game events
        game_events = raw_data | "ExtractGameEvents" >> ParDo(ExtractGameEvents())

        # Enrich event data (optional)
        # enriched_events = game_events | "EnrichEventData" >> ParDo(EnrichEventData())

        # Group events by game ID
        grouped_events = game_events | "GroupByGameId" >> ParDo(GroupByGameId())

        # Combine events per game (reduce to a single list)
        processed_data = (
            grouped_events
            | "CombineEvents" >> combiners.CombinePerKey(combine_events)
        )

        # Write processed data to Cloud Storage
        processed_data | "WriteProcessedData" >> WriteToText(
            known_args={"path": f"gs://{processed_bucket}/gumbo/processed/*.json"}
        )


if __name__ == "__main__":
    run_dataflow_pipeline()