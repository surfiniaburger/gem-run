from google.cloud import bigquery
import pandas as pd
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Updated schema definition
player_stats_schema = [
    bigquery.SchemaField("last_name", "STRING"),
    bigquery.SchemaField("first_name", "STRING"),
    bigquery.SchemaField("player_id", "STRING"),
    bigquery.SchemaField("year", "INTEGER"),
    bigquery.SchemaField("player_age", "INTEGER"),
    bigquery.SchemaField("ab", "INTEGER"),
    bigquery.SchemaField("pa", "INTEGER"),
    bigquery.SchemaField("hit", "INTEGER"),
    bigquery.SchemaField("single", "INTEGER"),
    bigquery.SchemaField("double", "INTEGER"),
    bigquery.SchemaField("triple", "INTEGER"),
    bigquery.SchemaField("home_run", "INTEGER"),
    bigquery.SchemaField("strikeout", "INTEGER"),
    bigquery.SchemaField("walk", "INTEGER"),
    bigquery.SchemaField("k_percent", "FLOAT"),
    bigquery.SchemaField("bb_percent", "FLOAT"),
    bigquery.SchemaField("batting_avg", "STRING"),
    bigquery.SchemaField("slg_percent", "STRING"),
    bigquery.SchemaField("on_base_percent", "STRING"),
    bigquery.SchemaField("on_base_plus_slg", "STRING"),
    bigquery.SchemaField("b_rbi", "INTEGER"),
    bigquery.SchemaField("r_run", "INTEGER"),
    bigquery.SchemaField("b_walkoff", "INTEGER"),
    bigquery.SchemaField("b_reached_on_int", "INTEGER"),
    bigquery.SchemaField("xba", "FLOAT"),
    bigquery.SchemaField("xslg", "FLOAT"),
    bigquery.SchemaField("woba", "FLOAT"),
    bigquery.SchemaField("xwoba", "FLOAT"),
    bigquery.SchemaField("xobp", "FLOAT"),
    bigquery.SchemaField("xiso", "FLOAT"),
    bigquery.SchemaField("wobacon", "FLOAT"),
    bigquery.SchemaField("xwobacon", "FLOAT"),
    bigquery.SchemaField("bacon", "FLOAT"),
    bigquery.SchemaField("xbacon", "FLOAT"),
    bigquery.SchemaField("xbadiff", "FLOAT"),
    bigquery.SchemaField("xslgdiff", "FLOAT"),
    bigquery.SchemaField("avg_swing_speed", "FLOAT"),
    bigquery.SchemaField("fast_swing_rate", "FLOAT"),
    bigquery.SchemaField("blasts_contact", "FLOAT"),
    bigquery.SchemaField("blasts_swing", "FLOAT"),
    bigquery.SchemaField("squared_up_contact", "FLOAT"),
    bigquery.SchemaField("squared_up_swing", "FLOAT"),
    bigquery.SchemaField("avg_swing_length", "STRING"),
    bigquery.SchemaField("swords", "INTEGER"),
    bigquery.SchemaField("exit_velocity_avg", "FLOAT"),
    bigquery.SchemaField("launch_angle_avg", "FLOAT"),
    bigquery.SchemaField("sweet_spot_percent", "FLOAT"),
    bigquery.SchemaField("barrel", "INTEGER"),
    bigquery.SchemaField("barrel_batted_rate", "FLOAT"),
    bigquery.SchemaField("poorlytopped_percent", "FLOAT"),
    bigquery.SchemaField("poorlyweak_percent", "FLOAT"),
    bigquery.SchemaField("hard_hit_percent", "FLOAT"),
    bigquery.SchemaField("avg_best_speed", "FLOAT"),
    bigquery.SchemaField("avg_hyper_speed", "FLOAT"),
    bigquery.SchemaField("edge_percent", "FLOAT"),
    bigquery.SchemaField("whiff_percent", "FLOAT"),
    bigquery.SchemaField("swing_percent", "FLOAT"),
]

class PlayerStatsPipeline:
    def __init__(self):
        self.client = bigquery.Client()
        self.dataset_id = "mlb_data"

    def create_player_stats_table(self):
        """Create the player stats table if it doesn't exist"""
        dataset_ref = self.client.dataset(self.dataset_id)
        table_ref = dataset_ref.table("batters_player_stats")

        try:
            self.client.get_table(table_ref)
            logger.info("Player stats table already exists")
        except Exception:
            table = bigquery.Table(table_ref, schema=player_stats_schema)
            self.client.create_table(table)
            logger.info("Created player stats table")

    def load_player_stats_data(self, file_path: str):
        """
        Load player stats from CSV file into BigQuery.

        Args:
            file_path (str): Path to the player stats CSV file
        """
        try:
            # Create the table if it doesn't exist
            self.create_player_stats_table()

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Split the name into last_name and first_name
            name_series = df.iloc[:, 0].str.extract(r'([^,]+),\s*(.+)')
            df['last_name'] = name_series[0]
            df['first_name'] = name_series[1]

            # Drop the original combined name column
            df = df.drop(df.columns[0], axis=1)

            # Ensure player_id is a string
            df['player_id'] = df['player_id'].astype(str)

            # Convert numeric columns to appropriate types
            for col in df.columns:
                if col not in ['last_name', 'first_name', 'player_id', 'avg_swing_length']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            # Convert specific columns to string format
            string_columns = ['batting_avg', 'slg_percent', 'on_base_percent', 'on_base_plus_slg', 'avg_swing_length' ]
            for col in string_columns:
                df[col] = df[col].astype(str)

            # Add last_updated timestamp
            df['last_updated'] = datetime.now(timezone.utc)

            # Print DataFrame info for debugging
            logger.info("DataFrame columns and types:")
            logger.info(df.dtypes)

            # Get reference to the table
            table_ref = f"{self.client.project}.{self.dataset_id}.player_stats"

            # Configure the load job
            job_config = bigquery.LoadJobConfig(
                schema=player_stats_schema,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
            )

            # Load the data
            job = self.client.load_table_from_dataframe(
                df, table_ref, job_config=job_config
            )

            # Wait for the job to complete
            job.result()

            logger.info(f"Loaded {len(df)} records into {table_ref}")

        except FileNotFoundError:
            logger.error(f"Could not find file at {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading player stats data: {str(e)}")
            raise

def main():
    pipeline = PlayerStatsPipeline()
    pipeline.load_player_stats_data("../../datasets/batters.csv")

if __name__ == "__main__":
    main()
