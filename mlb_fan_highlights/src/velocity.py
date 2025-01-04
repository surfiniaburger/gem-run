from google.cloud import bigquery
import pandas as pd
from datetime import datetime, timezone
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Schema definition for exit velocity data
exit_velocity_schema = [
    bigquery.SchemaField("last_name", "STRING"),
    bigquery.SchemaField("first_name", "STRING"),
    bigquery.SchemaField("player_id", "STRING"),
    bigquery.SchemaField("attempts", "INTEGER"),
    bigquery.SchemaField("avg_hit_angle", "FLOAT"),
    bigquery.SchemaField("anglesweetspotpercent", "FLOAT"),
    bigquery.SchemaField("max_hit_speed", "FLOAT"),
    bigquery.SchemaField("avg_hit_speed", "FLOAT"),
    bigquery.SchemaField("ev50", "FLOAT"),
    bigquery.SchemaField("fbld", "FLOAT"),
    bigquery.SchemaField("gb", "FLOAT"),
    bigquery.SchemaField("max_distance", "INTEGER"),
    bigquery.SchemaField("avg_distance", "INTEGER"),
    bigquery.SchemaField("avg_hr_distance", "INTEGER"),
    bigquery.SchemaField("ev95plus", "INTEGER"),
    bigquery.SchemaField("ev95percent", "FLOAT"),
    bigquery.SchemaField("barrels", "INTEGER"),
    bigquery.SchemaField("brl_percent", "FLOAT"),
    bigquery.SchemaField("brl_pa", "FLOAT"),
    bigquery.SchemaField("last_updated", "TIMESTAMP")
]

class ExitVelocityPipeline:
    def __init__(self):
        self.client = bigquery.Client()
        self.dataset_id = "mlb_data"  # Using the same dataset as the main pipeline
        
    def create_exit_velocity_table(self):
        """Create the exit velocity table if it doesn't exist"""
        dataset_ref = self.client.dataset(self.dataset_id)
        table_ref = dataset_ref.table("pitchers_ev")
        
        try:
            self.client.get_table(table_ref)
            logger.info("Exit velocity table already exists")
        except Exception:
            table = bigquery.Table(table_ref, schema=exit_velocity_schema)
            self.client.create_table(table)
            logger.info("Created exit velocity table")

    def load_exit_velocity_data(self, file_path: str, batch_size: int = 1000):
        """
        Load exit velocity statistics from CSV file into BigQuery.
        
        Args:
            file_path (str): Path to the exit velocity CSV file
            batch_size (int): Number of records to insert in each batch
        """
        try:
            # Create the table if it doesn't exist
            self.create_exit_velocity_table()
            
            # Read the CSV file with the correct column name
            df = pd.read_csv(file_path)
            
            # Process the name column - it's the first column in the CSV
            # Split the name into last_name and first_name
            name_series = df.iloc[:, 0].str.extract(r'([^,]+),\s*(.+)')
            df['last_name'] = name_series[0]
            df['first_name'] = name_series[1]
            
            # Drop the original combined name column
            df = df.drop(df.columns[0], axis=1)
            
            # Convert player_id to string explicitly
            df['player_id'] = df['player_id'].astype(str)
            
            # Convert numeric columns
            numeric_columns = {
                'attempts': 'Int64',
                'avg_hit_angle': 'float64',
                'anglesweetspotpercent': 'float64',
                'max_hit_speed': 'float64',
                'avg_hit_speed': 'float64',
                'ev50': 'float64',
                'fbld': 'float64',
                'gb': 'float64',
                'max_distance': 'Int64',
                'avg_distance': 'Int64',
                'avg_hr_distance': 'Int64',
                'ev95plus': 'Int64',
                'ev95percent': 'float64',
                'barrels': 'Int64',
                'brl_percent': 'float64',
                'brl_pa': 'float64'
            }
            
            for col, dtype in numeric_columns.items():
                if col in df.columns:  # Only process if column exists
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if dtype == 'Int64':
                        df[col] = df[col].astype('Int64')

            # Add last_updated timestamp
            df['last_updated'] = datetime.now(timezone.utc)

            # Print DataFrame info for debugging
            logger.info("DataFrame columns and types:")
            logger.info(df.dtypes)

            # Get reference to the table
            table_ref = f"{self.client.project}.{self.dataset_id}.exit_velocity"
            
            # Configure the load job
            job_config = bigquery.LoadJobConfig(
                schema=exit_velocity_schema,
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
            logger.error(f"Error loading exit velocity data: {str(e)}")
            raise

def main():
    pipeline = ExitVelocityPipeline()
    pipeline.load_exit_velocity_data("../../datasets/pitchers_ev.csv")

if __name__ == "__main__":
    main()