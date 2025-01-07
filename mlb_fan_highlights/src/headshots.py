from google.cloud import bigquery
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignedUrlsBigQueryLoader:
    def __init__(self):
        self.client = bigquery.Client()
        self.dataset_id = "mlb_data"  # Replace with your dataset name
        self.table_id = "signed_urls"  # Table name
        
    def create_table_if_not_exists(self):
        """Create the table if it doesn't exist with the appropriate schema"""
        dataset_ref = self.client.dataset(self.dataset_id)
        table_ref = dataset_ref.table(self.table_id)
        
        schema = [
            bigquery.SchemaField("file_name", "STRING"),
            bigquery.SchemaField("signed_url", "STRING")
        ]
        
        try:
            self.client.get_table(table_ref)
            logger.info("Table already exists")
        except Exception:
            table = bigquery.Table(table_ref, schema=schema)
            self.client.create_table(table)
            logger.info("Created new table")

    def load_data(self, file_path: str):
        """Load data from CSV file into BigQuery"""
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            logger.info(f"Successfully read CSV file with {len(df)} rows")

            # Create full table reference
            table_id = f"{self.client.project}.{self.dataset_id}.{self.table_id}"

            # Configure the load job
            job_config = bigquery.LoadJobConfig(
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
                source_format=bigquery.SourceFormat.CSV,
            )

            # Load the dataframe into BigQuery
            job = self.client.load_table_from_dataframe(
                df, 
                table_id, 
                job_config=job_config
            )
            
            # Wait for the job to complete
            job.result()
            
            # Get table details and log results
            table = self.client.get_table(table_id)
            logger.info(
                f"Loaded {table.num_rows} rows and {len(table.schema)} columns to {table_id}"
            )

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

def main():
    # File path of your CSV
    csv_file_path = "c:/Users/CCL/Desktop/mlb/google-cloud-mlb-hackathon/mlb_fan_highlights/src/signed_urls.csv"
    
    # Initialize and run the loader
    loader = SignedUrlsBigQueryLoader()
    
    try:
        # Create table if it doesn't exist
        loader.create_table_if_not_exists()
        
        # Load the data
        loader.load_data(csv_file_path)
        
        logger.info("Data loading process completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")

if __name__ == "__main__":
    main()

