from google.cloud import bigquery
from google.api_core import retry
import datetime
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, List
import time
from google.api_core.exceptions import RetryError, ServerError, BadRequest, NotFound

class MetricsStorage:
    def __init__(self, project_id: str, dataset_id: str, max_retries: int = 3):
        """Initialize BigQuery client and table references."""
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.dataset_ref = f"{project_id}.{dataset_id}"
        self.summary_table_id = f"{project_id}.{dataset_id}.evaluation_summary_metrics"
        self.detailed_table_id = f"{project_id}.{dataset_id}.evaluation_detailed_metrics"
        self.max_retries = max_retries
        self._ensure_dataset_exists()
        self._ensure_tables_exist()

    def _ensure_dataset_exists(self):
        """Create dataset if it doesn't exist."""
        try:
            self.client.get_dataset(self.dataset_ref)
            print(f"Dataset {self.dataset_ref} already exists")
        except NotFound:
            dataset = bigquery.Dataset(self.dataset_ref)
            dataset.location = "us-central1"
            dataset = self.client.create_dataset(dataset)
            print(f"Created dataset {self.dataset_ref}")

    def _ensure_tables_exist(self):
        """Create tables if they don't exist."""
        # Summary metrics table schema
        summary_schema = [
            bigquery.SchemaField("experiment_run_name", "STRING"),
            bigquery.SchemaField("experiment_name", "STRING"),
            bigquery.SchemaField("evaluation_type", "STRING"),
            bigquery.SchemaField("metric_name", "STRING"),
            bigquery.SchemaField("metric_value", "FLOAT"),
            bigquery.SchemaField("timestamp", "TIMESTAMP")
        ]

        # Detailed metrics table schema with string type for metric_value
        detailed_schema = [
            bigquery.SchemaField("experiment_run_name", "STRING"),
            bigquery.SchemaField("experiment_name", "STRING"),
            bigquery.SchemaField("evaluation_type", "STRING"),
            bigquery.SchemaField("prompt", "STRING"),
            bigquery.SchemaField("response", "STRING"),
            bigquery.SchemaField("predicted_trajectory", "STRING"), # Store as STRING
            bigquery.SchemaField("reference_trajectory", "STRING"), # Store as STRING
            bigquery.SchemaField("intermediate_steps", "STRING"), # Store as STRING
            bigquery.SchemaField("metric_name", "STRING"),
            bigquery.SchemaField("metric_value", "STRING"),  # Changed to STRING to handle all types
            bigquery.SchemaField("timestamp", "TIMESTAMP")
        ]

        # Create tables if they don't exist
        for table_id, schema in [(self.summary_table_id, summary_schema),
                               (self.detailed_table_id, detailed_schema)]:
            try:
                self.client.get_table(table_id)
                print(f"Table {table_id} already exists")
            except NotFound:
                table = bigquery.Table(table_id, schema=schema)
                table.streaming_buffer_max_bytes = 10 * 1024 * 1024  # 10MB
                self.client.create_table(table)
                print(f"Created table {table_id}")

    def _check_existing_run(self, experiment_run_name: str) -> bool:
        """Check if metrics for this run already exist."""
        query = f"""
        SELECT COUNT(*) as count
        FROM `{self.summary_table_id}`
        WHERE experiment_run_name = @run_name
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("run_name", "STRING", experiment_run_name)
            ]
        )
        results = self.client.query(query, job_config=job_config).result()
        return next(results).count > 0

    def _is_valid_numeric(self, value: Any) -> bool:
        """Check if a value is a valid numeric value for BigQuery."""
        try:
            if pd.isna(value) or np.isnan(value):
                return False
            float_val = float(value)
            return not np.isinf(float_val)
        except (ValueError, TypeError):
            return False
        
    @retry.Retry(predicate=retry.if_exception_type(ServerError))
    def _stream_insert_with_retry(self, table_id: str, rows: List[Dict]) -> List[Dict]:
        """Perform streaming insert with retry logic."""
        return self.client.insert_rows_json(table_id, rows)

    def _batch_streaming_insert(self, table_id: str, rows: List[Dict], batch_size: int = 500) -> bool:
        """Insert rows in batches with proper error handling."""
        success = True
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    errors = self._stream_insert_with_retry(table_id, batch)
                    if not errors:
                        break
                    retry_count += 1
                    if retry_count < self.max_retries:
                        wait_time = 2 ** retry_count
                        st.warning(f"Retry {retry_count} after {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        st.error(f"Failed to insert batch after {self.max_retries} retries. Errors: {errors}")
                        success = False
                except BadRequest as e:
                    st.error(f"Invalid data format: {e}")
                    success = False
                    break
                except Exception as e:
                    st.error(f"Unexpected error during streaming insert: {e}")
                    success = False
                    break
        return success

    def save_evaluation_results(self, 
                              eval_result: Any,
                              experiment_name: str,
                              evaluation_type: str,
                              experiment_run_name: str) -> bool:
        """Save evaluation results to BigQuery using streaming inserts."""
        # Check if results for this run already exist
        if self._check_existing_run(experiment_run_name):
            st.info(f"Results for run {experiment_run_name} already exist in BigQuery")
            return True

        timestamp = datetime.datetime.utcnow()

        # Prepare summary metrics - these should all be numeric
        summary_rows = []
        for metric_name, metric_value in eval_result.summary_metrics.items():
            try:
                float_value = float(metric_value)
                summary_rows.append({
                    "experiment_run_name": experiment_run_name,
                    "experiment_name": experiment_name,
                    "evaluation_type": evaluation_type,
                    "metric_name": metric_name,
                    "metric_value": float_value,
                    "timestamp": timestamp.isoformat()
                })
            except (ValueError, TypeError) as e:
                st.warning(f"Skipping non-numeric summary metric {metric_name}: {metric_value}")
                continue

        # Prepare detailed metrics - handle both numeric and non-numeric values
        detailed_rows = []
        for _, row in eval_result.metrics_table.iterrows():
            for metric_name in eval_result.metrics_table.columns:
                if metric_name not in ['prompt', 'response']:
                    metric_value = row[metric_name]
                    # For detailed metrics, convert everything to string but handle NaN
                    if pd.isna(metric_value) or np.isnan(metric_value) if isinstance(metric_value, (float, np.floating)) else False:
                        metric_value = "N/A"
                    detailed_rows.append({
                        "experiment_run_name": experiment_run_name,
                        "experiment_name": experiment_name,
                        "evaluation_type": evaluation_type,
                        "prompt": str(row.get('prompt', '')),
                        "response": str(row.get('response', '')),
                        "predicted_trajectory": str(row.get("predicted_trajectory", '')),
                        "reference_trajectory": str(row.get("reference_trajectory",'')),
                        "intermediate_steps": str(row.get("intermediate_steps", '')),
                        "metric_name": metric_name,
                        "metric_value": str(metric_value),  # Store all values as strings
                        "timestamp": timestamp.isoformat()
                    })

        # Insert rows with batching and error handling
        summary_success = True
        if summary_rows:
            summary_success = self._batch_streaming_insert(self.summary_table_id, summary_rows)
        
        detailed_success = True
        if detailed_rows:
            detailed_success = self._batch_streaming_insert(self.detailed_table_id, detailed_rows)

        if summary_success and detailed_success:
            st.success("Successfully saved new metrics to BigQuery")
            return True
        else:
            st.warning("Some metrics may not have been saved successfully")
            return False