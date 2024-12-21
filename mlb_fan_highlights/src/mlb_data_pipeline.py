from google.cloud import pubsub_v1, bigquery, storage
from google.cloud import aiplatform
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import json
import pandas as pd
from datetime import datetime
from historical_games import fetch_historical_games

class MLBDataPipeline:
    def __init__(self, project_id):
        self.project_id = project_id
        self.bq_client = bigquery.Client(project=project_id)
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
        
    def setup_infrastructure(self):
        """Setup required BigQuery tables and Pub/Sub topics"""
        
        # Create BigQuery dataset and tables
        dataset_ref = self.bq_client.dataset('mlb_analytics')
        
        # Schema for live game data
        game_schema = [
            bigquery.SchemaField("game_id", "STRING"),
            bigquery.SchemaField("game_date", "DATETIME"),
            bigquery.SchemaField("home_team", "STRING"),
            bigquery.SchemaField("away_team", "STRING"),
            bigquery.SchemaField("current_state", "STRING"),
            bigquery.SchemaField("inning", "INTEGER"),
            bigquery.SchemaField("score_home", "INTEGER"),
            bigquery.SchemaField("score_away", "INTEGER"),
            bigquery.SchemaField("timestamp", "TIMESTAMP")
        ]
        
        # Create tables
        table_ref = dataset_ref.table('live_games')
        table = bigquery.Table(table_ref, schema=game_schema)
        self.bq_client.create_table(table, exists_ok=True)

    def create_streaming_pipeline(self):
        """Create real-time data processing pipeline"""
        
        pipeline_options = PipelineOptions([
            '--streaming',
            '--project', self.project_id,
            '--region', 'us-central1',
            '--runner', 'DataflowRunner'
        ])

        def process_game_data(element):
            """Process individual game events"""
            game_data = json.loads(element.decode('utf-8'))
            
            # Extract relevant fields
            processed_data = {
                'game_id': game_data.get('gamePk'),
                'game_date': game_data.get('gameDate'),
                'home_team': game_data.get('teams', {}).get('home', {}).get('team', {}).get('name'),
                'away_team': game_data.get('teams', {}).get('away', {}).get('team', {}).get('name'),
                'current_state': game_data.get('status', {}).get('detailedState'),
                'timestamp': datetime.utcnow().isoformat()
            }
            return processed_data

        def enrich_game_data(element, context):
            """Enrich game data with additional statistics"""
            # Add historical context and statistics
            return element

        # Define the pipeline
        with beam.Pipeline(options=pipeline_options) as pipeline:
            # Read from Pub/Sub
            games = (pipeline 
                    | 'Read from PubSub' >> beam.io.ReadFromPubSub(
                        subscription=f'projects/{self.project_id}/subscriptions/mlb-games-sub'
                    )
                    | 'Process Games' >> beam.Map(process_game_data)
                    | 'Enrich Data' >> beam.Map(enrich_game_data)
                    | 'Write to BigQuery' >> beam.io.WriteToBigQuery(
                        f'{self.project_id}:mlb_analytics.live_games',
                        schema='game_id:STRING,game_date:DATETIME,home_team:STRING,away_team:STRING,current_state:STRING,timestamp:TIMESTAMP',
                        write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                        create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
                    ))

    def historical_data_processor(self):
         """Process historical game data"""
         historical_data = fetch_historical_games(start_year=2015)
    
         # Configure the load job
         job_config = bigquery.LoadJobConfig(
             schema=[
                 bigquery.SchemaField("game_id", "STRING"),
                 bigquery.SchemaField("game_date", "TIMESTAMP"),
                 bigquery.SchemaField("season", "INTEGER"),
                 bigquery.SchemaField("teams", "RECORD", mode="REQUIRED", fields=[
                    bigquery.SchemaField("home", "RECORD", fields=[
                        bigquery.SchemaField("team_id", "INTEGER"),
                        bigquery.SchemaField("team_name", "STRING"),
                        bigquery.SchemaField("score", "INTEGER")
                  ]),
                    bigquery.SchemaField("away", "RECORD", fields=[
                        bigquery.SchemaField("team_id", "INTEGER"),
                        bigquery.SchemaField("team_name", "STRING"),
                        bigquery.SchemaField("score", "INTEGER")
                    ])
                 ]),
                 bigquery.SchemaField("venue", "STRING"),
                 bigquery.SchemaField("status", "STRING"),
                 bigquery.SchemaField("game_type", "STRING"),
                 bigquery.SchemaField("season_display", "STRING"),
                 bigquery.SchemaField("run_differential", "INTEGER"),
                 bigquery.SchemaField("home_team_result", "STRING"),
                 bigquery.SchemaField("batting_stats", "RECORD", fields=[
                     bigquery.SchemaField("home", "RECORD", fields=[
                         bigquery.SchemaField("avg", "FLOAT"),
                         bigquery.SchemaField("atBats", "INTEGER"),
                         bigquery.SchemaField("hits", "INTEGER"),
                         bigquery.SchemaField("doubles", "INTEGER"),
                         bigquery.SchemaField("triples", "INTEGER"),
                         bigquery.SchemaField("homeRuns", "INTEGER"),
                         bigquery.SchemaField("rbi", "INTEGER"),
                         bigquery.SchemaField("baseOnBalls", "INTEGER"),
                        bigquery.SchemaField("strikeOuts", "INTEGER")
                     ]),
                     bigquery.SchemaField("away", "RECORD", fields=[
                         bigquery.SchemaField("avg", "FLOAT"),
                         bigquery.SchemaField("atBats", "INTEGER"),
                         bigquery.SchemaField("hits", "INTEGER"),
                         bigquery.SchemaField("doubles", "INTEGER"),
                         bigquery.SchemaField("triples", "INTEGER"),
                         bigquery.SchemaField("homeRuns", "INTEGER"),
                         bigquery.SchemaField("rbi", "INTEGER"),
                         bigquery.SchemaField("baseOnBalls", "INTEGER"),
                         bigquery.SchemaField("strikeOuts", "INTEGER")
                    ])
                 ]),
                 bigquery.SchemaField("pitching_stats", "RECORD", fields=[
                     bigquery.SchemaField("home", "RECORD", fields=[
                         bigquery.SchemaField("era", "FLOAT"),
                         bigquery.SchemaField("inningsPitched", "FLOAT"),
                         bigquery.SchemaField("hits", "INTEGER"),
                         bigquery.SchemaField("runs", "INTEGER"),
                         bigquery.SchemaField("earnedRuns", "INTEGER"),
                         bigquery.SchemaField("baseOnBalls", "INTEGER"),
                         bigquery.SchemaField("strikeOuts", "INTEGER"),
                         bigquery.SchemaField("homeRuns", "INTEGER")
                    ]),
                    bigquery.SchemaField("away", "RECORD", fields=[
                        bigquery.SchemaField("era", "FLOAT"),
                        bigquery.SchemaField("inningsPitched", "FLOAT"),
                        bigquery.SchemaField("hits", "INTEGER"),
                        bigquery.SchemaField("runs", "INTEGER"),
                        bigquery.SchemaField("earnedRuns", "INTEGER"),
                        bigquery.SchemaField("baseOnBalls", "INTEGER"),
                        bigquery.SchemaField("strikeOuts", "INTEGER"),
                        bigquery.SchemaField("homeRuns", "INTEGER")
                    ])
                ]),
                bigquery.SchemaField("batting_comparison", "RECORD", fields=[
                    bigquery.SchemaField("home_ba", "FLOAT"),
                    bigquery.SchemaField("away_ba", "FLOAT"),
                    bigquery.SchemaField("ba_differential", "FLOAT")
                ]),
                # Time-based features
                bigquery.SchemaField("year", "INTEGER"),
                bigquery.SchemaField("month", "INTEGER"),
                bigquery.SchemaField("day_of_week", "STRING"),
                # Rolling statistics
                bigquery.SchemaField("team_rolling_stats", "RECORD", mode="REPEATED", fields=[
                    bigquery.SchemaField("team_id", "INTEGER"),
                    bigquery.SchemaField("rolling_runs_avg", "FLOAT")
                ])
            ],
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            clustering_fields=["season", "game_date"],
            time_partitioning=bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.MONTH,
                field="game_date"
            )
        )
    
        # Load to BigQuery
         table_id = f"{self.project_id}.mlb_analytics.historical_games"
         job = self.bq_client.load_table_from_json(
            historical_data,
            table_id,
            job_config=job_config
        )
    
         job.result()  # Wait for the job to complete
    
         print(f"Loaded {len(historical_data)} games to {table_id}")
         return historical_data


        

    def create_ml_pipeline(self):
        """Create ML pipeline for game predictions"""
        
        # Initialize Vertex AI
        aiplatform.init(project=self.project_id, location='us-central1')
        
        def train_prediction_model():
            # Create dataset
            dataset = aiplatform.TabularDataset.create(
                display_name="game_predictions",
                gcs_source=f"gs://{self.project_id}-mlb/processed/*.csv"
            )
            
            # Train model
            training_job = aiplatform.AutoMLTabularTrainingJob(
                display_name="game_prediction_model",
                optimization_objective="minimize-rmse"
            )
            
            model = training_job.run(
                dataset=dataset,
                target_column="result",
                budget_milli_node_hours=1000
            )
            
            return model

    def setup_monitoring(self):
        """Setup monitoring and alerting"""
        from google.cloud import monitoring_v3
        
        client = monitoring_v3.MetricServiceClient()
        project_name = f"projects/{self.project_id}"
        
        # Create custom metrics
        descriptor = monitoring_v3.MetricDescriptor(
            type_="custom.googleapis.com/mlb/pipeline_latency",
            metric_kind=monitoring_v3.MetricDescriptor.MetricKind.GAUGE,
            value_type=monitoring_v3.MetricDescriptor.ValueType.DOUBLE,
            description="Pipeline processing latency"
        )
        
        client.create_metric_descriptor(
            name=project_name,
            metric_descriptor=descriptor
        )

def main():
    project_id = "gem-creation"
    pipeline = MLBDataPipeline(project_id)
    
    # Setup infrastructure
    pipeline.setup_infrastructure()
    
    # Start real-time pipeline
    pipeline.create_streaming_pipeline()
    
    # Process historical data
    pipeline.historical_data_processor()
    
    # Setup monitoring
    pipeline.setup_monitoring()
    
    # Create and train ML model
    pipeline.create_ml_pipeline()

if __name__ == "__main__":
    main()
