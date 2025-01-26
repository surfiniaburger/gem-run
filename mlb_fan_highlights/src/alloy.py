from google.cloud import storage
from google.cloud import aiplatform
import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel
from langchain_google_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore, Column
from datetime import datetime, timedelta
import asyncio
import logging
from typing import Optional
import sys
import os
from google.auth import credentials
from google.auth.credentials import Credentials
from sqlalchemy import text
from google.cloud import secretmanager_v1




def get_secret(project_id: str, secret_id: str, version_id: str = 'latest'):
    """
    Retrieve a secret from Google Secret Manager.
    
    Args:
        project_id: Google Cloud project ID
        secret_id: ID of the secret in Secret Manager
        version_id: Version of the secret (default is 'latest')
    
    Returns:
        str: Decoded secret value
    """
    # Create the secret manager client
    client = secretmanager_v1.SecretManagerServiceClient()
    
    # Construct the resource name of the secret version
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    
    # Access the secret version
    response = client.access_secret_version(request={"name": name})
    
    # Return the decoded secret
    return response.payload.data.decode('UTF-8')


def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """
    Set up a logger with the specified log level and formatting.
    
    Args:
        log_level: Logging level (default: "INFO")
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("PlayerEmbeddings")
    logger.setLevel(getattr(logging, log_level))

    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger if it doesn't already have handlers
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger

def parse_player_info(file_name: str, logger: Optional[logging.Logger] = None) -> dict:
    """
    Parse player information from file name.
    Format: PlayerName_Team_Year.jpg (e.g., A.J._Minter_ATL_2024.jpg)
    """
    if logger:
        logger.debug(f"Parsing player info from filename: {file_name}")
    
    try:
        # Remove .jpg extension and split
        parts = file_name.replace('.jpg', '').split('_')
        
        # Extract year and team from the end
        year = parts[-1]
        team = parts[-2]
        
        # Remaining parts make up the player name
        player_name = ' '.join(parts[:-2])
        
        player_info = {
            "player_name": player_name,
            "team": team,
            "year": year
        }
        
        if logger:
            logger.debug(f"Successfully parsed player info: {player_info}")
        
        return player_info
    except Exception as e:
        if logger:
            logger.error(f"Error parsing filename {file_name}: {str(e)}")
        raise

async def test_database_connection(engine: AlloyDBEngine, logger: logging.Logger) -> bool:
    """Test database connection with proper SQLAlchemy syntax."""
    try:
        async with engine._pool.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            await result.fetchone()
            logger.info("Database connection test successful")
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        return False
    

async def create_player_embeddings_workflow(
    project_id: str,
    bucket_name: str,
    region: str,
    cluster: str,
    instance: str,
    database: str,
    table_name: str = "player_embeddings",
    log_level: str = "INFO",
    db_user: Optional[str] = None,
    db_password: Optional[str] = None
):
    """
    Workflow to generate and store multimodal embeddings for player headshots
    using signed URLs and AlloyDB vector storage.
    Args:
        project_id: GCP project ID
        bucket_name: GCS bucket name
        region: GCP region
        cluster: AlloyDB cluster name
        instance: AlloyDB instance name
        database: Database name
        table_name: Table name for vector store
        log_level: Logging level
        db_user: Database user (optional, defaults to env var ALLOYDB_USER)
        db_password: Database password (optional, defaults to env var ALLOYDB_PASSWORD)    
    """
    # Set up logger
    logger = setup_logger(log_level)
    logger.info("Starting player embeddings workflow")
    
    try:
        # Get database credentials
        db_user = db_user or os.getenv('ALLOYDB_USER')
        db_password = db_password or os.getenv('ALLOYDB_PASSWORD')
        
        if not db_user or not db_password:
            raise ValueError(
                "Database credentials not provided. Set ALLOYDB_USER and ALLOYDB_PASSWORD "
                "environment variables or pass credentials as parameters."
            )
        
        logger.info("Validating database credentials")
        logger.debug(f"Using database user: {db_user}")        
        # Initialize storage client
        logger.info(f"Initializing storage client for bucket: {bucket_name}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        logger.debug("Storage client initialized successfully")

        # Initialize Vertex AI
        logger.info(f"Initializing Vertex AI for project: {project_id}, region: {region}")
        vertexai.init(project=project_id, location=region)
        model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
        logger.debug("Vertex AI initialized successfully")

        # Initialize AlloyDB
        logger.info(f"Connecting to AlloyDB instance: {instance}")
        engine = await AlloyDBEngine.afrom_instance(
            project_id=project_id,
            region=region,
            cluster=cluster,
            instance=instance,
            database=database,
            user=db_user,
            password=db_password
        )
        logger.debug("AlloyDB connection established")

        # Test database connection
        if not await test_database_connection(engine, logger):
            raise Exception("Failed to establish database connection")

        # Initialize vector store table with enhanced schema
        logger.info(f"Initializing vector store table: {table_name}")
        await engine.ainit_vectorstore_table(
            table_name=table_name,
            vector_size=1408,
            metadata_columns=[
                Column("player_name", "VARCHAR(255)"),
                Column("team", "VARCHAR(10)"),
                Column("year", "VARCHAR(4)"),
                Column("file_name", "VARCHAR(255)"),
                Column("signed_url", "TEXT"),
                Column("generated_at", "TIMESTAMP")
            ]
        )
        logger.debug("Vector store table initialized successfully")

        # Initialize vector store
        vector_store = await AlloyDBVectorStore.create(
            engine=engine,
            table_name=table_name,
            embedding_service=None,
        )
        logger.debug("Vector store initialized successfully")

        def generate_signed_url(blob_name: str) -> str:
            """Generate signed URL for a player headshot."""
            logger.debug(f"Generating signed URL for blob: {blob_name}")
            blob = bucket.blob(blob_name)
            url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.utcnow() + timedelta(minutes=30),
                method="GET"
            )
            return url

        embeddings_data = []
        
        # List all headshots in the bucket
        logger.info("Starting to process headshots from bucket")
        blobs = bucket.list_blobs()
        processed_count = 0
        error_count = 0
        
        for blob in blobs:
            if not blob.name.endswith('.jpg'):
                continue
                
            try:
                logger.info(f"Processing headshot: {blob.name}")
                
                # Parse player info from filename
                player_info = parse_player_info(blob.name, logger)
                
                # Generate signed URL
                signed_url = generate_signed_url(blob.name)
                logger.debug(f"Generated signed URL for {blob.name}")
                
                # Generate embedding using signed URL
                logger.debug("Loading image and generating embedding")
                image = Image.load_from_file(signed_url)
                
                # Get embeddings with contextual text
                embeddings = model.get_embeddings(
                    image=image,
                    contextual_text=f"Player: {player_info['player_name']}, Team: {player_info['team']}, Year: {player_info['year']}",
                    dimension=1408
                )
                logger.debug(f"Successfully generated embeddings for {blob.name}")

                # Prepare data for storage
                embeddings_data.append({
                    "vector": embeddings.image_embedding,
                    "metadata": {
                        **player_info,
                        "file_name": blob.name,
                        "signed_url": signed_url,
                        "generated_at": datetime.utcnow().isoformat()
                    }
                })
                processed_count += 1
                logger.info(f"Successfully processed {blob.name}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing {blob.name}: {str(e)}")
                continue

        # Store embeddings in AlloyDB
        if embeddings_data:
            logger.info(f"Storing {len(embeddings_data)} embeddings in AlloyDB")
            await vector_store.aadd_embeddings(
                embeddings=[data["vector"] for data in embeddings_data],
                metadatas=[data["metadata"] for data in embeddings_data]
            )
            logger.debug("Successfully stored embeddings in AlloyDB")

        # Example similarity search function
        async def find_similar_players(query_file: str, k: int = 5):
            logger.info(f"Finding similar players for {query_file}")
            query_url = generate_signed_url(query_file)
            query_image = Image.load_from_file(query_url)
            query_embeddings = model.get_embeddings(
                image=query_image,
                dimension=1408
            )

            results = await vector_store.asimilarity_search_by_vector(
                query_embeddings.image_embedding,
                k=k
            )
            logger.info(f"Found {len(results)} similar players")
            return results

        logger.info(f"Workflow completed. Processed: {processed_count}, Errors: {error_count}")
        return {"vector_store": vector_store, "find_similar": find_similar_players}

    except Exception as e:
        logger.error(f"Critical error in workflow: {str(e)}")
        raise

# Example usage:
async def main():
    try:
        project_id="gem-rush-007"
        db_password = get_secret(project_id, "ALLOYDB_PASSWORD")  
        workflow = await create_player_embeddings_workflow(
            project_id=project_id,
            bucket_name="mlb-headshot",
            region="us-east4",
            cluster="my-cluster",
            instance="my-cluster-primary",
            database="player_headshots",
            log_level="INFO",  # Can be set to "DEBUG" for more detailed logs,
            db_user='postgres',
            db_password=db_password
        )
        
        # Example: Find similar players for a specific headshot
        similar_players = await workflow["find_similar"]("A.J._Minter_ATL_2024.jpg")
        print("Similar players found:", similar_players)
        
    except ValueError as ve:
        logging.error(f"Configuration error: {str(ve)}")
        sys.exit(1)       
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
    