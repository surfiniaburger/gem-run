from google.cloud import storage
import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel
from langchain_google_firestore import FirestoreVectorStore
from langchain_google_vertexai import VertexAIEmbeddings
import logging
from datetime import datetime, timedelta, UTC
from typing import Optional, List, Dict
from google.cloud import secretmanager
from google.oauth2 import service_account
import json
import tempfile
import os

def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Set up a logger with the specified log level and formatting."""
    logger = logging.getLogger("PlayerImageSimilarity")
    logger.setLevel(getattr(logging, log_level))
    
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger

def parse_player_info(file_name: str, logger: Optional[logging.Logger] = None) -> dict:
    """Parse player information from filename (e.g., A.J._Minter_ATL_2024.jpg)."""
    if logger:
        logger.debug(f"Parsing player info from filename: {file_name}")
    
    try:
        parts = file_name.replace('.jpg', '').split('_')
        year = parts[-1]
        team = parts[-2]
        player_name = ' '.join(parts[:-2])
        
        return {
            "player_name": player_name,
            "team": team,
            "year": year
        }
    except Exception as e:
        if logger:
            logger.error(f"Error parsing filename {file_name}: {str(e)}")
        raise

class PlayerImageSimilarity:
    def __init__(
        self,
        project_id: str,
        bucket_name: str,
        region: str,
        collection_name: str,
        secret_id: str,
        secret_version: str = 'latest',
        log_level: str = "INFO"
    ):
        self.logger = setup_logger(log_level)
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.region = region
        self.collection_name = collection_name
        
        # Initialize with service account credentials
        try:
            # Initialize Secret Manager client
            secret_client = secretmanager.SecretManagerServiceClient()
            
            # Get secret path
            secret_path = secret_client.secret_version_path(
                project_id, secret_id, secret_version
            )
            
            # Access the secret payload
            response = secret_client.access_secret_version(request={"name": secret_path})
            service_account_key = json.loads(response.payload.data.decode('UTF-8'))
            
            # Create credentials
            self.credentials = service_account.Credentials.from_service_account_info(
                service_account_key,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Initialize storage client with credentials
            self.storage_client = storage.Client(
                project=project_id,
                credentials=self.credentials
            )
            self.bucket = self.storage_client.bucket(bucket_name)
        
            # Initialize Vertex AI
            vertexai.init(project=project_id, location=region)
            self.model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
        
            # Initialize embedding service for text
            self.embedding_service = VertexAIEmbeddings(
               model_name="textembedding-gecko@latest",
               project=project_id,
            )
        
            # Initialize Firestore vector store
            self.vector_store = FirestoreVectorStore(
               collection=collection_name,
               embedding_service=self.embedding_service,
            )
        
            self.logger.info("PlayerImageSimilarity initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            raise

    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Set up a logger with the specified log level and formatting."""
        logger = logging.getLogger("PlayerImageSimilarity")
        logger.setLevel(getattr(logging, log_level))
        
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(console_handler)
        
        return logger

    def verify_bucket_access(self) -> bool:
        """Verify bucket access before processing."""
        try:
            self.logger.info(f"Verifying access to bucket: {self.bucket_name}")
            self.bucket.reload()
            self.logger.info(f"Successfully verified access to {self.bucket_name}")
            return True
        except Exception as e:
            self.logger.error(f"Bucket access error: {str(e)}")
            raise

    def generate_signed_url(self, blob_name: str) -> str:
        """Generate a signed URL for accessing a player headshot."""
        blob = self.bucket.blob(blob_name)
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=30),
            method="GET",
            service_account_email=self.credentials.service_account_email,
            credentials=self.credentials
        )
        return url

    async def process_player_images(self) -> Dict[str, int]:
        """Process all player images in the bucket and store their embeddings."""
        processed_count = 0
        error_count = 0
        
        try:
            blobs = self.bucket.list_blobs()
            
            for blob in blobs:
                if not blob.name.endswith('.jpg'):
                    continue
                    
                try:
                    self.logger.info(f"Processing headshot: {blob.name}")
                    
                    # Parse player info
                    player_info = parse_player_info(blob.name, self.logger)
                    
                    # Create GCS URI for the image
                    gcs_uri = f"gs://{self.bucket_name}/{blob.name}"
                    
                    # Download the image to a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                        blob.download_to_filename(temp_file.name)
                        temp_file_path = temp_file.name
                    
                    # Generate embedding using the local file
                    image = Image.load_from_file(temp_file_path)
                    embeddings = self.model.get_embeddings(
                        image=image,
                        contextual_text=f"Player: {player_info['player_name']}, Team: {player_info['team']}, Year: {player_info['year']}",
                        dimension=1408
                    )
                    
                    # Store in Firestore with metadata
                    self.vector_store.add_texts(
                        texts=[str(embeddings.image_embedding)],
                        metadatas=[{
                            **player_info,
                            "file_name": blob.name,
                            "gcs_uri": gcs_uri,
                            "generated_at": datetime.now(UTC)
                        }],
                        ids=[blob.name]
                    )
                   
                    
                    processed_count += 1
                    self.logger.info(f"Successfully processed {blob.name}")
                    
                    # Clean up the temporary file
                    os.remove(temp_file_path)
                    
                except Exception as e:
                    error_count += 1
                    self.logger.error(f"Error processing {blob.name}: {str(e)}")
                    continue
            
            return {
                "processed": processed_count,
                "errors": error_count
            }
            
        except Exception as e:
            self.logger.error(f"Critical error in processing: {str(e)}")
            raise

    async def find_similar_players(self, query_file: str, k: int = 5) -> List[Dict]:
        """Find similar players based on a query image."""
        try:
            self.logger.info(f"Finding similar players for {query_file}")
            
            # Create GCS URI for the query image
            gcs_uri = f"gs://{self.bucket_name}/{query_file}"
            
            # Download the image to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                blob = self.bucket.blob(query_file)
                blob.download_to_filename(temp_file.name)
                temp_file_path = temp_file.name
            

            # Perform similarity search
            results = self.vector_store.similarity_search(
                query=temp_file_path,
                k=k
            )
            
            self.logger.info(f"Found {len(results)} similar players")
            
            # Clean up the temporary file
            os.remove(temp_file_path)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in similarity search: {str(e)}")
            raise

# Example usage:
async def main():
    try:
        similarity_search = PlayerImageSimilarity(
            project_id="gem-rush-007",
            bucket_name="mlb-headshots",
            region="us-central1",
            collection_name="player_embeddings",
            secret_id="cloud-run-invoker"
        )
        # Verify bucket access first
       #similarity_search.verify_bucket_access()

        # Process all images and store embeddings
      # results = await similarity_search.process_player_images()
      # print(f"Processed {results['processed']} images with {results['errors']} errors")
        
        # Find similar players
        similar_players = await similarity_search.find_similar_players("A.J._Puk_ARI_2024.jpg")
        print("Similar players found:", similar_players)
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())