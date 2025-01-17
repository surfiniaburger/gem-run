from google.cloud import storage
from google.oauth2 import service_account
from datetime import timedelta
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel, Image
from pymongo import MongoClient
import os

class HeadshotHandler:
    def __init__(self, mongo_uri: str, key_file_path: str):
        """
        Initialize headshot handler with MongoDB and GCS credentials.
        
        Args:
            mongo_uri: MongoDB connection URI with Vector Search enabled
            key_file_path: Path to GCS service account key file
        """
        # Initialize GCS client
        self.credentials = service_account.Credentials.from_service_account_file(
            key_file_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        self.storage_client = storage.Client(
            project="gem-rush-007",
            credentials=self.credentials
        )
        self.headshots_bucket = "mlb-headshots"
        
        # Initialize MongoDB client with Vector Search
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client.mlb_database
        self.headshots_collection = self.db.headshots
        
        # Initialize Vertex AI with multimodal embedding model
        vertexai.init(project="gem-rush-007", credentials=self.credentials)
        self.embedding_model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

    def parse_player_info(self, file_name: str) -> dict:
        """
        Parse player information from file name.
        Format: PlayerName_Team_Year.jpg (e.g., A.J._Minter_ATL_2024.jpg)
        
        Args:
            file_name: Name of the headshot file
            
        Returns:
            dict: Player information including name, team, and year
        """
        # Remove .jpg extension and split
        parts = file_name.replace('.jpg', '').split('_')
        
        # Extract year and team from the end
        year = parts[-1]
        team = parts[-2]
        
        # Remaining parts make up the player name
        player_name = ' '.join(parts[:-2])
        
        return {
            "player_name": player_name,
            "team": team,
            "year": year
        }

    def get_headshot_url(self, file_name: str) -> str:
        """
        Generate a signed URL for a headshot image in GCS.
        
        Args:
            file_name: Name of the file in GCS (e.g., A.J._Minter_ATL_2024.jpg)
            
        Returns:
            str: Signed URL for accessing the image
        """
        try:
            bucket = self.storage_client.bucket(self.headshots_bucket)
            blob = bucket.blob(file_name)
            
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=30),
                method="GET",
                credentials=self.credentials
            )
            return url
        except Exception as e:
            raise Exception(f"Error generating signed URL: {str(e)}")

    def generate_embedding(self, image_path: str) -> list:
        """
        Generate embedding for an image using Vertex AI Multimodal Embedding Model.
        
        Args:
            image_path: GCS path to the image
            
        Returns:
            list: Image embedding vector
        """
        try:
            # Load image from GCS path
            image = Image.load_from_file(image_path)
            
            # Generate embedding with dimension 1408 (default)
            embeddings = self.embedding_model.get_embeddings(
                image=image,
                dimension=1408
            )
            
            return embeddings.image_embedding
        except Exception as e:
            raise Exception(f"Error generating embedding: {str(e)}")

    def store_headshot_embedding(self, file_name: str):
        """
        Generate and store embedding for a player's headshot.
        
        Args:
            file_name: Name of the headshot file
        """
        try:
            # Parse player info from filename
            player_info = self.parse_player_info(file_name)
            
            # Generate embedding from GCS path
            gcs_path = f"gs://{self.headshots_bucket}/{file_name}"
            embedding = self.generate_embedding(gcs_path)
            
            # Store in MongoDB with vector search index
            self.headshots_collection.update_one(
                {"file_name": file_name},
                {
                    "$set": {
                        **player_info,
                        "embedding": embedding
                    }
                },
                upsert=True
            )
        except Exception as e:
            raise Exception(f"Error storing headshot embedding: {str(e)}")

    def find_player_headshot(self, player_name: str, team: str = None, year: str = "2024") -> dict:
        """
        Find a player's headshot and generate signed URL.
        
        Args:
            player_name: Player's name (e.g., "A.J. Minter")
            team: Optional team filter (e.g., "ATL")
            year: Year of the headshot (default: 2024)
            
        Returns:
            dict: Player information including signed URL
        """
        query = {
            "player_name": {"$regex": f"^{player_name}", "$options": "i"},
            "year": year
        }
        
        if team:
            query["team"] = team
            
        result = self.headshots_collection.find_one(query)
        if result:
            result["signed_url"] = self.get_headshot_url(result["file_name"])
        return result

    def find_similar_players(self, file_name: str, limit: int = 5) -> list:
        """
        Find similar players based on headshot appearance.
        
        Args:
            file_name: Headshot file name
            limit: Maximum number of similar players to return
            
        Returns:
            list: Similar players with their headshot URLs
        """
        try:
            # Get player's embedding
            player = self.headshots_collection.find_one({"file_name": file_name})
            if not player:
                raise Exception(f"No headshot found for file {file_name}")

            # Perform vector similarity search
            similar = self.headshots_collection.aggregate([
                {
                    "$search": {
                        "index": "headshot_vector_index",
                        "knnBeta": {
                            "vector": player["embedding"],
                            "path": "embedding",
                            "k": limit + 1
                        }
                    }
                },
                {
                    "$match": {
                        "file_name": {"$ne": file_name}
                    }
                },
                {
                    "$limit": limit
                }
            ])

            # Add signed URLs to results
            results = []
            for doc in similar:
                doc["signed_url"] = self.get_headshot_url(doc["file_name"])
                results.append(doc)

            return results
        except Exception as e:
            raise Exception(f"Error finding similar players: {str(e)}")

    def batch_process_headshots(self):
        """
        Process all headshots in the GCS bucket and generate embeddings.
        """
        try:
            bucket = self.storage_client.bucket(self.headshots_bucket)
            
            for blob in bucket.list_blobs():
                if not blob.name.endswith('.jpg'):
                    continue
                    
                try:
                    self.store_headshot_embedding(blob.name)
                except Exception as e:
                    print(f"Error processing {blob.name}: {str(e)}")
                    continue
                    
        except Exception as e:
            raise Exception(f"Error in batch processing: {str(e)}")