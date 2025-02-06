import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud import secretmanager
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

def setup_logging():
    """
    Set up Google Cloud Logging
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Instantiate the Cloud Logging client
    client = google.cloud.logging.Client()
    
    # Configure the Cloud Logging handler
    handler = CloudLoggingHandler(client)
    
    # Set up the logger
    logger = logging.getLogger('mongodb_connection')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    return logger

def get_secret(project_id, secret_id, version_id="latest", logger=None):
    """
    Retrieve secret from Google Cloud Secret Manager.
    
    Args:
        project_id (str): Google Cloud project ID
        secret_id (str): Name of the secret
        version_id (str): Version of the secret (default: "latest")
        logger (logging.Logger): Logger instance
    
    Returns:
        str: Secret value
    """
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        
        if logger:
            logger.info(f"Attempting to access secret: {secret_id}")
            
        response = client.access_secret_version(request={"name": name})
        
        if logger:
            logger.info(f"Successfully retrieved secret: {secret_id}")
            
        return response.payload.data.decode("UTF-8")
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to retrieve secret {secret_id}: {str(e)}")
        raise

def connect_to_mongodb():
    """
    Connect to MongoDB using URI from Secret Manager
    
    Returns:
        MongoClient: MongoDB client instance
    """
    # Set up logging
    logger = setup_logging()
    
    try:
        # Replace with your Google Cloud project ID
        project_id = "gem-rush-007"
        # Replace with your secret name in Secret Manager
        secret_id = "mongodb-uri"
        
        logger.info("Starting MongoDB connection process")
        
        # Get MongoDB URI from Secret Manager
        uri = get_secret(project_id, secret_id, logger=logger)
        
        logger.info("Retrieved MongoDB URI from Secret Manager")
        
        # Create MongoDB client
        client = MongoClient(uri, server_api=ServerApi('1'))
        
        # Test connection
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB!")

        
        return client
        
    except Exception as e:
        error_message = f"Error connecting to MongoDB: {str(e)}"
        logger.error(error_message, exc_info=True)
        raise

if __name__ == "__main__":
    try:
        # Connect to MongoDB
        client = connect_to_mongodb()
    except Exception as e:
        print(f"Failed to connect: {e}")