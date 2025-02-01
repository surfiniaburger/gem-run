# Initialize Vertex AI
import vertexai
import time
from vertexai.preview.vision_models import ImageGenerationModel, GeneratedImage
from google.cloud import logging as cloud_logging
import logging

PROJECT_ID = "gem-rush-007"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Configure cloud logging at the top of the script, before other imports
logging.basicConfig(level=logging.INFO)
log_client = cloud_logging.Client()
log_client.setup_logging()

imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
fast_imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")

image_prompt = "Green neon sign jellyfish photography"

response = fast_imagen_model.generate_images(
    prompt=image_prompt,
)

response.images[0].show()