# Initialize Vertex AI
import vertexai
import time
from vertexai.preview.vision_models import ImageGenerationModel, GeneratedImage
from google.cloud import logging as cloud_logging
import logging
import os
from typing import List

PROJECT_ID = "gem-rush-007"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Configure cloud logging
logging.basicConfig(level=logging.INFO)
log_client = cloud_logging.Client()
log_client.setup_logging()

# Initialize models
imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
fast_imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")

def generate_image(prompt: str, model=None):
    """
    Generate an image based on the provided prompt.
    
    Args:
        prompt: Text prompt for image generation
        model: Model to use (defaults to fast_imagen_model if None)
        
    Returns:
        Generated image or None if generation failed
    """
    if model is None:
        model = fast_imagen_model
        
    logging.info(f"Generating image for prompt: '{prompt}'")
    try:
        response = model.generate_images(prompt=prompt)
        
        if not response.images or len(response.images) == 0:
            logging.error(f"No images were generated for prompt: '{prompt}'")
            return None
            
        return response.images[0]
    except Exception as e:
        logging.error(f"Error generating image: {str(e)}")
        return None

def save_image(image, filename: str):
    """
    Save the generated image to a file.
    
    Args:
        image: The image to save
        filename: The filename to save the image to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        image.save(filename)
        logging.info(f"Image saved successfully to {filename}")
        return True
    except Exception as e:
        logging.error(f"Error saving image to {filename}: {str(e)}")
        return False

def process_prompts(prompts: List[str], output_dir: str = ".", delay_seconds: int = 120):
    """
    Process a list of prompts, generate images, and save them with delays between generations.
    
    Args:
        prompts: List of text prompts
        output_dir: Directory to save images to
        delay_seconds: Time to wait between prompt processing
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for i, prompt in enumerate(prompts):
        prompt_num = i + 1
        filename = os.path.join(output_dir, f"image_{prompt_num}.png")
        
        logging.info(f"Processing prompt {prompt_num}/{len(prompts)}: '{prompt}'")
        
        # Generate the image
        image = generate_image(prompt)
        
        if image:
            # Save the image
            success = save_image(image, filename)
            if success:
                logging.info(f"Successfully processed prompt {prompt_num}/{len(prompts)}")
            else:
                logging.warning(f"Failed to save image for prompt {prompt_num}/{len(prompts)}")
        else:
            logging.warning(f"Failed to generate image for prompt {prompt_num}/{len(prompts)}")
        
        # Don't sleep after the last prompt
        if i < len(prompts) - 1:
            logging.info(f"Waiting {delay_seconds} seconds before processing the next prompt...")
            time.sleep(delay_seconds)
        else:
            logging.info("All prompts processed.")

if __name__ == '__main__':
    # Example prompts
    my_prompts = [
        "A batter at home plate, in a Dodgers uniform, digging into the batter's box. The pitcher is on the mound, ready to deliver. Tense atmosphere. Painting.",
        "A wide shot from space, the baseball burning as it reenters atmosphere, the curve of the earth on the bottom and the endless black space in front. Cinematic, painting",
        "A black and white photograph of a lone wolf howling at the moon.",
        "A surreal dreamscape with floating islands and a giant clock."
    ]
    
    # Process the prompts
    process_prompts(
        prompts=my_prompts,
        output_dir="generated_images",  # Images will be saved in a directory called "generated_images"
        delay_seconds=120  # Wait 2 minutes between prompts
    )