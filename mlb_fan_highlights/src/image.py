# Initialize Vertex AI
import vertexai
import time
from vertexai.preview.vision_models import ImageGenerationModel, GeneratedImage
from google.cloud import logging as cloud_logging
import logging
import tempfile
import os
from typing import List, Optional

PROJECT_ID = "gem-rush-007"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Configure cloud logging
logging.basicConfig(level=logging.INFO)
log_client = cloud_logging.Client()
log_client.setup_logging()

imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
fast_imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")

# List of prompts
prompts = [
    "A photorealistic image of a cat riding a unicorn through a rainbow.",
    "A futuristic cityscape with flying cars and neon lights at night.",
    "A serene landscape with a snow-capped mountain and a crystal-clear lake.",
    "A close-up portrait of a majestic lion with golden eyes.",
    "A watercolor painting of a field of blooming sunflowers.",
    "A digital art illustration of a dragon breathing fire.",
    "A black and white photograph of a lone wolf howling at the moon.",
    "A surreal dreamscape with floating islands and a giant clock.",
    "A hyperrealistic drawing of a drop of water splashing on a leaf.",
    "An abstract painting with bold colors and geometric shapes.",
    "A cartoon-style image of a group of playful penguins on an iceberg.",
    "A photorealistic image of a vintage car parked on a cobblestone street.",
    "An impressionistic painting of a ballet dancer in mid-leap.",
    "An oil painting of a bustling city street in the style of Van Gogh.",
]

# Function to generate and save image with a delay.  Includes error handling.
def generate_and_save_image(prompt, delay_seconds=120):
    try:
        response = imagen_model.generate_images(prompt=prompt)

        # *** IMPORTANT CHECK ***
        if not response.images:  # Check if the list is empty
            print(f"No images generated for prompt '{prompt}'.")
            return None  # Or handle the error as appropriate for your use case

        image = response.images[0]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
             temp_filename = f.name
             image.save(temp_filename)
             # the image is now saved into the temporary file.
             print(f"Image for prompt '{prompt}' saved to temporary file: {temp_filename}")
             f.seek(0) #important to seek to the beginning of the file before read.
             image_data = f.read() # Read the image data.
        os.remove(temp_filename) #Remove the temp file
        print(f"Temporary file {temp_filename} removed.")

        print(f"Waiting {delay_seconds} seconds before next prompt...")
        time.sleep(delay_seconds)
        return image_data # return the image data.

    except Exception as e:
        print(f"Error generating or saving image for prompt '{prompt}': {e}")
        return None # Return None on error.


def process_prompts_and_generate_images(prompts: List[str]) -> None: #Removed generate_and_save_image_func
    """
    Processes a list of prompts, generates images for each, and saves them.

    Args:
        prompts: A list of strings, where each string is a prompt for image generation.

    Returns:
        None.  Images are saved as files.  Prints status messages.
    """

    #Removed the function check.
    if not isinstance(prompts, list):
        raise TypeError("prompts must be a list")
    if not all(isinstance(prompt, str) for prompt in prompts):
        raise TypeError("All elements in prompts must be strings")

    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i + 1}: {prompt}")

        try:
            image_data = generate_and_save_image(prompt)  # Calls the local generate_and_save_image
        except Exception as e:
            print(f"Error generating image for prompt '{prompt}': {e}")
            continue  # Skip to the next prompt if there's an error

        if image_data:  # Check that we have valid result.
            # Save the image to a file
            try:
                with open(f"image_{i + 1}.png", "wb") as outfile:  # Save to a numbered file.
                    outfile.write(image_data)
                print(f"Image data for prompt '{prompt}' saved to image_{i+1}.png")
            except Exception as e:
                print(f"Error saving image for prompt '{prompt}': {e}")
        else:
            print(f"No image data returned for prompt '{prompt}'.")


# --- Example Usage (and dummy image generation function) ---
if __name__ == '__main__':
    #Example prompts
    my_prompts = [
        "Close-up on the pitcher, face contorted with effort, in the middle of his wind-up. His muscles are tense, and sweat is visible on his brow. Slight motion blur on the arm and ball. Dramatic lighting, focusing on the pitcher's intensity. Painting.",
        "A wide shot from space, the baseball burning as it reenters atmosphere, the curve of the earth on the bottom and the endless black space in front. Cinematic, painting",
        
        
    ]
    process_prompts_and_generate_images(my_prompts) #Removed generate_and_save_image