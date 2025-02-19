# Initialize Vertex AI
import vertexai
import time
from vertexai.preview.vision_models import ImageGenerationModel, GeneratedImage
from google.cloud import logging as cloud_logging
import logging
import tempfile
import os

PROJECT_ID = "gem-rush-007"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Configure cloud logging at the top of the script, before other imports
logging.basicConfig(level=logging.INFO)
log_client = cloud_logging.Client()
log_client.setup_logging()

imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
fast_imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")

# List of prompts
prompts = [
    "A photorealistic image of a cat riding a unicorn through a rainbow.",
    "An oil painting of a bustling city street in the style of Van Gogh."
]

# Function to generate and save image with a delay.  Includes error handling.
def generate_and_save_image(prompt, model, delay_seconds=20):
    try:
        response = model.generate_images(prompt=prompt)
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


# Iterate through the prompts
for i, prompt in enumerate(prompts):
    print(f"Processing prompt {i + 1}: {prompt}")
    image_data = generate_and_save_image(prompt, fast_imagen_model)  # Use fast model.
    if image_data: #check that we have valid result.
        # Do something with the image data (optional). You could save it using different logic here, too.
        # This part is just an example, you could store the image_data in a list, etc.
        with open(f"image_{i+1}.png", "wb") as outfile:  # Save to a numbered file.
            outfile.write(image_data)
        print(f"Image data for prompt '{prompt}' saved to image_{i+1}.png")