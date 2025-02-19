import logging
import time
import asyncio
from PIL import Image
from io import BytesIO
import re
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel, Part, Image

# Configure logging
logging.basicConfig(level=logging.INFO)

class MockGCSHandler:
    def __init__(self, project_id):
        self.project_id = project_id

class CloudVideoGenerator:
    def __init__(self, gcs_handler):
        self.gcs_handler = gcs_handler
        self.project_id = gcs_handler.project_id
        self.location = "us-central1"
        # Initialize the Vertex AI client.
        aiplatform.init(project=self.project_id, location=self.location)
        self.model = GenerativeModel("imagegeneration@005")  # Use correct model name.

        self.safety_config = []  # Not needed for isolated tests
        logging.info("CloudVideoGenerator initialized successfully.")

    async def _generate_images(self, analysis: dict) -> list:
        """Generates images asynchronously for all prompts in the analysis."""
        logging.info("Starting image generation for key moments.")
        prompts = [self._enhance_prompt(moment['visual_prompt']) for moment in analysis['key_moments']]
        images = await self._generate_images_from_prompts(prompts)  # Use the new method
        logging.info("Completed image generation for all key moments.")
        return images

    async def _generate_images_from_prompts(self, prompts: list[str]) -> list[bytes]:
        """Generates images from a list of prompts, handling retries and defaults."""
        tasks = [self._generate_single_image_with_retry(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        return results

    async def _generate_single_image_with_retry(self, prompt: str) -> bytes:
        """Generates a single image with retry logic."""
        try:
            image_bytes = await self._retry_with_backoff(
                lambda: self._generate_image_with_imagen(prompt)
            )
            return image_bytes
        except Exception as e:
            logging.warning(f"Image generation failed for prompt: {prompt}. Using default image. Error: {e}")
            return self._create_default_image()

    async def _generate_image_with_imagen(self, prompt: str) -> bytes:
        """Helper function to interact with Imagen (async) and handle blank images."""
        try:
            response = await self.model.generate_content_async(
                [prompt],
                sample_count=1,
                aspect_ratio="16:9",
            )
            part = response.candidates[0].content.parts[0]
            if part:
                image_bytes = part.file_data.data
                img = Image.open(BytesIO(image_bytes))
                if img.getbbox() is None:
                    logging.warning("Generated image is blank.")
                    raise ValueError("Blank image generated.")
                return image_bytes
            else:
                raise ValueError("Empty image response")
        except Exception as e:
            logging.error(f"Error during image generation: {e}")
            raise

    def _create_default_image(self) -> bytes:
        """Creates a white default image with proper RGB format"""
        logging.info("Creating a default image as fallback.")
        image = Image.new('RGB', (1920, 1080), (255, 255, 255))
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=85)
        logging.info("Default image created successfully.")
        return img_byte_arr.getvalue()

    async def _retry_with_backoff(self, operation, max_retries=6, initial_delay=5):
        """Execute operation with exponential backoff retry logic (async version)."""
        logging.info("Starting retry with exponential backoff.")
        last_exception = None
        delay = initial_delay

        for attempt in range(max_retries):
            try:
                result = await operation()
                logging.info(f"Operation succeeded on attempt {attempt + 1}.")
                return result
            except Exception as e:
                last_exception = e
                logging.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    logging.info(f"Waiting for {delay} seconds before retrying.")
                    await asyncio.sleep(delay)
                    delay *= 2
        logging.error("Maximum retry attempts reached. Operation failed.")
        raise last_exception

    def _enhance_prompt(self, prompt: str) -> str:
        """
        Enhance the input prompt for an MLB podcast, making it more descriptive.
        """
        base_prompt = prompt.strip()

        if "MLB" not in base_prompt.upper():
            base_prompt = f"MLB: {base_prompt}"

        if "hitting ball" in base_prompt.lower():
            enhanced_prompt = f"{base_prompt}, close-up, dynamic action, batter swinging, ball in flight, stadium background, daytime"
        elif "running bases" in base_prompt.lower():
            enhanced_prompt = f"{base_prompt}, wide shot, runner sliding into base, dust cloud, intense expression, stadium background"
        elif "pitcher throwing ball" in base_prompt.lower():
            enhanced_prompt = f"{base_prompt}, medium shot, pitcher on mound, windup, focused expression, catcher in background, daytime"
        else:
            enhanced_prompt = (
                f"{base_prompt} – In a game that defies expectations, witness heart-stopping plays, "
                f"thunderous home runs, and strategic brilliance unfolding on the diamond. "
                f"Feel the roar of the crowd, the crack of the bat, and the adrenaline-pumping tension "
                f"of every inning. Get ready for an immersive, play-by-play narrative that brings America's pastime to life!"
            )
        return enhanced_prompt
# --- TEST CODE ---
async def main():
    mock_analysis = {
        'key_moments': [
            {'description': 'Moment 1', 'visual_prompt': 'baseball player hitting ball'},
            {'description': 'Moment 2', 'visual_prompt': 'baseball player running bases'},
            {'description': 'Moment 3', 'visual_prompt': 'baseball pitcher throwing ball'},
        ]
    }

    mock_gcs_handler = MockGCSHandler(project_id="gem-rush-007")  # Replace with your project ID.
    generator = CloudVideoGenerator(mock_gcs_handler)
    images = await generator._generate_images(mock_analysis)
    print(f"Generated {len(images)} images.")

    for i, img_bytes in enumerate(images):
        with open(f"image_{i}.jpg", "wb") as f:
            f.write(img_bytes)
    print("Images saved to files (optional).")

if __name__ == "__main__":
    asyncio.run(main())