import time
import os
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, ImageGenerationResponse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Section ---
# Define all your settings here
PROJECT_ID = "silver-455021"  # Your Google Cloud project ID
LOCATION = "us-central1"      # Your Google Cloud region
OUTPUT_DIR = "generated_images" # Folder to save images
MODEL_NAME = "imagen-3.0-generate-002" # The model to use
SEED = 1                      # Seed for generation (requires add_watermark=False)
ADD_WATERMARK = False         # Set to False to use seed, True to disable seed
SLEEP_SECONDS = 60       # Seconds to wait between API calls

# Define the list of prompts to run
PROMPTS_TO_RUN = [
    # --- Hitting Highlights ---
    "Photorealistic action shot: A baseball player hitting a towering home run out of a brightly lit stadium at night.",
    "Dramatic low-angle view of a batter making contact for a walk-off grand slam, teammates starting to celebrate in the dugout.",
    "Action photography style: A runner sliding headfirst into home plate, narrowly avoiding the catcher's tag as dust flies.",
    "Close-up shot of a baseball hitting the sweet spot of a wooden bat, causing splinters to fly.",
    "A line drive screaming past the outstretched glove of a diving third baseman.",

    # --- Pitching Highlights ---
    "Intense focus: A pitcher in mid-delivery during a crucial moment, sweat beading on their brow under stadium lights.",
    "A batter swinging and missing dramatically at a curveball for the final out of the game, catcher jumping up.",
    "Close up of a baseball spinning rapidly towards the plate, captured with a fast shutter speed.",

    # --- Fielding Highlights ---
    "Spectacular diving catch by an outfielder crashing into the wall to rob a home run, ball secured in glove.",
    "Dynamic action shot of a shortstop leaping high to snag a line drive.",
    "A second baseman turning a fast double play, leaping over the sliding runner while throwing to first.",
    "Catcher framing a perfect strike on the edge of the zone, umpire signalling strike three.",

    # --- Moments & Atmosphere ---
    "Wide angle shot of a baseball field during sunset, grounds crew preparing the diamond.",
    "Joyful celebration: A baseball team piling onto the field after winning the championship.",
    "Thousands of fans cheering wildly in the stands after a game-winning play, viewed from the field level."
]
# --- End of Configuration Section ---


def generate_images_batch(
    project_id: str,
    location: str,
    output_dir: str,
    prompts: list[str],
    model_name: str,
    seed: int,
    add_watermark: bool,
    sleep_seconds: int,
) -> list[str]:
    """
    Generates multiple images using a list of text prompts with rate limiting.

    Args:
      project_id: Google Cloud project ID.
      location: Google Cloud region.
      output_dir: Local directory path to save the output image files.
      prompts: A list of text prompts.
      model_name: The specific Imagen model to use.
      seed: Seed for deterministic image generation (only used if add_watermark is False).
      add_watermark: Whether to add a watermark (cannot be used with seed).
      sleep_seconds: Number of seconds to pause between generation requests.

    Returns:
        A list of file paths where the generated images were saved.
        Returns an empty list if initialization or model loading fails.
    """
    saved_files = []

    # --- Parameter Validation ---
    if add_watermark and seed is not None:
        logging.warning("Watermark is enabled, but a seed was provided. "
                        "The 'seed' parameter will be ignored by the API.")
        # We don't need to set seed to None here, the API call logic handles it.

    try:
        logging.info(f"Initializing Vertex AI for project '{project_id}' in location '{location}'...")
        vertexai.init(project=project_id, location=location)
        logging.info("Vertex AI initialized successfully.")

        logging.info(f"Loading image generation model: {model_name}...")
        model = ImageGenerationModel.from_pretrained(model_name)
        logging.info("Model loaded successfully.")

        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Ensured output directory exists: '{output_dir}'")

    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI or load the model: {e}")
        return saved_files # Return empty list on critical failure

    total_prompts = len(prompts)
    for i, prompt in enumerate(prompts):
        start_time = time.time()
        logging.info(f"Processing prompt {i+1}/{total_prompts}: '{prompt}'")

        # Generate a unique filename for each image
        # Using index and seed (if applicable) in the name
        seed_suffix = f"_seed{seed}" if not add_watermark else ""
        output_filename = os.path.join(output_dir, f"image_{i+1}{seed_suffix}.jpeg")

        try:
            logging.info(f"Requesting image generation for prompt {i+1}...")

            # Prepare arguments for the API call
            generation_args = {
                "prompt": prompt,
                "number_of_images": 1,
                "add_watermark": add_watermark,
                # Add other parameters like aspect_ratio if needed
            }
            # Conditionally add seed only if watermark is disabled
            if not add_watermark:
                generation_args["seed"] = seed
                logging.info(f"Using seed={seed} because watermark is disabled.")

            # Make the API call
            images: ImageGenerationResponse = model.generate_images(**generation_args)

            generation_time = time.time() - start_time
            logging.info(f"Image generation API call completed in {generation_time:.2f} seconds.")

            # --- CORRECTED IMAGE ACCESS ---
            # Check if the response object is valid and assume the first image is the one we want
            if images and images[0]:
                logging.info(f"Saving generated image to: {output_filename}")
                # Access the first image directly using index [0]
                images[0].save(location=output_filename, include_generation_parameters=True)
                saved_files.append(output_filename)
                logging.info(f"Successfully saved {output_filename}")
            else:
                # This condition might be less likely now, but good to keep
                logging.warning(f"No valid image data received for prompt {i+1}: '{prompt}'")
            # --- END CORRECTION ---

        except AttributeError as ae:
             generation_time = time.time() - start_time
             logging.error(f"AttributeError encountered processing prompt {i+1} ('{prompt}') after {generation_time:.2f} seconds: {ae}. Perhaps the response structure changed?", exc_info=True)
        except Exception as e:
            generation_time = time.time() - start_time
            logging.error(f"Failed to generate or save image for prompt {i+1} ('{prompt}') after {generation_time:.2f} seconds: {e}", exc_info=True) # Add exc_info for more details on errors

        # Rate Limiting: Sleep after each attempt (success or failure)
        # Avoid sleeping after the very last prompt
        if i < total_prompts - 1:
            logging.info(f"Sleeping for {sleep_seconds} seconds before next request...")
            time.sleep(sleep_seconds)
        else:
            logging.info("Finished processing all prompts.")

    return saved_files

# --- Main execution block ---
if __name__ == "__main__":
    # Validate the configuration combination
    if ADD_WATERMARK and SEED is not None:
         logging.warning("Configuration conflict: ADD_WATERMARK is True and SEED is set. "
                         "The seed will be ignored by the API. Set SEED to None or ADD_WATERMARK to False to avoid this warning.")

    if not PROMPTS_TO_RUN:
        logging.error("The list 'PROMPTS_TO_RUN' in the script is empty. Please add prompts.")
    else:
        logging.info(f"Starting image generation for {len(PROMPTS_TO_RUN)} prompts defined in the script.")
        logging.info(f"Config: Project={PROJECT_ID}, Location={LOCATION}, Model={MODEL_NAME}, "
                     f"Seed={SEED if not ADD_WATERMARK else 'N/A'}, Watermark={ADD_WATERMARK}, Output={OUTPUT_DIR}, Delay={SLEEP_SECONDS}s")

        generated_files = generate_images_batch(
            project_id=PROJECT_ID,
            location=LOCATION,
            output_dir=OUTPUT_DIR,
            prompts=PROMPTS_TO_RUN,
            model_name=MODEL_NAME,
            seed=SEED,
            add_watermark=ADD_WATERMARK,
            sleep_seconds=SLEEP_SECONDS,
        )

        if generated_files:
            logging.info("\n--- Generation Summary ---")
            logging.info(f"Successfully generated {len(generated_files)} images:")
            for file_path in generated_files:
                logging.info(f"- {file_path}")
        else:
            logging.warning("No images were generated successfully.")