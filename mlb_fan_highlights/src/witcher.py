# -----------------from IPython.display import HTML, Markdown, display  # These are only needed in Jupyter/Colab
# from google import genai # Not directly needed if using Vertex AI
from google.cloud import logging as cloud_logging
from vertexai.preview.vision_models import ImageGenerationModel
from typing import List, Dict, Any
import vertexai
import logging
import json
import time
import re
import os
import tempfile
from google.genai.types import (
    FunctionDeclaration,
    GenerateContentConfig,
    GoogleSearch,
    Part,
    Retrieval,
    SafetySetting,
    Tool,
    ToolCodeExecution,
    VertexAISearch,
)
from google import genai

# Configure cloud logging
logging.basicConfig(level=logging.INFO)
log_client = cloud_logging.Client()
log_client.setup_logging()

# Vertex AI Initialization (Do this once at the beginning)
PROJECT_ID = "gem-rush-007"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)
MODEL_ID = "gemini-2.0-pro-exp-02-05" #For gemini model
fast_imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001") #for imagen model

def generate_video_editing_suggestions(prompt_text: str) -> Dict[str, Any]:
    """Generates video editing suggestions (including visual prompts) from a podcast script."""

    system_instruction = """
    You are a video editor assistant. Analyze the provided baseball podcast script and produce a JSON output containing video editing suggestions. The podcast summarizes a baseball game. All game data is typically sourced from a stats API (like the MLB Stats API, as mentioned in the example, but adapt to any source). The podcast starts at 00:00:00.

    Follow this chain of thought:

    1. **Identify Key Moments:** Read the podcast script and identify the most important events (e.g., hits, runs, home runs, errors, pitching changes, game start, game end, key player stats, mentions of specific innings). These will be the basis of your video segments. Prioritize moments that significantly impact the game's score or momentum.

    2. **Process Each Key Moment:** For each key moment identified in step 1, create a dictionary with the following keys and values:
        *  `"timestamp"`: Estimate the time the event is discussed in the podcast script. The podcast begins at 00:00:00. Express the timestamp in "HH:MM:SS" format (Hours:Minutes:Seconds). Increment the time logically based on the flow of the conversation. Assume each speaker's turn takes approximately 10-20 seconds, but adjust based on the length of their dialogue.
        *  `"description"`: Briefly describe the key moment (e.g., "Player X hits a single", "Team Y scores a run", "Pitcher Z is replaced"). Use the names provided in the script.
        *  `"visual_prompt"`: Provide a concise prompt suitable for an image generation model (like Imagen) to create a visual representing this key moment. Be specific about the players (using their names), the action, the team uniforms (if identifiable), and the setting (e.g., "Close-up of Player X hitting a baseball, Team Y uniform, daytime game, baseball stadium in background"). If a player's name isn't given, use a generic term like "batter" or "pitcher". Focus on the *action* and key visual elements.  If the team is mentioned, include it in the prompt.
        *  `"duration"`: Suggest a duration, in seconds, for this video segment. Keep all durations at 5 seconds, as requested.
        * `"transition"`: Suggest a transition effect to the *next* segment. Choose from "fade", "cut", or "zoom". Use "cut" for abrupt changes, "fade" for smoother transitions, and "zoom" to emphasize a particular detail.

    3. **Determine Overall Theme:** Based on the entire podcast script, suggest an overall theme for the video. Choose *one* of the following:
        *  `"modern"`: A clean, contemporary style.
        *  `"retro"`: A vintage, old-school look.
        *  `"dramatic"`: An intense, high-energy style. Consider using "dramatic" if the game was close or had significant turning points.

    4. **Define Color Palette:** Based on the chosen `theme`, suggest a color palette.  If team colors are clearly identifiable from the podcast, try to incorporate them subtly. Otherwise, choose colors appropriate to the theme. Provide hex color codes (e.g., "#RRGGBB") for:
        *  `"primary"`: The main color.
        *  `"secondary"`: A complementary color.
        *  `"accent"`: A color used for highlights and emphasis.

    5. **Choose Graphics Style:** Based on the `theme` and the content, select a graphics style. Choose *one* of the following:
        *  `"dynamic"`: Fast-paced, with moving elements. Suitable for exciting games.
        *  `"animated"`: Uses animations to illustrate events. Good for explaining complex plays.
        *  `"static"`: Uses still images and text. Best for slower-paced analysis or games with less action.

    6. **Suggest Audio Intensity:** On a scale of 0-100 (0 being silent, 100 being very loud), suggest an overall audio intensity level for the video. Consider the excitement level of the game and the commentary. Higher intensity for exciting games, lower for more analytical discussions.

    Finally, output your suggestions in the following JSON format:

    ```json
    {
        "key_moments": [
            {
                "timestamp": "HH:MM:SS",
                "description": "text",
                "visual_prompt": "Imagen prompt",
                "duration": 5,
                "transition": "fade/cut/zoom"
            },
            ...
        ],
        "theme": "modern/retro/dramatic",
        "color_palette": {
            "primary": "#hex",
            "secondary": "#hex",
            "accent": "#hex"
        },
        "graphics_style": "dynamic/animated/static",
        "audio_intensity": "0-100"
    }
    ```
    """
    prompt = f"""
    {prompt_text}
    """

    safety_settings = [
        SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_LOW_AND_ABOVE",
        ),
        SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_LOW_AND_ABOVE",
        ),
        SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_LOW_AND_ABOVE",
        ),
        SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_LOW_AND_ABOVE",
        ),
    ]

    client = vertexai.preview.generative_models.GenerativeModel(MODEL_ID)
    response = client.generate_content(
        contents=[prompt],
        generation_config=vertexai.preview.generative_models.GenerationConfig(
            temperature=0.4,
            top_p=0.95,
            top_k=20,
            candidate_count=1,
            # seed=5,  #  Seed is not supported in the Vertex AI SDK yet.
            max_output_tokens=2048,
            system_instruction=system_instruction,
            stop_sequences=["STOP!"],
            # presence_penalty=0.0, # Not supported
            # frequency_penalty=0.0, # Not Supported
        ),
        safety_settings=safety_settings,
        # tools=tools, # If you had function calling/tools.
    )

    # print(response.text)
    return _parse_gemini_response(response.text)

def _parse_gemini_response(raw_response: str) -> Dict[str, Any]:
    """Parse and validate Gemini response with error handling."""
    logging.info("Parsing Gemini response.")
    try:
        # Use a regular expression to find the JSON content
        match = re.search(r"```(json)?(.*)```", raw_response, re.DOTALL)
        if match:
            json_str = match.group(2).strip()
        else:
            logging.warning("No JSON block found in response.")
            print("Raw response:", raw_response)  # Debug: Print raw response
            raise ValueError("No JSON block found in response.")

        # Check if JSON is incomplete
        if not json_str.endswith("}"):
            logging.warning("Incomplete JSON detected.  Generation might have been cut off.")
            print("Raw response:", raw_response)
            # You could try re-prompting here, or handle the partial JSON
            raise ValueError("Incomplete JSON detected.")

        parsed = json.loads(json_str)
        logging.info("Successfully parsed Gemini response.")
        return parsed
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {str(e)}")
        print("Raw response:", raw_response)  # Debug: Print raw response
        print("Extracted JSON string:", json_str)  # Print Extracted string
        raise ValueError("Failed to parse AI response")
    except ValueError as ve:
        raise ve  # Re-Raise ValueErrors



def enhance_prompts(prompts_to_enhance: List[str]) -> List[str]:
    """Enhances a list of visual prompts for better image generation."""

    enhancement_prompt = """
    You are a creative prompt engineer for a visual generation model.  I will give you a list of prompts that describe scenes from a baseball game.  Your task is to enhance each prompt, making it more vivid, descriptive, and suitable for generating high-quality, engaging images. Add details about the stadium environment, the weather, the time of day, the camera angle, and the overall atmosphere.  Focus on capturing the energy and excitement of a Major League Baseball game. Output the enhanced prompts as a numbered list, where each enhanced prompt is on a new line and prepended by number in the format "1. ".

    Original Prompts:
    """ + "\n".join(prompts_to_enhance)

    client = vertexai.preview.generative_models.GenerativeModel(MODEL_ID)

    safety_settings = [ #copy from above
        SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_LOW_AND_ABOVE",
        ),
        SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_LOW_AND_ABOVE",
        ),
        SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_LOW_AND_ABOVE",
        ),
        SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_LOW_AND_ABOVE",
        ),
    ]

    enhanced_response = client.generate_content(
        contents=[enhancement_prompt],
        generation_config=vertexai.preview.generative_models.GenerationConfig(
            temperature=0.7,  # Slightly higher temperature for more creativity
            top_p=0.95,
            top_k=40,
            candidate_count=1,
            max_output_tokens=4096, #increase max output tokens
        ),
        safety_settings=safety_settings
    )

    return _parse_enhanced_prompts(enhanced_response.text)


def _parse_enhanced_prompts(raw_response: str) -> List[str]:
    """Extracts a list of enhanced prompts from the Gemini response."""
    try:
        # Regular expression to find numbered list items.
        matches = re.findall(r"^\s*(\d+)\.\s*(.*)", raw_response, re.MULTILINE)

        # Check for incomplete list.
        if not matches:
            if "1." in raw_response: # Check if the numbering just didn't complete
                logging.warning("Incomplete numbered list detected. Generation may have been cut off.")
                print("Raw Response", raw_response)
            else:
                logging.warning("No numbered list found in the response")
                print("Raw Response", raw_response)
            raise ValueError("No numbered list found in the response.")


        # Sort the matches by number (important in case of out-of-order generation)
        # and extract just the prompt text.
        enhanced_prompts = [prompt for _, prompt in sorted(matches, key=lambda x: int(x[0]))]

        return enhanced_prompts

    except Exception as e:
        logging.error(f"Error parsing enhanced prompts: {e}")
        print("Raw Response:", raw_response) #Print raw response.
        raise ValueError(f"Failed to parse enhanced prompts: {e}")


def generate_and_save_image(prompt: str, model: ImageGenerationModel, filename: str, delay_seconds: int = 20, num_images: int = 1) -> None:
    """Generates and saves an image from a prompt, with error handling."""
    try:
        response = model.generate_images(prompt=prompt, number_of_images=num_images)

        if not response.images:
            print(f"No images generated for prompt '{prompt}'.")
            return

        image = response.images[0]
        image.save(filename)  # Save directly to the desired filename
        print(f"Image for prompt '{prompt}' saved to {filename}")
        print(f"Waiting {delay_seconds} seconds before next prompt...")
        time.sleep(delay_seconds)

    except Exception as e:
        print(f"Error generating or saving image for prompt '{prompt}': {e}")

def main():
    """Main function to orchestrate the entire process."""

    podcast_script = """
    [{'speaker': 'Play-by-play Announcer', 'text': "Welcome, everyone, to today's podcast! Today is 2025-02-19 11:03:17. We're going to be discussing the Minnesota Twins' last game on 2024-09-29. All game data and statistics are sourced from the MLB Stats API."}, {'speaker': 'Color Commentator', 'text': "That's right! The Twins faced off against the Detroit Tigers on September 29th, 2024. According to the MLB Stats API, the final score was 4-3, with the Tigers taking the win."}, {'speaker': 'Play-by-play Announcer', 'text': "Let's dive into some of the key moments of that game. In the top of the 1st inning, with one out, Wenceel Perez singled on a line drive to right fielder Max Kepler. 
    Kerry Carpenter then doubled on a sharp line drive to right fielder Max Kepler. Wenceel Perez scored."}, {'speaker': 'Color Commentator', 'text': 'A quick start for the Tigers, putting them up 1-0 early. The MLB Stats API data shows how the Tigers capitalized on those early hits.'}, {'speaker': 'Player Quotes', 'text': 'We needed to get something going early, and I was able to put a good swing on it. Got us on the board.'}, {'speaker': 'Play-by-play Announcer', 'text': 'The score remained 1-0 in favor of the Tigers until the top of the 3rd inning. With one out, Wenceel Perez homered to right field. A solo shot!'}, {'speaker': 'Color Commentator', 'text': "Perez, having a great game, 
    extending the Tigers' lead to 2-0. Data from the MLB Stats API shows that was a key hit to increase the lead early."}, {'speaker': 'Player Quotes', 'text': 'I felt good at the plate that day, you know, was seeing the ball well.'}, {'speaker': 'Play-by-play Announcer', 'text': 'In the bottom of the 3rd, the Twins responded. With no outs, Willi Castro singled on a line drive to left fielder Kerry Carpenter. Then, Manuel Margot doubled on a line drive to left fielder Kerry Carpenter. Willi Castro scored.'}, {'speaker': 'Color Commentator', 'text': 'The Twins getting on the board, making it 2-1. The MLB Stats API confirms that run was crucial for the Twins to get back into the game.'}, {'speaker': 'Player Quotes', 'text': 'We needed an answer, and I was just trying to put the ball in play. Glad I could help get us on the board!'}, {'speaker': 'Play-by-play 
    Announcer', 'text': 'In the top of the 6th, with two outs, Parker Meadows homered to right center field. A solo shot.'}, {'speaker': 'Color Commentator', 'text': "Another home run, this time for the Tigers' Parker Meadows, extending their lead to 3-1. MLB Stats API shows another key hit late in the game."}, {'speaker': 'Player Quotes', 'text': 
    'Yeah I connected on that one. Felt great off the bat.'}, {'speaker': 'Play-by-play Announcer', 'text': 'In the top of the 8th inning, Colt Keith homered to right field. A solo shot making it 4-1 for Detroit!'}, {'speaker': 'Color Commentator', 'text': 'Colt Keith with another home run for Detroit. That extends the lead to 4-1. The MLB Stats API confirms that was a solo shot.'}, {'speaker': 'Player Quotes', 'text': 'I saw it and had to swing, Glad I could help the team!'}, {'speaker': 'Play-by-play Announcer', 'text': 'The Twins tried to mount a comeback in the bottom of the 9th. Carlos Santana homered to right, a solo shot with no outs, cutting the lead to 4-2.'}, {'speaker': 'Color Commentator', 'text': 'Santana trying to spark a rally, making it a two-run game! MLB Stats API data confirms that the home run brought the score to 4-2.'}, {'speaker': 'Player Quotes', 'text': 'Just trying to get something started, you know? Never give up.'}, {'speaker': 'Play-by-play Announcer', 'text': 'Then with one out, Kyle Farmer homered to left field. The lead is cut down to 1 run.'}, {'speaker': 'Color Commentator', 'text': 'Back to Back home runs, and the lead is cut down to one run. Farmer bringing the Twins within striking distance! The MLB Stats API is showing the score is now 4-3!'}, {'speaker': 'Player Quotes', 'text': "Gotta keep fighting. That's what we tried to do."}, {'speaker': 'Play-by-play Announcer', 'text': "But that's all the Twins could muster. The Tigers held on to win 4-3."}, {'speaker': 'Color Commentator', 'text': "A close game, but Detroit's early offense and a couple of late home runs proved to be the difference. Let's look at some player stats, courtesy of the MLB Stats API. For the Twins, Carlos Santana went 1 for 4 with a home run and an RBI. Kyle Farmer also went 1 for 4 with a home run and an RBI. For the Tigers, Wenceel Perez went 3 for 4, with a home run, a double, an RBI, and two runs scored. Kerry Carpenter went 2 for 4 with a double and an RBI"}, {'speaker': 'Play-by-play Announcer', 'text': 'The starting pitcher for the Twins was Bailey Ober. The MLB Stats API reports he pitched 6 innings, allowing 7 hits, 3 earned runs and striking out 6. For the Tigers, the starting pitcher was Reese Olson. He pitched 5.1 innings, allowing 5 hits, 1 earned run, and struck out 7, according to the MLB Stats API.'}, {'speaker': 'Color Commentator', 'text': 'For the Twins, other key contributors were Willi Castro, who went 1 for 4 with a run, and Manuel Margot who went 2 for 4, with a double, an RBI, and a walk. On the Tigers side, Parker Meadows had one hit, a home run, and Colt Keith also had a home run going 1 for 3.'}]
    """

    # 1. Generate video editing suggestions (and initial prompts).
    video_suggestions = generate_video_editing_suggestions(podcast_script)

    # 2. Extract and enhance the visual prompts.
    original_prompts = [item["visual_prompt"] for item in video_suggestions["key_moments"]]
    enhanced_prompts = enhance_prompts(original_prompts)


    # 3. Update the video suggestions with the enhanced prompts.
    if len(enhanced_prompts) == len(video_suggestions["key_moments"]):
        for i, moment in enumerate(video_suggestions["key_moments"]):
            moment["visual_prompt"] = enhanced_prompts[i]
    else:  # Crucial error handling.
        print("Error: Number of enhanced prompts does not match number of key moments.  Using original prompts.")
        # In a real application, you might want more sophisticated error handling
        # (e.g., retrying the enhancement, logging the error, etc.)

    print(json.dumps(video_suggestions, indent=4))

    # 4. Generate and save images.
    for i, moment in enumerate(video_suggestions["key_moments"]):
        filename = f"output_images/moment_{i}.png"  # Save to a numbered file
        #Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        generate_and_save_image(moment["visual_prompt"], fast_imagen_model, filename)

if __name__ == "__main__":
    main()