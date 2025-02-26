# Copyright 2025 [Your Name or Organization]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import sys

# from google import genai  # Use Vertex AI SDK
# from google.genai import types # Use Vertex AI SDK
from vertexai.language_models import TextGenerationModel  # For Gemini
from vertexai.vision_models import Image, ImageGenerationModel  # For Imagen
import vertexai
import matplotlib.pyplot as plt
from typing import List, Dict, Union

# --- Authentication and Setup (Modified for Vertex AI SDK) ---

PROJECT_ID = "gem-rush-007"  # Replace with your actual project ID
LOCATION = "us-central1"
DATASET_ID = "mlb_data_2024" #keep dataset name
TABLE_ID = "game_events_hybrid" #keep dataset name
BUCKET_URI = f"gs://{PROJECT_ID}-vs-hybridsearch-mlb"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"

vertexai.init(project=PROJECT_ID, location=LOCATION) # Using Vertex SDK
#Use both Text and image generation model.
text_generation_model = TextGenerationModel.from_pretrained("gemini-2.0-flash-001") # Gemini
image_generation_model = ImageGenerationModel.from_pretrained("imagegeneration@005") # Imagen


# --- Helper Functions ---

def display_images_in_grid(images: List[Image]):
    """Displays a list of Vertex AI Image objects in a grid."""
    #Note the Vertex AI image object does not have _pil_image, we use the as_image function.
    nrows = math.ceil(len(images) / 4)
    ncols = min(len(images) + 1, 4)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6))

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].as_image())  # Use .as_image() for Vertex AI Image
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def generate_content(prompt: str, temperature: float = 0.5, top_p: float = 0.8, top_k: int = 10) -> str:
    """Generates text content using the Gemini model."""
    #Removed generation_config, using parameters directly.
    responses = text_generation_model.predict(
        prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return responses.text


def generate_image(prompt: str, number_of_images: int = 1) -> List[Image]:
    """Generates images using the Imagen model."""
     #Removed GenerateImagesConfig, using parameters directly.
    response = image_generation_model.generate_images(
        prompt=prompt,
        number_of_images=number_of_images,
    )
    return response.images


# --- Storyboard Prompts (Organized by Category) ---

STORYBOARD_PROMPTS = {
    "general": {
        "G1": "Wide, panoramic view of a bustling baseball stadium filled with fans.  It's {time_of_day}. The field is perfectly manicured, and the stadium lights are bright.  Photorealistic, high detail.",
        "G2": "Inside the announcer's booth. Two announcers, {announcer_1} and {announcer_2}, are sitting at a desk with microphones and monitors.  They are looking intently at the field.  Realistic, well-lit.",
        "G3": "Close-up of a digital scoreboard graphic showing the team logos, the score, the inning, and the date.  {team_a} vs. {team_b}. Score: {score_a}-{score_b}. Inning: {inning}. Date: {date}. Clean, modern design.",
        "G4": "A medium shot that showcases both home and visiting teams dugouts, the tension can be appreciated",
    },
    "home_run": [
        "A batter at home plate, in a {team_name} uniform, digging into the batter's box. The pitcher is on the mound, ready to deliver. Tense atmosphere. Photorealistic.",
        "Close-up on the pitcher, mid-wind-up. Focus on the intense expression and the grip on the ball. Motion blur on the arm. Dramatic lighting.",
        "The batter, mid-swing, connecting powerfully with the ball.  Slight motion blur on the bat.  The catcher is reacting behind him.  Dynamic angle.",
        "The baseball, soaring high above the field, leaving a slight trail (you can decide if it's a fiery trail or a normal one).  The outfielders are looking up, tracking the ball. Wide shot of the stadium.",
        "Fans in the stands jumping to their feet, cheering and pointing at the sky.  A mix of excitement and awe on their faces.",
        "The batter, rounding first base, watching the ball fly.  A confident or triumphant expression.  Slight blur on the background.",
    ],
    "base_hit": [
        "Close-up of the bat hitting the ball.  No flames, just a solid connection.  Wood grain of the bat is visible.  Sharp focus.",
        "The baseball, bouncing or rolling through the infield.  Infielders are reacting, moving to field the ball.  Motion blur on the ball.",
        "The batter, sprinting towards first base, head down, determined.  Slight motion blur on the legs.",
        "A close play at first base (or second, if it's a double).  The runner sliding or reaching for the bag, the fielder stretching to make the catch and tag.  Dust flying.  Intense focus.",  # Combine out/safe
    ],
    "strikeout": [
        "The batter at the plate, looking tense.  The count is 3-2.  The catcher is giving signs.  Focus on the batter's eyes.",
        "Close-up of the pitcher's hand releasing the ball.  Motion blur.  Focus on the grip and the spin on the ball.",
        "The batter swings and misses.  The bat whips through the air.  The catcher catches the ball cleanly.  Dejected expression on the batter's face.",
        "Close-up of the catcher, pumping his fist or giving a signal to the pitcher.  Confident expression.",
        "The batter walking away, helmet in hand and the catcher celebrating behind him with the umpire",
    ],
    "walk": [
        "Close-up of the umpire's hand signaling ball four.  The pitch is outside the strike zone.",
        "The batter, dropping his bat and starting to jog to first base.  A relieved or neutral expression.",
        "Close up on a frustated pitcher, taking his hat off",
    ],
     "on_base_percentage": [ #This is more conceptual, so prompts would depend.
        "A montage of different successful at-bats: a walk, a single, a hit-by-pitch, a double. Each shot is very short (1-2 seconds). Fast-paced editing.",
        "A graphic showing the player's OBP statistic, prominently displayed. Clean, modern design. The player's name and team logo are also visible."
    ],
    "dominant_pitching": [
        "The pitcher, standing on the mound, looking confident and in control.  The stadium is in the background.  Slight low angle to make the pitcher look imposing.",
        "A series of close-ups showing different grips and releases: fastball, curveball, slider, changeup.  Motion blur on each pitch.  Focus on the hand and ball.",
        "A montage of batters swinging and missing, or looking at called third strikes.  Fast-paced editing.",
        "Close-up of the pitcher, after a key strikeout.  A subtle fist pump or a determined nod.  Controlled emotion.",
        "A close up of the screen, showing the Pitcher's name and a low ERA",
    ],
    "struggling_pitching": [
        "The pitcher on the mound, looking dejected or frustrated.  The pitching coach or manager is walking out to talk to him.  Concerned expressions.",
        "A montage of batters hitting the ball hard: line drives, doubles, home runs.  Focus on the impact and the ball's flight.",
        "Close-up of the pitcher, shaking his head, wiping sweat from his brow, or looking down at the ground.  Clear signs of frustration or disappointment.",
        "The manager walking out to the mound and signaling to the bullpen.  The pitcher handing over the ball and walking off the field.",
    ],
    "great_play": [
        "A hard-hit ball heading towards the outfield (or a specific infield position).",
        "The fielder making a spectacular diving catch, a leaping grab, or a long stretch to field the ball.  Motion blur to emphasize the athleticism.",
        "Close-up of the glove making the catch.  The ball is securely in the webbing.  Dust or grass flying.",
        "The fielder celebrating the catch with teammates.  High fives, fist bumps, or a triumphant yell.",
    ],
    "key_player_statistics": [
        "Close up of the player name on the back of the jersey",
        "The player connecting with the ball",
        "The player running the bases",
        "A display showing relevant stats",
    ],
    "quotes": [
        "A close-up of the player's face during an interview. He's wearing his team uniform or casual clothes. Expressive eyes, possibly with a microphone visible at the edge of the frame.",
        "A clean, modern graphic overlay with the player's quote in text. Use a legible font and contrasting colors. Include the player's name and team logo in a smaller size.",
        # Optional: "If the quote is related to a specific play, show a replay of that play *before* showing the quote graphic. This adds context."  (This is an instruction, not a prompt)
    ],
}

def improve_image_prompt(description: str) -> str:
    """Improves a given description into a more detailed image prompt."""
    prompt_template = """
      Rewrite "{image_prompt}" into an image prompt.
      For example: A sketch of a modern apartment building surrounded by skyscrapers.
      "A sketch" is a style.
      "A modern apartment building" is a subject.
      "Surrounded by skyscrapers" is a context and background.

      Here are a few "styles" to get inspiration from:
      - A studio photo
      - A professional photo
      - Photorealistic
      - Cinematic

      Here are a few "context and background" to inspiration from:
      - In a kitchen on a wooden surface with natural lighting
      - On a marble counter top with studio lighting
      - In an Italian restaurant
      - In a baseball stadium

      The final rewritten prompt should be a single sentence.
    """
    text_prompt = prompt_template.format(image_prompt=description)
    return generate_content(text_prompt)

def generate_menu_json(prompt:str) -> Dict:
    """Generates a restaurant menu in JSON format using Gemini."""
    response = generate_content(prompt)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        print("Error: Gemini did not return valid JSON.  Returning an empty menu.")
        return {"starters": [], "main courses": [], "desserts": [], "drinks": []}


def generate_visual_assets(menu: Dict):
    """Generates visual assets for a menu, using prompt improvement."""
    for category, items in menu.items():
        for item in items:
            improved_prompt = improve_image_prompt(item["description"])
            print(f"Generating images for {item['name']}:")
            print(f"  Original Description: {item['description']}")
            print(f"  Improved Prompt: {improved_prompt}")
            try:
                images = generate_image(improved_prompt, number_of_images=4)
                display_images_in_grid(images)
            except Exception as e:
                print(f"Error generating images for {item['name']}: {e}")
            print("-" * 20)



def generate_storyboard_images(script: List[Dict], game_data: Dict = None):
    """
    Generates images for a given MLB podcast script, using the storyboard prompts.

    Args:
        script: A list of dictionaries, where each dictionary represents a line
                in the podcast script (with 'speaker' and 'text' keys).
        game_data: A dictionary containing game-specific information
                   (e.g., team names, player names, scores, dates).  This is
                   used to fill in placeholders in the prompts.
    """

    if game_data is None:
        game_data = {}  # Provide a default if no game data is given

    for line in script:
        speaker = line["speaker"]
        text = line["text"]
        print(f"Processing line: Speaker: {speaker}, Text: {text}")


        # General Attribution / Introduction
        if speaker == "General Attribution":
          try:
            images = generate_image(STORYBOARD_PROMPTS["general"]["G1"].format(**game_data), number_of_images=1)
            display_images_in_grid(images)
          except Exception as e:
            print(f"An error occurred: {e}")

        elif speaker == "Play-by-play Announcer" and "Welcome" in text:
          try:
            images = generate_image(STORYBOARD_PROMPTS["general"]["G2"].format(**game_data), number_of_images=1)
            display_images_in_grid(images)
          except Exception as e:
            print(f"An error occurred: {e}")
            images = generate_image(STORYBOARD_PROMPTS["general"]["G3"].format(**game_data), number_of_images=1)
            display_images_in_grid(images)
          except Exception as e:
            print(f"An error occurred: {e}")

        # Home Run
        elif "homered" in text or "home run" in text:
            for prompt_template in STORYBOARD_PROMPTS["home_run"]:
                prompt = prompt_template.format(**game_data)
                try:
                    images = generate_image(prompt, number_of_images=1)
                    display_images_in_grid(images)
                except Exception as e:
                    print(f"An error occurred on prompt image generation {prompt}: {e}")

        # Base Hit (Single/Double)
        elif "singled" in text or "doubled" in text:
            for prompt_template in STORYBOARD_PROMPTS["base_hit"]:
                prompt = prompt_template.format(**game_data)
                try:
                    images = generate_image(prompt, number_of_images=1)
                    display_images_in_grid(images)
                except Exception as e:
                   print(f"An error occurred on prompt image generation {prompt}: {e}")
        # Strikeout
        elif "struck out" in text:  # Could also look for specific pitcher stats
             for prompt_template in STORYBOARD_PROMPTS["strikeout"]:
                prompt = prompt_template.format(**game_data)
                try:
                    images = generate_image(prompt, number_of_images=1)
                    display_images_in_grid(images)
                except Exception as e:
                   print(f"An error occurred on prompt image generation {prompt}: {e}")

        # Walk (less common, but possible)
        elif "walked" in text:
            for prompt_template in STORYBOARD_PROMPTS["walk"]:
                prompt = prompt_template.format(**game_data)
                try:
                  images = generate_image(prompt, number_of_images=1)
                  display_images_in_grid(images)
                except Exception as e:
                  print(f"An error occurred on prompt image generation {prompt}: {e}")

        # Pitcher Performance (Dominant or Struggling)
        elif "pitched" in text:
            if "ERA of 0.00" in text:  # Example of a dominant performance indicator
                for prompt_template in STORYBOARD_PROMPTS["dominant_pitching"]:
                    prompt = prompt_template.format(**game_data)
                    try:
                        images = generate_image(prompt, number_of_images=1)
                        display_images_in_grid(images)
                    except Exception as e:
                        print(f"An error occurred on prompt image generation {prompt}: {e}")
            elif "giving up" in text and "runs" in text: # Example of a struggling performance
                for prompt_template in STORYBOARD_PROMPTS["struggling_pitching"]:
                    prompt = prompt_template.format(**game_data)
                    try:
                        images = generate_image(prompt, number_of_images=1)
                        display_images_in_grid(images)
                    except Exception as e:
                      print(f"An error occurred on prompt image generation {prompt}: {e}")

        # Player Quotes
        elif speaker == "Player Quotes":
            for prompt_template in STORYBOARD_PROMPTS["quotes"]:
                prompt = prompt_template.format(**game_data)
                try:
                  images = generate_image(prompt, number_of_images=1)
                  display_images_in_grid(images)
                except Exception as e:
                   print(f"An error occurred on prompt image generation {prompt}: {e}")

        # Key Player Stats:
        elif speaker == "Color Commentator" and "stats" in text:
          for prompt_template in STORYBOARD_PROMPTS["key_player_statistics"]:
                prompt = prompt_template.format(**game_data)
                try:
                    images = generate_image(prompt, number_of_images=1)
                    display_images_in_grid(images)
                except Exception as e:
                    print(f"An error occurred on prompt image generation {prompt}: {e}")


        # Other lines (e.g., general commentary) - No specific image, or use a general stadium shot
        else:
            print("No specific image prompt for this line.")
            #Optionally use a default prompt.
            # images = generate_image(STORYBOARD_PROMPTS["general"]["G1"].format(**game_data), number_of_images=1)
            # display_images_in_grid(images)

        print("-" * 20)

def extract_game_data(script: List[Dict]) -> Dict:
    """
    Extracts relevant game data from the script to use in prompt formatting.
    This is a simplified example and would need to be expanded for more robust data extraction.
    """
    game_data = {
        "team_a": "",
        "team_b": "",
        "score_a": "0",
        "score_b": "0",
        "inning": "1st",
        "date": "Unknown",
        "announcer_1": "Play-by-play Announcer",
        "announcer_2": "Color Commentator",
        "time_of_day": "Daytime",  # Default
        "team_name": ""
    }

    for line in script:
        text = line["text"].lower()
        speaker = line["speaker"]

        # Extract team names (very basic example)
        if " vs. " in text:
            parts = text.split(" vs. ")
            if len(parts) == 2:
                game_data["team_a"] = parts[0].split()[-1] #last word
                game_data["team_b"] = parts[1].split()[0] #first word
        if "yankees" in text.lower():
            game_data["team_name"] = "Yankees"
        elif "dodgers" in text.lower():
            game_data["team_name"] = "Dodgers"
        if "diamondbacks" in text.lower():
            game_data["team_name"] = "Diamondbacks"
        #Extract Date
        if speaker == "Play-by-play Announcer" and "welcome" in text.lower():
            words = text.split()
            for i, word in enumerate(words):
                if word.lower() == "today" and i + 2 < len(words):
                  month_day_year = words[i+2] + words[i+3]

        # Extract score (very basic - needs improvement)
        if "final score was" in text:
            try:
                score_part = text.split("final score was ")[1]
                score_a, score_b = score_part.split(" in favor")[0].split("-")
                game_data["score_a"] = score_a.strip()
                game_data["score_b"] = score_b.strip()
            except Exception:
              pass

        if "inning" in text:
            try:
              for inning_word in ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "top", "bottom"]:
                if inning_word in text:
                  game_data["inning"] = text.split(inning_word)[0].strip() + " " + inning_word
            except:
              pass

        # Time of day
        if "night" in text:
            game_data["time_of_day"] = "Nighttime"

    return game_data

# --- Main Script Execution ---

if __name__ == "__main__":
    # Example usage with one of your provided scripts:
    script1 = [
        {'speaker': 'General Attribution', 'text': 'All game data and statistics are sourced from the MLB Stats API.'},
        {'speaker': 'Play-by-play Announcer', 'text': "Welcome, everyone! Today is October 9, 2024, and we're going to recap the Dodgers' last game, which took place on October 7, 2024, against the Arizona Diamondbacks. The final score was 8-5 in favor of the Diamondbacks."},
        {'speaker': 'Color Commentator', 'text': 'This was Game 3 of the NLDS. The Dodgers were looking to bounce back after splitting the first two games in Los Angeles.'},
        {'speaker': 'Play-by-play Announcer', 'text': 'In the top of the first inning, Mookie Betts singled, and then Freddie Freeman doubled, driving in Betts. Dodgers took an early 1-0 lead.'},
        {'speaker': 'Color Commentator', 'text': 'A great start for the Dodgers. According to the MLB Stats API, Betts went 1 for 4 in this game, while Freeman went 1 for 5 with an RBI.'},
        {'speaker': 'Player Quotes', 'text': 'Just trying to get on base and start things off. Freddie came through with the big hit.'},
        {'speaker': 'Play-by-play Announcer', 'text': 'The Diamondbacks answered back in the bottom of the first. Ketel Marte homered off Dodgers starter Bobby Miller. Score tied at 1-1.'},
        {'speaker': 'Color Commentator', 'text': 'Marte had himself a game, going 3 for 5 with that home run and 2 RBIs. Miller, on the other hand, lasted only 1.2 innings, giving up 5 earned runs.'},
        {'speaker': 'Player Quotes', 'text': "I just didn't have my best stuff tonight. They were hitting everything I threw up there."},
        {'speaker': 'Play-by-play Announcer', 'text': 'In the bottom of the second, Christian Walker homered off Miller, a three-run shot, making it 4-1 Diamondbacks.'},
        {'speaker': 'Color Commentator', 'text': 'Walker also had a strong game at the plate: 1 for 4 with that crucial 3-run homer. That really changed the momentum.'},
        {'speaker': 'Player Quotes', 'text': 'I got a fastball I could handle and put a good swing on it.'},
        {'speaker': 'Play-by-play Announcer', 'text': 'The Diamondbacks continued the hit parade and scored 4 more runs in the second inning.'},
        {'speaker': 'Color Commentator', 'text': "The game quickly got out of hand. The final score reflects how well the Diamondback's offense played, and how much the Dodger's pitching struggled."},
        {'speaker': 'Play-by-play Announcer', 'text': 'The Dodgers tried to mount a comeback, scoring 3 runs in the top of the third. Will Smith hit a three run home run, bringing the score to 8-4.'},
        {'speaker': 'Color Commentator', 'text': "Will Smith had a good night at the plate, going 2-4 with a home run and 3 RBIs. According to the MLB Stats API, that home run made it a game again, but the Dodgers couldn't keep up the pressure."},
        {'speaker': 'Player Quotes', 'text': 'We were trying to fight back, but their pitchers held us down. I am glad I could do something to help.'},
        {'speaker': 'Play-by-play Announcer', 'text': 'In the top of the 9th inning, Mookie Betts hit a single to score Chris Taylor, making it a final score of 8-5.'},
        {'speaker': 'Color Commentator', 'text': 'A little too late at that point. The Diamondbacks had already done the damage.'},
        {'speaker': 'Play-by-play Announcer', 'text': 'For the Diamondbacks, Brandon Pfaadt started and earned the win, pitching 5.1 innings and giving up 4 earned runs. Key relief pitchers include Ryan Thompson and Paul Sewald. Sewald secured the save. For the Dodgers, as mentioned, Bobby Miller took the loss. Key relief pitchers for the Dodgers were Ryan Yarbrough, and Evan Phillips.'},
        {'speaker': 'Color Commentator', 'text': "That's a wrap on the Dodgers' last game. A tough loss, giving the Diamondbacks a 2-1 series lead."}
    ]

    game_data1 = extract_game_data(script1)
    generate_storyboard_images(script1, game_data1)

    script2 = [{'speaker': 'Play-by-play Announcer', 'text': "Welcome, everyone, to today's podcast! Today is October 9, 2024. We're going to recap the Yankees' last game. All game data and statistics are sourced from the MLB Stats API. The Yankees played against the Orioles on May 2, 2024. According to the MLB Stats API, the final score was 4-2, with the Yankees winning."}, {'speaker': 'Color Commentator', 'text': 'This was a really solid game for the Yankees. They managed to put up a strong offensive performance, but their real strength was in their solid pitching and defense.'}, {'speaker': 'Play-by-play Announcer', 'text': "Let's dive into some of the key moments. In the top of the 1st inning, with one out, and a runner on first, Juan Soto came to the plate. Data from the MLB Stats API shows that he hit a home run on a 1-1 count, driving in 2 runs. The score after the play was 2-0, Yankees leading."}, {'speaker': 'Color Commentator', 'text': "That home run by Soto really set the tone for the game. He's been having an incredible season so far, and that hit was a testament to his ability to deliver in clutch situations."}, {'speaker': 'Player Quotes', 'text': 'I was just trying to stay locked in and put a good swing on the ball. It was a good pitch to hit, and I connected.'}, {'speaker': 'Play-by-play Announcer', 'text': 'In the bottom of the 3rd inning, with no outs and runners on first and second, Anthony Santander hit a single, driving in one run. The score was then 2-1.'}, {'speaker': 'Color Commentator', 'text': 'The Orioles managed to get one back there, capitalizing on some baserunners.'}, {'speaker': 'Player Quotes', 'text': 'We were trying to get some runs on the board. That hit got us a bit closer.'}, {'speaker': 'Play-by-play Announcer', 'text': "Then, in the top of the 4th, with two outs, Gleyber Torres hit a home run, extending the Yankees' lead to 3-1."}, {'speaker': 'Color Commentator', 'text': 'Torres really came through there with a crucial hit, adding an insurance run for the Yankees.'}, {'speaker': 'Player Quotes', 'text': "I got a good pitch to hit and I didn't miss it."}, {'speaker': 'Play-by-play Announcer', 'text': 'In the top of the 9th, Jose Trevino singled, driving in another run, putting the Yankees up 4-1.'}, {'speaker': 'Color Commentator', 'text': 'Another insurance run here, and it came at a crucial time.'}, {'speaker': 'Player Quotes', 'text': 'Just doing what I can to extend our lead and make things tough on them.'}, {'speaker': 'Play-by-play Announcer', 'text': 'Finally, in the bottom of the 9th inning, with one out, Ryan Mountcastle homered, bringing the score to 4-2. That was the end of the scoring though, as the Yankees defense closed it out.'}, {'speaker': 'Color Commentator', 'text': "A little late-game surge from the Orioles, but ultimately not enough to overcome the Yankees' lead."}, {'speaker': 'Player Quotes', 'text': 'We tried to rally there at the end, but fell a bit short.'}, {'speaker': 'Color Commentator', 'text': "Let's take a look at some of the key player stats from the game, according to the MLB Stats API. For the Yankees, Juan Soto went 1 for 4 with a home run, 2 RBIs, and 1 run. Gleyber Torres also went 1 for 4, hitting a home run, with 1 RBI and 1 run. On the Orioles side, Anthony Santander went 2 for 4 with 1 RBI. Ryan Mountcastle went 1 for 4 with 1 Home Run, 1 RBI and 1 run."}, {'speaker': 'Play-by-play Announcer', 'text': 'And for the pitchers, The MLB Stats API reports that for the Yankees, Luis Gil pitched 5.2 innings, allowing 4 hits, 1 earned run, and 3 walks, with 8 strikeouts. For the Orioles, Grayson Rodriguez pitched 6.0 innings, allowing 5 hits, 3 Earned Runs, and 0 walk, while striking out 5.'}, {'speaker': 'Color Commentator', 'text': "Overall, a great performance from the Yankees, backed by strong pitching and timely hitting. The Orioles fought hard but couldn't quite match the Yankees' output."}]

    game_data2 = extract_game_data(script2)
    generate_storyboard_images(script2, game_data2)
