# Copyright 2025 GEM RUN
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

"""
MLB Podcast Visual Asset Generator
This script creates visual assets for MLB podcast episodes using Imagen and Gemini.
"""

import sys
import os
import json
import math
import matplotlib.pyplot as plt
from google import genai
from google.genai import types
from IPython.display import Markdown, display

# --- Authentication and Setup ---


# Project and Location Setup
PROJECT_ID = "gem-rush-007" 


LOCATION = "us-central1"

# Initialize the Vertex AI client
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# --- Model Definitions ---

imagen_model = "imagen-3.0-generate-002"
gemini_model = "gemini-2.0-flash-001"  # Or a more recent version if available


# --- Helper Functions ---

def display_images_in_grid(images):
    """Displays a list of images in a grid."""
    if not images:
      print("No images to display.")
      return

    nrows = math.ceil(len(images) / 4)
    ncols = min(len(images) + 1, 4)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6))

    # Flatten axes array for easy iteration, handling single image case
    if len(images) == 1:
        axes = [axes] # Make it iterable even with one image.
    else:
        axes = axes.flat


    for i, ax in enumerate(axes):
        if i < len(images):
            try:
                ax.imshow(images[i].image._pil_image)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
            except AttributeError:
                print(f"Error displaying image {i+1}.  It may not be a valid image object.")
                ax.axis("off") # Turn off axis even for error.
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()



def generate_content(prompt, temperature=0.5, top_p=0.8, top_k=10, candidate_count=1, max_output_tokens=2048): # Increased max_output_tokens
    """Generates text content using Gemini."""
    generation_config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        candidate_count=candidate_count,
        max_output_tokens=max_output_tokens,
    )

    responses = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
        config=generation_config,
    )
    return responses.text


def generate_image(prompt, num_images=1):
    """Generates images using Imagen."""
    try:
        response = client.models.generate_image(
            model=imagen_model,
            prompt=prompt,
            config=types.GenerateImageConfig(
                number_of_images=num_images,
            ),
        )
        return response.generated_image
    except Exception as e:
        print(f"Error generating image: {e}")
        return []



# --- Storyboard Prompts (Combined and Refactored) ---

storyboard_prompts = {
    "general": {
        "G1": "Wide, panoramic view of a bustling baseball stadium filled with fans. It's {time_of_day}. The field is perfectly manicured, and the stadium lights are {lights_state}. Photorealistic, high detail, {camera_angle}",
        "G2": "Inside the announcer's booth. Two announcers, {announcer_1} and {announcer_2}, are sitting at a desk with microphones and monitors. They are {announcer_activity}. Realistic, well-lit.",
        "G3": "Close-up of a digital scoreboard graphic showing the team logos, the score, the inning, the outs, the count (balls and strikes), and the date. {team_a} vs. {team_b}. Score: {score_a}-{score_b}. Inning: {inning_top_bottom} {inning_number}. Outs: {outs}. Count: {balls}-{strikes}. Date: {date}. Clean, modern design.",
        "G4": "A medium shot showcasing {dugout_focus} dugout. Players are {player_activity}. The atmosphere is {dugout_atmosphere}. Show details like bats, helmets, and gloves.",
        "G5": "A wide shot of the crowd, showing a section. Include details of people cheering, wearing hats, and drinking.",
    },
    "hitting": {
        "home_run": {
            "HR1": "A batter at home plate, in a {team_name} uniform, {batter_setup}. The pitcher is on the mound, ready to deliver. The catcher is in position. Tense atmosphere. Photorealistic.",
            "HR2": "Close-up on the pitcher, mid-wind-up. Focus on the intense expression, the grip on the ball, and the details of the uniform. Motion blur on the arm. Dramatic lighting.",
            "HR3": "The batter, mid-swing, connecting powerfully with the ball. Show the bat impacting the ball. Slight motion blur on the bat. The catcher is reacting. Dynamic angle, {camera_angle}.",
            "HR4": "The baseball, soaring high above the field, {trail_option}. The outfielders are {outfielder_reaction}. Wide shot of the stadium, or a tracking shot following the ball.",
            "HR5": "Fans in the stands {crowd_reaction}. A mix of excitement and awe on their faces. Show a diverse crowd.",
            "HR6": "The batter, {batter_rounding}, {batter_expression}. Slight blur on the background. Tracking shot following the batter.",
            "HR7": "The batter is crossing the home plate, as their teammates celebrate with them at the plate."
        },
        "base_hit": {
            "BH1": "Close-up of the bat hitting the ball. No flames, just a solid connection. Show the {bat_detail}. Sharp focus.",
            "BH2": "The baseball, {ball_in_play}. {fielder_reaction}. Motion blur on the ball. {hit_type}",
            "BH3": "The batter, sprinting towards first base, head down, determined. Slight motion blur on the legs. Tracking shot following the runner.",
            "BH4a": "A close play at {base}. The runner {runner_action} *feet first/head first*, the fielder {fielder_action}. Dust flying. Intense focus. Show the umpire's 'out' signal clearly.",
            "BH4b": "A close play at {base}. The runner {runner_action} *feet first/head first*, the fielder {fielder_action}. Dust flying. Intense focus. Show the umpire's 'safe' signal clearly.",
            "BH5": "The runner standing on the base, safe. With the defensive players on the background"
        },
        "strikeout": {
            "SO1": "The batter at the plate, looking {batter_look}. The count is 3-2. The catcher is giving signs. Focus on the batter's eyes and grip on the bat. Show the runners on base (if any).",
            "SO2": "Close-up of the pitcher's hand releasing the ball. Motion blur. Focus on the grip and the spin on the ball. Tracking shot following the ball. {pitch_type}",
            "SO3a": "The batter swings and misses. The bat whips through the air. The catcher catches the ball cleanly. Dejected expression on the batter's face.",
            "SO3b": "The batter {batter_reaction_called}. The umpire makes the 'strike three' call emphatically. The catcher holds the ball.",
            "SO4": "Close-up of the catcher, {catcher_reaction}. Confident expression.",
            "SO5": "The batter walking away, {batter_walking}. The catcher and umpire are in the background.",
        },
        "walk":{
            "WK1": "Close-up of the umpire's hand signaling ball four. The pitch is {pitch_location}.",
            "WK2": "The batter, {batter_reaction_walk}. A {batter_expression_walk} expression.",
            "WK3": "Close-up on the pitcher, {pitcher_reaction_walk}. Show frustration or disappointment.",
            "WK4": "The batter is walking to first base, on the background the pitcher and the catcher are showing frustration."
        },

  "on_base_percentage": {
            "OBP1": "Montage of different successful at-bats for {player_name}: a walk, a single, a hit-by-pitch, a double.  Fast-paced editing.",  # This is a *text* prompt for Gemini.
            "OBP2": "A graphic showing {player_name}'s OBP statistic, prominently displayed. Clean, modern design.  Include team logo and photo. Compare OBP to league average.",
        },
    },
   "pitching": {
        "dominant_performance": {
            "DP1": "The pitcher, {pitcher_name}, standing on the mound, looking {pitcher_look}. The stadium is in the background. Slight low angle. Show the scoreboard (optional).",
            "DP2": "Series of close-ups showing different grips and releases for {pitcher_name}: fastball, curveball, slider, changeup, knuckleball (if applicable). Motion blur. Use tracking shots.", # This is a *text* prompt for Gemini.
            "DP3": "Montage of batters striking out against {pitcher_name}. Fast-paced editing. Vary camera angles, show different pitches.", # This is a *text* prompt for Gemini
            "DP4": "Close-up of the pitcher, {pitcher_name}, after a key strikeout. {pitcher_reaction_so}.",
            "DP5": "A close-up of the screen/scoreboard, showing {pitcher_name}'s name and a low ERA. Show other stats (strikeouts, innings).",
        },
        "struggling_performance": {
            "SP1": "The pitcher, {pitcher_name}, on the mound, looking {pitcher_look_struggling}. The pitching coach or manager is walking out. Concerned expressions.",
            "SP2": "Montage of batters hitting the ball hard against {pitcher_name}: line drives, doubles, home runs. Focus on impact and ball flight.", # This is a *text* prompt for Gemini
            "SP3": "Close-up of the pitcher, {pitcher_name}, {pitcher_reaction_struggling}.",
            "SP4": "The manager walking out to the mound and signaling to the bullpen. The pitcher, {pitcher_name}, {pitcher_pulled}.",
        },
    },
  "fielding": {
        "great_play": {
            "GP1": "A {hit_type} heading towards {fielding_position}. Show the batter's reaction.",
            "GP2": "The fielder, {player_name}, making a {play_type}. Motion blur. Tracking shot.",
            "GP3": "Close-up of the glove making the catch. The ball is securely in the webbing. {debris_flying}. Show the fielder's focus.",
            "GP4": "The fielder, {player_name}, {celebration_type}.",
            "GP5": "The fielder showing the ball after catching it."
        },
    },
    "quotes": {
        "PQ1": "A close-up of {player_name}'s face during an {interview_type}.  Expressive eyes, microphone visible. Show emotion: {emotion}.",
        "PQ2": "A clean, modern graphic overlay with {player_name}'s quote: '{quote}'. Legible font, contrasting colors. Include name, team logo, and date.",
        "PQ3": "Replay of the play related to {player_name}'s quote: '{quote}'. Show the player making a relevant gesture.",  # This is a *text* prompt for Gemini
        "PQ4": "Close up of the reporter making the question"
    },
     "key_players_statistics":{
        "KP1": "Close up of {player_name} name on the back of the jersey",
        "KP2": "The player {player_name} {player_action}",
        "KP3": "The Player {player_name} running the bases",
        "KP4": "A display showing relevant stats for {player_name}: {relevant_stats}"

     }

}

# --- Script Processing and Visualization ---

def process_script(script):
    """Processes a game recap script and generates visual assets."""
    for segment in script:
        speaker = segment['speaker']
        text = segment['text']
        print(f"Processing segment: Speaker - {speaker}, Text - {text}")

        if speaker == "Play-by-play Announcer" or speaker == "Color Commentator":
          # Extract key game events and information.  Use Gemini for more complex analysis.
            if "home run" in text.lower():
                # Home Run Sequence
                batter, team_name = extract_player_and_team(text)
                if batter and team_name:

                    #HR1
                    prompt_values = {
                        "team_name": team_name,
                        "batter_setup": get_batter_setup_description(),
                        "camera_angle": "low angle looking up"
                    }

                    image_prompt_hr1 = storyboard_prompts["hitting"]["home_run"]["HR1"].format(**prompt_values)
                    images = generate_image(image_prompt_hr1)

                    if(len(images) > 0):
                        display_images_in_grid(images)

                    #HR2
                    prompt_values = {
                        "team_name": team_name,
                        "batter_setup": get_batter_setup_description(),
                        "camera_angle": "low angle looking up"
                    }
                    image_prompt_hr2 = storyboard_prompts["hitting"]["home_run"]["HR2"].format(**prompt_values)
                    images = generate_image(image_prompt_hr2)

                    if(len(images) > 0):
                      display_images_in_grid(images)

                    #HR3

                    prompt_values = {
                        "team_name": team_name,
                        "batter_setup": get_batter_setup_description(),
                        "camera_angle": "low angle looking up"
                    }

                    image_prompt_hr3 = storyboard_prompts["hitting"]["home_run"]["HR3"].format(**prompt_values)
                    images = generate_image(image_prompt_hr3)

                    if(len(images) > 0):
                      display_images_in_grid(images)

                    #HR4
                    prompt_values = {
                      "trail_option": "leaving a faint vapor trail",  # or "a fiery trail"
                      "outfielder_reaction": "looking up, tracking the ball",  # or "giving up on the play"
                    }
                    image_prompt_hr4 = storyboard_prompts["hitting"]["home_run"]["HR4"].format(**prompt_values)
                    images = generate_image(image_prompt_hr4)

                    if(len(images) > 0):
                      display_images_in_grid(images)

                    #HR5

                    prompt_values = {
                        "crowd_reaction": "jumping to their feet, cheering and pointing at the sky",

                    }

                    image_prompt_hr5 = storyboard_prompts["hitting"]["home_run"]["HR5"].format(**prompt_values)
                    images = generate_image(image_prompt_hr5)

                    if(len(images) > 0):
                      display_images_in_grid(images)

                    #HR6
                    prompt_values = {
                        "batter_rounding": "rounding first base",
                        "batter_expression": "watching the ball fly with a confident expression"
                    }

                    image_prompt_hr6 = storyboard_prompts["hitting"]["home_run"]["HR6"].format(**prompt_values)
                    images = generate_image(image_prompt_hr6)

                    if(len(images) > 0):
                      display_images_in_grid(images)

                    #HR7

                    image_prompt_hr7 = storyboard_prompts["hitting"]["home_run"]["HR7"]
                    images = generate_image(image_prompt_hr7)

                    if(len(images) > 0):
                      display_images_in_grid(images)

            elif "single" in text.lower() or "double" in text.lower() or "base hit" in text.lower():
                #Base Hit Sequence
                batter, team_name = extract_player_and_team(text)
                base = "first base"
                if "double" in text.lower():
                    base = "second base"
                if batter and team_name:
                    # Use BH prompts, filling in details.
                    prompt_values = {
                      "bat_detail" : "wood grain of the bat",
                      "ball_in_play": "bouncing through the infield",
                      "fielder_reaction": "are reacting, moving to field the ball",
                      "hit_type": "a ground ball to short",
                      "base": base,
                      "runner_action": "sliding",
                      "fielder_action": "stretching to make the catch and tag"
                    }

                    #BH1
                    image_prompt_bh1 = storyboard_prompts["hitting"]["base_hit"]["BH1"].format(**prompt_values)
                    images = generate_image(image_prompt_bh1)

                    if(len(images) > 0):
                        display_images_in_grid(images)

                    #BH2
                    image_prompt_bh2 = storyboard_prompts["hitting"]["base_hit"]["BH2"].format(**prompt_values)
                    images = generate_image(image_prompt_bh2)

                    if(len(images) > 0):
                        display_images_in_grid(images)

                    #BH3
                    image_prompt_bh3 = storyboard_prompts["hitting"]["base_hit"]["BH3"].format(**prompt_values)
                    images = generate_image(image_prompt_bh3)

                    if(len(images) > 0):
                        display_images_in_grid(images)

                    #BH4
                    # Choose BH4a (out) or BH4b (safe) based on further text analysis, or randomly.
                    # For this example, let's assume safe.

                    image_prompt_bh4 = storyboard_prompts["hitting"]["base_hit"]["BH4b"].format(**prompt_values)
                    images = generate_image(image_prompt_bh4)
                    if(len(images) > 0):
                        display_images_in_grid(images)

            elif "strikeout" in text.lower():
               #Strikeout Sequence
                pitcher, team_name = extract_player_and_team(text)
                #SO prompts

                prompt_values = {
                    "batter_look": "tense",
                    "pitch_type": "a blazing fastball",
                    "batter_reaction_called": "watches the pitch go by",
                    "catcher_reaction": "pumping his fist",
                    "batter_walking": "helmet in hand and the catcher celebrating behind him with the umpire"
                }

                #SO1

                image_prompt_so1 = storyboard_prompts["hitting"]["strikeout"]["SO1"].format(**prompt_values)
                images = generate_image(image_prompt_so1)
                if(len(images) > 0):
                    display_images_in_grid(images)

                #SO2

                image_prompt_so2 = storyboard_prompts["hitting"]["strikeout"]["SO2"].format(**prompt_values)
                images = generate_image(image_prompt_so2)
                if(len(images) > 0):
                    display_images_in_grid(images)

                #SO3
                #Choose SO3a or SO3b
                image_prompt_so3 = storyboard_prompts["hitting"]["strikeout"]["SO3a"].format(**prompt_values)
                images = generate_image(image_prompt_so3)

                if(len(images) > 0):
                    display_images_in_grid(images)

                #SO4

                image_prompt_so4 = storyboard_prompts["hitting"]["strikeout"]["SO4"].format(**prompt_values)
                images = generate_image(image_prompt_so4)
                if(len(images) > 0):
                    display_images_in_grid(images)

                #SO5
                image_prompt_so5 = storyboard_prompts["hitting"]["strikeout"]["SO5"].format(**prompt_values)
                images = generate_image(image_prompt_so5)
                if(len(images) > 0):
                    display_images_in_grid(images)
            elif "walk" in text.lower() or "base on balls" in text.lower():
                pitcher, team_name = extract_player_and_team(text)

                prompt_values = {
                    "pitch_location": "clearly outside the strike zone",
                    "batter_reaction_walk": "dropping his bat and starting to jog to first base",
                    "batter_expression_walk": "relieved",
                    "pitcher_reaction_walk": "taking his hat off"
                }

                #WK1
                image_prompt_wk1 = storyboard_prompts["hitting"]["walk"]["WK1"].format(**prompt_values)
                images = generate_image(image_prompt_wk1)
                if(len(images) > 0):
                    display_images_in_grid(images)

                #WK2
                image_prompt_wk2 = storyboard_prompts["hitting"]["walk"]["WK2"].format(**prompt_values)
                images = generate_image(image_prompt_wk2)
                if(len(images) > 0):
                    display_images_in_grid(images)

                #WK3
                image_prompt_wk3 = storyboard_prompts["hitting"]["walk"]["WK3"].format(**prompt_values)
                images = generate_image(image_prompt_wk3)
                if(len(images) > 0):
                    display_images_in_grid(images)

                #WK4
                image_prompt_wk4 = storyboard_prompts["hitting"]["walk"]["WK4"].format(**prompt_values)
                images = generate_image(image_prompt_wk4)
                if(len(images) > 0):
                    display_images_in_grid(images)
            elif "pitched" in text.lower():
                #Dominant or Struggling Pitcher
                pitcher, team_name = extract_player_and_team(text)

                if(pitcher):
                    if "struggled" in text.lower() or "ERA" in text.lower():
                      #Struggling Pitcher

                        prompt_values = {
                            "pitcher_name": pitcher,
                            "pitcher_look_struggling": "dejected",
                            "pitcher_reaction_struggling": "shaking his head",
                            "pitcher_pulled": "handing over the ball and walking off the field."
                        }
                        #SP1
                        image_prompt_sp1 = storyboard_prompts["pitching"]["struggling_performance"]["SP1"].format(**prompt_values)
                        images = generate_image(image_prompt_sp1)

                        if(len(images) > 0):
                            display_images_in_grid(images)

                        #SP2
                        #Montage

                        #SP3
                        image_prompt_sp3 = storyboard_prompts["pitching"]["struggling_performance"]["SP3"].format(**prompt_values)
                        images = generate_image(image_prompt_sp3)

                        if(len(images) > 0):
                            display_images_in_grid(images)

                        #SP4
                        image_prompt_sp4 = storyboard_prompts["pitching"]["struggling_performance"]["SP4"].format(**prompt_values)
                        images = generate_image(image_prompt_sp4)

                        if(len(images) > 0):
                            display_images_in_grid(images)
                    else:
                      #Dominant Pitcher

                        prompt_values = {
                            "pitcher_name": pitcher,
                            "pitcher_look": "confident",
                            "pitcher_reaction_so": "subtle fist pump"
                        }
                        #DP1
                        image_prompt_dp1 = storyboard_prompts["pitching"]["dominant_performance"]["DP1"].format(**prompt_values)
                        images = generate_image(image_prompt_dp1)

                        if(len(images) > 0):
                            display_images_in_grid(images)

                        #DP2
                        #Montage

                        #DP3
                        #Montage
                        #DP4
                        image_prompt_dp4 = storyboard_prompts["pitching"]["dominant_performance"]["DP4"].format(**prompt_values)
                        images = generate_image(image_prompt_dp4)

                        if(len(images) > 0):
                            display_images_in_grid(images)

                        #DP5
                        image_prompt_dp5 = storyboard_prompts["pitching"]["dominant_performance"]["DP5"].format(**prompt_values)
                        images = generate_image(image_prompt_dp5)

                        if(len(images) > 0):
                            display_images_in_grid(images)
            else:
                # General game atmosphere (G prompts). Use Gemini to decide which G prompt is most relevant.
                prompt_for_g = f"Given the following text from a baseball game recap, which of the following 'general' storyboard prompts is MOST relevant?  Answer with the prompt code (e.g., 'G1', 'G2') ONLY.\n\nText: {text}\n\nPrompts:\n"
                for code, prompt_text in storyboard_prompts['general'].items():
                    prompt_for_g += f"{code}: {prompt_text}\n"

                relevant_g_prompt_code = generate_content(prompt_for_g, max_output_tokens=5).strip()

                if relevant_g_prompt_code in storyboard_prompts['general']:
                    # Fill in the placeholders in the chosen G prompt.  This is a bit tricky, as the G prompts
                    # have different placeholders.  We'll need a helper function.
                    prompt_values = fill_general_prompt_values(text, relevant_g_prompt_code)
                    image_prompt = storyboard_prompts['general'][relevant_g_prompt_code].format(**prompt_values)
                    images = generate_image(image_prompt)
                    if(len(images) > 0):
                        display_images_in_grid(images)
                else:
                    print(f"Invalid G prompt code: {relevant_g_prompt_code}")


        elif speaker == "Player Quotes":
            # Player Quote sequence (PQ prompts)
            player_name = extract_player_name(text)
            if player_name:
                prompt_values = {
                    "player_name": player_name,
                    "interview_type": "post-game press conference",  # or "on-field interview"
                    "emotion": get_emotion_from_quote(text),  # Use Gemini to determine emotion
                    "quote": text,
                }
                image_prompt_pq1 = storyboard_prompts["quotes"]["PQ1"].format(**prompt_values)

                images = generate_image(image_prompt_pq1)
                if(len(images)>0):
                    display_images_in_grid(images)

                #PQ2 use text directly.

        elif speaker == "Introduction" or speaker == "General Attribution":

            # General game atmosphere (G prompts). Use Gemini to decide which G prompt is most relevant.
            prompt_for_g = f"Given the following text from a baseball game recap, which of the following 'general' storyboard prompts is MOST relevant?  Answer with the prompt code (e.g., 'G1', 'G2') ONLY.\n\nText: {text}\n\nPrompts:\n"
            for code, prompt_text in storyboard_prompts['general'].items():
                prompt_for_g += f"{code}: {prompt_text}\n"

            relevant_g_prompt_code = generate_content(prompt_for_g, max_output_tokens=5).strip()

            if relevant_g_prompt_code in storyboard_prompts['general']:
                # Fill in the placeholders in the chosen G prompt.  This is a bit tricky, as the G prompts
                # have different placeholders.  We'll need a helper function.
                prompt_values = fill_general_prompt_values(text, relevant_g_prompt_code)
                image_prompt = storyboard_prompts['general'][relevant_g_prompt_code].format(**prompt_values)
                images = generate_image(image_prompt)
                if(len(images) > 0):
                    display_images_in_grid(images)
            else:
                print(f"Invalid G prompt code: {relevant_g_prompt_code}")
        # Add more conditions for other speakers (e.g., "Statistician") and event types as needed.

def fill_general_prompt_values(text, prompt_code):
    """Fills in the placeholder values for the general (G) prompts."""
    values = {}

    if prompt_code == "G1":
        # Use Gemini to extract time of day, lighting conditions.
        time_of_day_prompt = f"Given this text: '{text}', what time of day is it?  Answer concisely (e.g., 'early afternoon', 'night')."
        values["time_of_day"] = generate_content(time_of_day_prompt, max_output_tokens=10)
        lights_prompt = f"Given this text: '{text}', are the stadium lights on bright, or casting long shadows? Answer concisely (e.g. 'bright', 'casting long shadows')."
        values["lights_state"] = generate_content(lights_prompt, max_output_tokens=10)
        values["camera_angle"] = "high angle" #default

    elif prompt_code == "G2":
        # Extract announcer names (this might require some clever text processing or named entity recognition).
        # For simplicity, we'll use placeholders.
        values["announcer_1"] = "John Doe"
        values["announcer_2"] = "Jane Smith"
        values["announcer_activity"] = "looking intently at the field"

    elif prompt_code == "G3":
        # Extract game information. This requires careful parsing of the text.
        # For simplicity, we'll use placeholders here, but in a real application, you'd extract this data.
        values["team_a"] = "Team A"
        values["team_b"] = "Team B"
        values["score_a"] = "0"
        values["score_b"] = "0"
        values["inning_top_bottom"] = "Top"
        values["inning_number"] = "1"
        values["outs"] = "0"
        values["balls"] = "0"
        values["strikes"] = "0"
        values["date"] = "October 9, 2024"

    elif prompt_code == "G4":
        values["dugout_focus"] = "home team's"
        values["player_activity"] = "sitting on the bench"
        values["dugout_atmosphere"] = "tense"

    return values

def extract_player_and_team(text):
    """Extracts player and team names from the text (Simplified)."""
    prompt = f"""From the following text, extract the name of ONE player mentioned and ONE team name,
                it does not matter if they are home or away. If the player and the team are not in the same sentence select one at random. Return the answer in JSON format.
                Do not include JSON decorators. The response should start with an opening curly brace.
                The parent fields should be player, team.
                Text: {text}"""
    response = generate_content(prompt)
    try:
        json_response = json.loads(response)
        return json_response["player"], json_response["team"]
    except:
        return None, None

def extract_player_name(text):
    """Extract player name for quote attribution (Simplified)."""
    # Ideally, use a named entity recognition model here.
    prompt = f"""From the following text, extract the name of ONE player mentioned. Return the answer in JSON format.
            Do not include JSON decorators. The response should start with an opening curly brace.
            The parent fields should be player.
            Text: {text}"""
    response = generate_content(prompt)
    try:
        json_response = json.loads(response)
        return json_response["player"]
    except:
        return None

def get_emotion_from_quote(text):
    """Uses Gemini to determine the emotion expressed in a quote."""
    prompt = f"Analyze the following quote and describe the speaker's emotion in one word (e.g., 'happy', 'frustrated', 'determined'). Quote: '{text}'"
    emotion = generate_content(prompt, max_output_tokens=5).strip()
    return emotion

def get_batter_setup_description():
    """Gets batter setup description"""
    prompt = f"Describe in one word how a batter sets up at the plate to bat. Examples: digging into the batter box, taking a practice swing, adjusting his helmet"
    return generate_content(prompt, max_output_tokens=10).strip()



script3 = [{'speaker': 'Introduction', 'text': "Welcome to today's podcast! Today is October 25, 2024. We'll be discussing the Minnesota Twins' last game. All game data and statistics are sourced from the MLB Stats API."}, {'speaker': 'Play-by-play Announcer', 'text': 'The Twins played their most recent game against the Chicago White Sox on May 12, 2024. According to the MLB Stats API, the final score was 4-0 in favor of the White Sox.'}, {'speaker': 'Color Commentator', 'text': 'This was a tough loss for the Twins, who were shut out by the White Sox. The starting pitcher for Minnesota was Bailey Ober.'}, {'speaker': 'Play-by-play Announcer', 'text': "For the White Sox, Erick Fedde started on the mound. Let's take a closer look at a key moment in the game."},
{'speaker': 'Play-by-play Announcer', 'text': 'In the bottom of the 1st inning, with the score still 0-0, Andrew Benintendi hit a home run. He was facing Twins pitcher Bailey Ober. The play brought the score to 1-0, White Sox.'}, {'speaker': 'Color Commentator', 'text': "That early home run by Benintendi really set the tone for the White Sox. It gave them a lead they wouldn't relinquish. Bailey Ober, the Twins' starter had a rough start there."}, {'speaker': 'Player Quotes', 'text': "I just didn't execute that pitch, and Benintendi made me pay. It was a mistake, and it cost us a run early. I needed to be better in that situation."}, {'speaker': 'Play-by-play Announcer', 'text': 'In the bottom of the 5th, Paul DeJong hit a home run, with the score now 2-0. This increased the White Sox lead to 3-0.'}, {'speaker': 'Color Commentator', 'text': "Paul DeJong's home run further extended the White Sox lead, giving them a comfortable cushion. The data from the MLB Stats API shows this was a crucial point, putting pressure on the Twins' offense."}, {'speaker': 'Player Quotes', 'text': 'As a pitcher you never want to give those up, especially with a two run lead. Makes the climb for the team that much steeper. I just wanted to have a lock down inning. So from that standpoint, that homer really hurt.'}, {'speaker': 'Play-by-play Announcer', 'text': "Let's go over Bailey Ober's stats, who was the starting pitcher for the Twins. According to the MLB Stats API, Ober pitched 5.0 innings, allowing 6 hits, 4 runs, and 4 earned runs, with 1 walk and 5 strikeouts. His ERA for the game was 7.20."}, {'speaker': 'Color Commentator', 'text': 'Ober definitely struggled. Allowing 4 runs in 5 innings is not the performance the Twins were hoping for.'}, {'speaker': 'Play-by-play Announcer', 'text': "Now, let's review Erick Fedde, the starter for the White Sox. Fedde pitched 5.2 innings, giving up just 2 hits and no runs. He walked 2 and struck out 6, with a game ERA of 0.00."}, {'speaker': 'Color Commentator', 'text': "Fedde had an outstanding performance, effectively shutting down the Twins' offense. His 0.00 ERA for the game really highlights his dominance."}, {'speaker': 'Play-by-play Announcer', 'text': 'For the Twins, key players like Carlos Correa went 0 for 4. Byron Buxton went 1 for 4.'}, {'speaker': 'Color Commentator', 'text': "The Twins' offense really struggled to get anything going against Fedde and the White Sox bullpen."}, {'speaker': 'Play-by-play Announcer', 'text': 'For the White Sox, Andrew Benintendi went 2 for 4 with a home run and 2 RBIs. Paul DeJong went 1 for 4 with a home run and 2 RBIs.'}, {'speaker': 'Color Commentator', 'text': 'Benintendi and DeJong provided the necessary fire power to defeat the Twins.'}]
# --- Example Usage ---
# Select one of the scripts to process.  For this example, we'll use script1.
process_script(script3)
#You can process here other scripts like:
#process_script(script2)
#process_script(script3)