from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from typing import List, Dict, Optional
import io
import logging
from google.cloud import logging as cloud_logging
from google.cloud import secretmanager_v1
from google.cloud import storage
from google.oauth2 import service_account
import json
from vertexai.preview.vision_models import ImageGenerationModel
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
import json
import uuid
import time
import logging
from typing import List, Dict, Any
from google import genai
import enum
import asyncio
from moviepy import (
    ImageClip,
    CompositeVideoClip,
    concatenate_videoclips,
    AudioFileClip,
    CompositeAudioClip,
)
import io
import logging
import vertexai
import random
import re


import tempfile
import os
from PIL import Image
from io import BytesIO
import time
import logging
import re




PROJECT_ID = "gem-rush-007"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)


# Configure cloud logging at the top of the script, before other imports
logging.basicConfig(level=logging.INFO)
log_client = cloud_logging.Client()
log_client.setup_logging()


class MLBAudioMixer:
    def __init__(self, project_id, secret_name):
        service_account_json = self._get_secret(secret_name, project_id)
        if service_account_json:
            credentials = service_account.Credentials.from_service_account_info(
                json.loads(service_account_json),
                scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
        else:
            # Fallback error handling
            raise Exception("Could not retrieve service account credentials")

        # Initialize storage client
        self.storage_client = storage.Client(
            project=project_id,
            credentials=credentials
        )
        self.bucket_name = "mlb-audio-assets"

        # Sound effects and music paths
        sound_effect_paths = {
            "crowd_cheer": "assets/sounds/crowd_cheer.mp3",
            "bat_hit": "assets/sounds/bat_hit.mp3",
            "crowd_tension": "assets/sounds/crowd_tension.mp3",
            "walkup_music": "assets/sounds/walkup_music.mp3",
            "stadium_ambience": "assets/sounds/stadium_ambience.mp3"
        }
        
        music_paths = {
            "intro": "assets/music/opener.mp3",
            "highlight": "assets/music/highlight.mp3",
            "outro": "assets/music/opener.mp3"
        }

        # Load sound effects and music
        self.sound_effects = {
            name: self._load_audio_from_gcs(path) 
            for name, path in sound_effect_paths.items()
        }

        self.background_music = {
            name: self._load_audio_from_gcs(path) 
            for name, path in music_paths.items()
        }

        
        # Niveles de volumen refinados
        self.VOICE_VOLUME = 0
        self.MUSIC_VOLUME = -25
        self.SFX_VOLUME = -18
        self.AMBIENCE_VOLUME = -30
        
        # Constantes de tiempo (en milisegundos)
        self.SPEAKER_PAUSE = 850
        self.CROSSFADE_LENGTH = 400
        self.INTRO_FADE = 2000
        self.EFFECT_FADE = 600

    def _get_secret(self, secret_name, project_id):
        """Retrieve secret from Secret Manager"""
        try:
            client = secretmanager_v1.SecretManagerServiceClient()
            name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            service_account_json = response.payload.data.decode("UTF-8")
            # Parse and validate JSON
            credentials_dict = json.loads(service_account_json)
            required_fields = ['token_uri', 'client_email', 'private_key']
        
            for field in required_fields:
               if field not in credentials_dict:
                   raise ValueError(f"Missing required service account field: {field}")
        
            return service_account_json            
        except Exception as e:
            logging.error(f"Error retrieving secret {secret_name}: {e}")
            raise

    def _load_audio_from_gcs(self, blob_path):
        """Load audio file from GCS bucket"""
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_path)
            
            # Download audio content
            audio_content = blob.download_as_bytes()
            
            # Convert to AudioSegment
            audio_buffer = io.BytesIO(audio_content)
            return AudioSegment.from_mp3(audio_buffer)
        except Exception as e:
            logging.error(f"Error loading audio from {blob_path}: {e}")
            # Fallback to silent audio if loading fails
            return AudioSegment.silent(duration=1000)
      
    def _compress_audio(self, audio: AudioSegment) -> AudioSegment:
        logging.info("compressing audio")
        """Apply compression to prevent audio peaks and crackling."""
        return compress_dynamic_range(audio, threshold=-20.0, ratio=4.0, attack=10, release=100)

    def _fade_effect(self, effect: AudioSegment) -> AudioSegment:
        logging.info("fading effect")
        """Apply smooth fading to sound effects."""
        return effect.fade_in(200).fade_out(self.EFFECT_FADE)

    def _process_voice_segment(self, voice_audio: AudioSegment) -> AudioSegment:
        logging.info("processing voice segments")
        """Process voice segments with compression and normalization."""
        # First normalize to ensure consistent volume
        voice_audio = self._normalize_audio(voice_audio)
        # Apply compression to prevent peaks
        voice_audio = self._compress_audio(voice_audio)
        # Final volume adjustment
        return voice_audio - abs(self.VOICE_VOLUME)

    def mix_podcast_audio(self, voice_segments: List[Dict[str, bytes]], 
                         include_background: bool = True) -> AudioSegment:
        """
        Mix podcast audio with improved handling of high-intensity moments.
        """
        final_mix = self.background_music["intro"].fade_in(self.INTRO_FADE)
        final_mix = self._normalize_audio(final_mix)
        
        if include_background:
            ambience = self.sound_effects["stadium_ambience"]
            ambience = ambience - abs(self.AMBIENCE_VOLUME)
            total_duration = sum(len(AudioSegment.from_mp3(io.BytesIO(segment["audio"]))) 
                               for segment in voice_segments)
            total_duration += len(voice_segments) * self.SPEAKER_PAUSE
            
            while len(ambience) < total_duration:
                ambience += ambience
            
            final_mix = final_mix.overlay(ambience[:len(final_mix)])

        previous_speaker = None
        for i, segment in enumerate(voice_segments):
            audio_bytes = io.BytesIO(segment["audio"])
            voice_audio = AudioSegment.from_mp3(audio_bytes)
            
            # Process voice audio
            voice_audio = self._process_voice_segment(voice_audio)
            
            # Handle sound effects with improved timing
            triggers = self._detect_event_triggers(segment["text"])
            if triggers:
                # Create a blank segment for effects
                effect_mix = AudioSegment.silent(duration=len(voice_audio))
                
                for trigger in triggers:
                    effect = self.sound_effects[trigger]
                    effect = self._fade_effect(effect)
                    effect = effect - abs(self.SFX_VOLUME)
                    
                    # Position effect slightly before the voice for home runs
                    if trigger == "crowd_cheer" and "home run" in segment["text"].lower():
                        # Start effect earlier and let it fade under the voice
                        voice_audio = AudioSegment.silent(duration=200) + voice_audio
                        effect_position = 0
                    else:
                        effect_position = 100
                    
                    effect_mix = effect_mix.overlay(effect, position=effect_position)
                
                # Overlay effects onto voice with careful volume control
                voice_audio = voice_audio.overlay(effect_mix)
                # Apply additional compression to prevent peaks
                voice_audio = self._compress_audio(voice_audio)
            
            # Handle highlight background music
            if "highlight" in segment["text"].lower() and include_background:
                highlight_music = self.background_music["highlight"]
                highlight_music = highlight_music - abs(self.MUSIC_VOLUME)
                highlight_music = self._fade_effect(highlight_music)
                voice_audio = voice_audio.overlay(highlight_music[:len(voice_audio)])
            
            # Add pause between different speakers
            current_speaker = segment.get("speaker", "")
            if previous_speaker and previous_speaker != current_speaker:
                final_mix = self._add_pause(final_mix, self.SPEAKER_PAUSE)
            
            # Smooth out transitions
            voice_audio = voice_audio.fade_in(150).fade_out(300)
            
            # Mix with appropriate crossfade
            if i == 0:
                final_mix = final_mix.append(voice_audio, crossfade=self.INTRO_FADE)
            else:
                final_mix = final_mix.append(voice_audio, crossfade=self.CROSSFADE_LENGTH)
            
            previous_speaker = current_speaker
        
        # Add outro
        outro = self.background_music["outro"].fade_in(self.INTRO_FADE)
        final_mix = self._add_pause(final_mix, self.SPEAKER_PAUSE)
        final_mix = final_mix.append(outro, crossfade=self.INTRO_FADE)
        
        # Final compression pass on the complete mix
        final_mix = self._compress_audio(final_mix)
        
        return self.to_bytes(final_mix)

    def _add_pause(self, audio: AudioSegment, duration: int) -> AudioSegment:
        logging.info("adding pause")
        return audio + AudioSegment.silent(duration=duration)

    def _detect_event_triggers(self, text: str) -> List[str]:
        triggers = []
        events = {
            "crowd_cheer": ["home run", "scores", "wins", "victory", "walk off"],
            "bat_hit": ["hits", "singles", "doubles", "triples", "batting"],
            "crowd_tension": ["full count", "bases loaded", "bottom of the ninth"],
        }
        
        for effect, keywords in events.items():
            if any(keyword.lower() in text.lower() for keyword in keywords):
                triggers.append(effect)
                
        return triggers

    def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        logging.info("normalizing audio")
        return normalize(audio)

    def to_bytes(self, mixed_audio: AudioSegment) -> bytes:
        logging.info("converting audio to bytes")
        """Converts the mixed AudioSegment to bytes."""
        
        # Export the AudioSegment to a byte array in mp3 format
        buffer = io.BytesIO()
        mixed_audio.export(buffer, format="mp3")
         # Get the bytes from the buffer
        audio_bytes = buffer.getvalue()
        
        # Close the buffer
        buffer.close()
        
        return audio_bytes



class CloudVideoGenerator:
    def __init__(self, gcs_handler):
        self.gcs_handler = gcs_handler
        self.parent = f"projects/{gcs_handler.project_id}/locations/us-central1"

        # Initialize AI models with latest configurations
        logging.info("Initializing Vertex AI client and Imagen model")
        self.genai_client = genai.Client(vertexai=True, project="gem-rush-007", location="us-central1")
        self.imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")  # Use fast model
        self.MODEL_ID = "gemini-2.0-pro-exp-02-05"  # Use Gemini 2.0

        # Configure safety settings
        self.safety_config = [  # Use self.safety_config for consistency
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
        logging.info("CloudVideoGenerator initialized successfully.")

    def _analyze_script(self, script_data: list) -> Dict[str, Any]:
        """Analyze podcast script using Gemini 2.0 with enhanced configurations."""
        logging.info("Starting script analysis.")
        full_text = " ".join([segment['text'] for segment in script_data])
        logging.debug(f"Full script text: {full_text[:100]}...")

        # Use the system instruction and prompt directly (more efficient)
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
"""


        try:
            logging.info("Sending analysis request to Gemini 2.0 model.")
            response = self.genai_client.models.generate_content(
                model=self.MODEL_ID,
                contents=[{"role": "user", "parts": [{"text": full_text}]}],
                config=GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                    top_p=0.95,
                    top_k=20,
                    max_output_tokens=2048,
                    safety_settings=self.safety_config,
                    candidate_count=1,
                    seed=5,
                ),
            )
            logging.info("Received response from Gemini 2.0.")
            parsed_response = self._parse_gemini_response(response.text)
            return parsed_response

        except Exception as e:
            logging.error(f"Script analysis failed: {str(e)}")
            raise

    def _parse_gemini_response(self, raw_response: str) -> Dict[str, Any]:
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

   
    def _generate_images(self, analysis: Dict[str, Any], delay_seconds=30) -> List[bytes]:
        """Generates images for each key moment using the external function."""
        logging.info("Starting image generation for key moments.")
        images_data = []

        for idx, moment in enumerate(analysis['key_moments']):
            logging.info(f"Generating image for key moment {idx + 1}: {moment.get('description', 'No description')}")
            try:
                enhanced_prompt = self._enhance_prompt(moment['visual_prompt'])
                logging.debug(f"Enhanced prompt: {enhanced_prompt}")
                image_data = generate_and_save_image(enhanced_prompt, self.imagen_model, delay_seconds)
                if image_data:
                    images_data.append(image_data)
                else:
                    logging.warning(f"Image generation failed for moment {idx + 1}. Using default image.")
                    images_data.append(self._create_default_image())

            except Exception as e:
                logging.warning(f"An unexpected error occurred for moment {idx + 1}: {e}. Using default image.")
                images_data.append(self._create_default_image())

        return images_data

    def _create_default_image(self) -> bytes:
        """Creates a white default image with proper RGB format"""
        logging.info("Creating a default image as fallback.")
        # Create RGB image instead of RGBA to avoid alpha channel issues
        image = Image.new('RGB', (1920, 1080), (255, 255, 255))
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=85)
        return img_byte_arr.getvalue()

class MLBVideoGenerator:
    def __init__(self, audio_mixer, video_generator: CloudVideoGenerator):  # Removed type hint for audio_mixer
        self.audio_mixer = audio_mixer
        self.video_generator = video_generator
        logging.info("MLBVideoGenerator initialized.")

    def generate_video(self, script_data: List[Dict[str, Any]], include_background: bool = True) -> bytes:
        """
        Generates a video from a script.
        """
        logging.info("Starting video generation process.")

        analysis = self.video_generator._analyze_script(script_data)
        if not analysis or 'key_moments' not in analysis:
            logging.error("Script analysis failed or returned incomplete data.")
            raise ValueError("Script analysis failed.")

        images_data = self.video_generator._generate_images(analysis)
        if not images_data:
            logging.error("Image generation failed.")
            raise ValueError("Image generation failed.")

        image_clips = []
        for i, (image_bytes, moment) in enumerate(zip(images_data, analysis['key_moments'])):
            duration = moment.get('duration', 5)
            if image_bytes:
                try:
                    img_clip = ImageClip(image_bytes, duration=duration)  # Directly use bytes
                    image_clips.append(img_clip)
                except Exception as e:
                    logging.error(f"Failed to create ImageClip for moment {i+1}: {e}")
                    image_clips.append(ImageClip("black", duration=duration, color=(0, 0, 0)))
            else:  # Handle potential None (though _generate_images should use default)
                logging.warning(f"No image data for moment {i+1}. Using black clip.")
                image_clips.append(ImageClip("black", duration=duration, color=(0, 0, 0)))


        if not image_clips:
           logging.warning("No image clips available. Returning an empty video.")
           return b"" # Return empty bytes

        video_clip = concatenate_videoclips(image_clips, method="compose")
        logging.info(f"Video clip created with total duration: {video_clip.duration}s")

        logging.info("Mixing audio for the podcast.")
        mixed_audio_bytes = self.audio_mixer.mix_podcast_audio(script_data, include_background)
        if not mixed_audio_bytes:
           logging.warning("Audio mixing returned empty bytes. Proceeding without audio.")
           video_clip.write_videofile("output.mp4", fps=24, codec="libx264", audio=False)
           with open("output.mp4", "rb") as f:
              return f.read()

        audio_buffer = io.BytesIO(mixed_audio_bytes)
        audio_clip = AudioFileClip(audio_buffer)

        # Ensure video and audio durations match.
        if video_clip.duration > audio_clip.duration:
            logging.warning("Video is longer than audio. Trimming video.")
            video_clip = video_clip.subclip(0, audio_clip.duration)
        elif audio_clip.duration > video_clip.duration:
            logging.warning("Audio is longer than video.  Trimming audio.")
            audio_clip = audio_clip.subclip(0, video_clip.duration)

        video_clip = video_clip.set_audio(audio_clip)

        logging.info("Writing final video to file.")
        video_buffer = io.BytesIO()
        try:
            # Write to buffer and get bytes
            video_clip.write_videofile(video_buffer, fps=24, codec="libx264", audio_codec="aac", temp_audiofile="temp-audio2.m4a", remove_temp=True, verbose=False, logger=None)
            video_bytes = video_buffer.getvalue()

        except Exception as e:
              logging.error(f"Error writing video file: {str(e)}")
              raise
        finally:
            # Clean Up.
            video_clip.close()
            if 'audio_clip' in locals():
               audio_clip.close()
            for clip in image_clips:
               clip.close()
        return video_bytes

# ---  Standalone generate_and_save_image function (for completeness) ---
def generate_and_save_image(prompt, model, delay_seconds=30):
    try:
        response = model.generate_images(prompt=prompt)
        image = response.images[0]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
             temp_filename = f.name
             image.save(temp_filename)
             # the image is now saved into the temporary file.
             logging.info(f"Image for prompt '{prompt}' saved to temporary file: {temp_filename}")
             f.seek(0) #important
             image_data = f.read()
        os.remove(temp_filename)
        logging.info(f"Temporary file {temp_filename} removed.")

        logging.info(f"Waiting {delay_seconds} seconds before next prompt...")
        time.sleep(delay_seconds)
        return image_data

    except Exception as e:
        logging.error(f"Error generating or saving image for prompt '{prompt}': {e}")
        return None