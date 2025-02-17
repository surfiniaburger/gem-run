from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from typing import List, Dict, Optional
import io
import logging
from google.cloud import logging as cloud_logging
from google.cloud import secretmanager_v1
from google.cloud import storage
from google.oauth2 import service_account
import google.generativeai as genai
import json
from vertexai.preview.vision_models import ImageGenerationModel
import json
import uuid
import time
import logging
from typing import List, Dict, Any
import enum
from moviepy import (
    ImageClip,
    CompositeVideoClip,
    concatenate_videoclips,
    AudioFileClip,
    CompositeAudioClip,
)
import io
import logging
from google.cloud import secretmanager
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler

def setup_logging():
    """Sets up Google Cloud Logging."""
    client = google.cloud.logging.Client()
    handler = CloudLoggingHandler(client)
    logger = logging.getLogger('mongodb_vector_search')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

logger = setup_logging()

def get_secret(project_id, secret_id, version_id="latest", logger=None):
    """Retrieves a secret from Google Cloud Secret Manager."""
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        if logger:
            logger.error(f"Failed to retrieve secret {secret_id}: {str(e)}")
        raise

# Assuming you have your Google Cloud project ID set as an environment variable
PROJECT_ID = "gem-rush-007"  # Replace with your actual project ID
secret_id = "GEM-RUN-API-KEY"
apiKey = get_secret(PROJECT_ID, secret_id, logger=logger)

if apiKey:
  GOOGLE_API_KEY = apiKey
  genai.configure(api_key=GOOGLE_API_KEY)
  

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
        self.imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")
        

        logging.info("CloudVideoGenerator initialized successfully.")

    def _analyze_script(self, script_data: list) -> Dict[str, Any]:
        """Analyze podcast script using Gemini 2.0 with enhanced configurations."""
        logging.info("Starting script analysis.")
        full_text = " ".join([segment['text'] for segment in script_data])
        logging.debug(f"Full script text: {full_text[:100]}...")  # Log first 100 characters

        analysis_prompt = """Analyze this baseball podcast script and generate:
        {
            "key_moments": [{
                "timestamp": "HH:MM:SS",
                "description": "text",
                "visual_prompt": "Imagen prompt",
                "duration": 5,
                "transition": "fade/cut/zoom"
            }],
            "theme": "modern/retro/dramatic",
            "color_palette": {
                "primary": "#hex",
                "secondary": "#hex",
                "accent": "#hex"
            },
            "graphics_style": "dynamic/animated/static",
            "audio_intensity": 0-100
        }"""
        
        logging.info("Sending analysis request to Gemini 2.0 model.")
        MODEL_ID = "gemini-2.0-pro-exp-02-05"
        model = genai.GenerativeModel(
        model_name=MODEL_ID,
      
        system_instruction=analysis_prompt
         )
        chat = model.start_chat(enable_automatic_function_calling=False) 
        try:
           response = chat.send_message()
           logging.info("Received response from Gemini 2.0.")
           parsed_response = self._parse_gemini_response(response.text)
           logging.info("Script analysis completed successfully.")
           return parsed_response
        
        except Exception as e:
            logging.error(f"Script analysis failed: {str(e)}")
            raise

    def _parse_gemini_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse and validate Gemini response with error handling."""
        logging.info("Parsing Gemini response.")
        try:
            clean_text = raw_response.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(clean_text)
            logging.info("Successfully parsed Gemini response.")
            return parsed
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {str(e)}")
            raise ValueError("Failed to parse AI response")
        except Exception as e:
            logging.error(f"Unexpected parsing error: {str(e)}")
            raise

    def _generate_images(self, analysis: Dict[str, Any]) -> List[bytes]:
        logging.info("Starting image generation for key moments.")
        images = []
        for idx, moment in enumerate(analysis['key_moments']):
            logging.info(f"Generating image for key moment {idx + 1}: {moment.get('description', 'No description')}")
            try:
                # Add prompt engineering/enhancement
                enhanced_prompt = self._enhance_prompt(moment['visual_prompt'])
                logging.debug(f"Enhanced prompt: {enhanced_prompt}")
                
                # Add better retry strategy
                response = self._retry_with_backoff(
                    lambda: self.imagen_model.generate_images(
                        prompt=enhanced_prompt,
                        aspect_ratio="16:9",
                    )
                )
                
                if not response.images:
                    raise ValueError("Empty image response")
                    
                logging.info(f"Image generation successful for moment {idx + 1}.")
                images.append(response.images[0]._image_bytes)
                
            except Exception as e:
                logging.warning(f"Image generation failed for moment {idx + 1}: {str(e)}. Using default image.")
                default_image = self._create_default_image()
                images.append(default_image)
        logging.info("Completed image generation for all key moments.")
        return images

    def _retry_with_backoff(self, operation, max_retries=3, initial_delay=1):
        """Execute operation with exponential backoff retry logic.
    
        Args:
            operation: Callable to execute
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
        """
        logging.info("Starting retry with exponential backoff.")
        last_exception = None
        delay = initial_delay

        for attempt in range(max_retries):
            try:
                result = operation()
                logging.info(f"Operation succeeded on attempt {attempt + 1}.")
                return result
            except Exception as e:
                last_exception = e
                logging.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    logging.info(f"Waiting for {delay} seconds before retrying.")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
        logging.error("Maximum retry attempts reached. Operation failed.")
        raise last_exception

class MLBVideoGenerator:
    def __init__(self, audio_mixer: MLBAudioMixer, video_generator: CloudVideoGenerator):
        self.audio_mixer = audio_mixer
        self.video_generator = video_generator
        logging.info("MLBVideoGenerator initialized.")

    def generate_video(self, script_data: List[Dict[str, Any]], include_background: bool = True) -> bytes:
        """
        Generates a video from a script, using Imagen for images and MoviePy for video editing.

        Args:
            script_data: List of dictionaries, each containing 'text', 'speaker', and 'audio' (as bytes).
            include_background: Whether to include background audio/music.

        Returns:
            Bytes representing the final video file (MP4).
        """
        logging.info("Starting video generation process.")

        logging.info("Analyzing script for key moments and visual prompts.")
        analysis = self.video_generator._analyze_script(script_data)
        if not analysis or 'key_moments' not in analysis:
            logging.error("Script analysis failed or returned incomplete data.")
            raise ValueError("Script analysis failed.")

        logging.info("Generating images based on analysis.")
        images = self.video_generator._generate_images(analysis)
        if not images:
            logging.error("Image generation failed.")
            raise ValueError("Image generation failed.")

        logging.info("Creating image clips from generated images.")
        image_clips = []
        for i, (image_bytes, moment) in enumerate(zip(images, analysis['key_moments'])):
            duration = moment.get('duration', 5)  # Default to 5 seconds if not specified
            logging.info(f"Creating ImageClip for moment {i+1} with duration {duration}s.")
            try:
                img_clip = ImageClip(io.BytesIO(image_bytes), duration=duration)
                image_clips.append(img_clip)
            except Exception as e:
                logging.error(f"Failed to create ImageClip for moment {i+1}: {e}")
                #  Fallback:  Create a 5-second black clip.  Important for error handling.
                image_clips.append(ImageClip("black", duration=duration, color=(0, 0, 0)))

        logging.info("Concatenating image clips to create video sequence.")

        if not image_clips:
           logging.warning("No image clips available. Returning an empty video.")
           return b"" # Return empty bytes to indicate no video generated

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

        # Ensure video and audio durations match. VERY IMPORTANT.
        if video_clip.duration > audio_clip.duration:
            logging.warning("Video is longer than audio. Trimming video.")
            video_clip = video_clip.subclip(0, audio_clip.duration)
        elif audio_clip.duration > video_clip.duration:
            logging.warning("Audio is longer than video.  Trimming audio.")
            audio_clip = audio_clip.subclip(0, video_clip.duration)

        logging.info("Setting audio to video clip.")
        video_clip = video_clip.set_audio(audio_clip)

        logging.info("Writing final video to file.")
        video_buffer = io.BytesIO()
        try:
            video_clip.write_videofile("output.mp4", fps=24, codec="libx264", audio_codec="aac", temp_audiofile="temp-audio.m4a", remove_temp=True) #Added temporary audiofile
            with open("output.mp4", "rb") as f:
               video_bytes = f.read()

        except Exception as e:
              logging.error(f"Error writing video file: {str(e)}")
              raise
        finally:
            # Clean Up: Close all clips to release resources
            video_clip.close()
            if 'audio_clip' in locals():
               audio_clip.close()
            for clip in image_clips:
               clip.close()
               
        logging.info("Video generation completed successfully.")

        return video_bytes