import os
import io
import json
import logging
import time
import uuid
from pydub import AudioSegment
from vertexai.preview.vision_models import ImageGenerationModel
from pydub.effects import normalize, compress_dynamic_range
from google.cloud import logging as cloud_logging
from google.cloud import secretmanager_v1, storage
from google.oauth2 import service_account
from google import genai
from google.genai import types
from PIL import Image
from moviepy import *

from typing import List, Dict

# Configure Cloud Logging
logging.basicConfig(level=logging.INFO)
log_client = cloud_logging.Client()
log_client.setup_logging()

##############################################
# Existing MLBAudioMixer (as you provided)  #
##############################################

class MLBAudioMixer:
    def __init__(self, project_id, secret_name):
        service_account_json = self._get_secret(secret_name, project_id)
        if service_account_json:
            credentials = service_account.Credentials.from_service_account_info(
                json.loads(service_account_json),
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
        else:
            raise Exception("Could not retrieve service account credentials")
        self.storage_client = storage.Client(project=project_id, credentials=credentials)
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
        self.sound_effects = {
            name: self._load_audio_from_gcs(path) 
            for name, path in sound_effect_paths.items()
        }
        self.background_music = {
            name: self._load_audio_from_gcs(path) 
            for name, path in music_paths.items()
        }
        self.VOICE_VOLUME = 0
        self.MUSIC_VOLUME = -25
        self.SFX_VOLUME = -18
        self.AMBIENCE_VOLUME = -30
        self.SPEAKER_PAUSE = 850
        self.CROSSFADE_LENGTH = 400
        self.INTRO_FADE = 2000
        self.EFFECT_FADE = 600

    def _get_secret(self, secret_name, project_id):
        try:
            client = secretmanager_v1.SecretManagerServiceClient()
            name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            service_account_json = response.payload.data.decode("UTF-8")
            credentials_dict = json.loads(service_account_json)
            for field in ['token_uri', 'client_email', 'private_key']:
                if field not in credentials_dict:
                    raise ValueError(f"Missing required service account field: {field}")
            return service_account_json            
        except Exception as e:
            logging.error(f"Error retrieving secret {secret_name}: {e}")
            raise

    def _load_audio_from_gcs(self, blob_path):
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(blob_path)
            audio_content = blob.download_as_bytes()
            audio_buffer = io.BytesIO(audio_content)
            return AudioSegment.from_mp3(audio_buffer)
        except Exception as e:
            logging.error(f"Error loading audio from {blob_path}: {e}")
            return AudioSegment.silent(duration=1000)
      
    def _compress_audio(self, audio: AudioSegment) -> AudioSegment:
        logging.info("Compressing audio")
        return compress_dynamic_range(audio, threshold=-20.0, ratio=4.0, attack=10, release=100)

    def _fade_effect(self, effect: AudioSegment) -> AudioSegment:
        logging.info("Fading effect")
        return effect.fade_in(200).fade_out(self.EFFECT_FADE)

    def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        logging.info("Normalizing audio")
        return normalize(audio)

    def _add_pause(self, audio: AudioSegment, duration: int) -> AudioSegment:
        logging.info("Adding pause")
        return audio + AudioSegment.silent(duration=duration)

    def to_bytes(self, mixed_audio: AudioSegment) -> bytes:
        logging.info("Converting audio to bytes")
        buffer = io.BytesIO()
        mixed_audio.export(buffer, format="mp3")
        audio_bytes = buffer.getvalue()
        buffer.close()
        return audio_bytes

    def mix_podcast_audio(self, voice_segments: List[Dict[str, bytes]], include_background: bool = True) -> bytes:
        logging.info("Starting podcast audio mixing")
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
            voice_audio = self._normalize_audio(voice_audio)
            voice_audio = self._compress_audio(voice_audio)
            current_speaker = segment.get("speaker", "")
            if previous_speaker and previous_speaker != current_speaker:
                final_mix = self._add_pause(final_mix, self.SPEAKER_PAUSE)
            if i == 0:
                final_mix = final_mix.append(voice_audio, crossfade=self.INTRO_FADE)
            else:
                final_mix = final_mix.append(voice_audio, crossfade=self.CROSSFADE_LENGTH)
            previous_speaker = current_speaker
        
        outro = self.background_music["outro"].fade_in(self.INTRO_FADE)
        final_mix = self._add_pause(final_mix, self.SPEAKER_PAUSE)
        final_mix = final_mix.append(outro, crossfade=self.INTRO_FADE)
        final_mix = self._compress_audio(final_mix)
        logging.info("Podcast audio mixing complete")
        return self.to_bytes(final_mix)

##########################################
# New Video Generation Components        #
##########################################

def generate_image(prompt: str) -> Image.Image:
    """
    Generate an image from a text prompt using Imagen via the Gen AI SDK.
    Ensure you have set your API key appropriately.
    """
    logging.info(f"Generating image for prompt: {prompt[:50]}...")
    client = genai.Client(vertexai=True, project="gem-rush-007", location="us-east4")
    imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-fast-generate-001")
    response = imagen_model.generate_images(
        model='imagen-3.0-generate-002',
        prompt=prompt,
        aspect_ratio='16:9'
        
    )
    image_bytes = response.generated_images[0].image.image_bytes
    return Image.open(io.BytesIO(image_bytes))

def analyze_script_for_scenes(script_data: List[Dict[str, str]]) -> List[str]:
    """
    Convert your podcast script (list of segments with speaker and text)
    into a list of scene prompts. This basic version concatenates the speaker and text.
    """
    scenes = []
    for segment in script_data:
        prompt = f"{segment['speaker']} says: {segment['text']}"
        scenes.append(prompt)
    return scenes

def generate_images_from_script(script_data: List[Dict[str, str]]) -> List[str]:
    """
    Generate and save one image per scene prompt.
    Returns a list of filenames.
    """
    scenes = analyze_script_for_scenes(script_data)
    image_files = []
    for i, prompt in enumerate(scenes):
        try:
            image = generate_image(prompt)
            filename = f"scene_{i+1}.jpg"
            image.save(filename)
            image_files.append(filename)
            logging.info(f"Saved image for scene {i+1} as {filename}")
        except Exception as e:
            logging.error(f"Error generating image for scene {i+1}: {e}")
    return image_files

def assemble_video_from_images_and_audio(image_files: List[str], audio_path: str, output_video: str):
    """
    Assemble a video from image files and overlay the given audio.
    The duration for each image is computed by dividing the total audio duration
    by the number of images.
    """
    logging.info("Starting video assembly...")
    clips = []
    # Use MoviePy to load audio and get its duration (in seconds)
    audio_clip = AudioFileClip(audio_path)
    total_duration = audio_clip.duration
    duration_per_scene = total_duration / len(image_files)
    logging.info(f"Total audio duration: {total_duration:.2f}s, each scene will last {duration_per_scene:.2f}s")
    for img_file in image_files:
        clip = ImageClip(img_file).set_duration(duration_per_scene)
        clips.append(clip)
    video_clip = concatenate_videoclips(clips, method="compose")
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(output_video, fps=24)
    logging.info("Video assembly complete.")

##########################################
# Integrated Podcast Video Generation    #
##########################################

def generate_podcast_video(script_data: List[Dict[str, str]],
                           voice_segments: List[Dict[str, bytes]],
                           audio_mixer: MLBAudioMixer,
                           output_video_filename: str):
    """
    Combines the audio mixing and video generation pipelines:
      1. Mix podcast audio from voice_segments.
      2. Generate scene images from the podcast script.
      3. Assemble the video with the mixed audio.
    """
    logging.info("Starting podcast video generation pipeline...")

    # Step 1: Mix Audio
    logging.info("Mixing podcast audio...")
    mixed_audio_bytes = audio_mixer.mix_podcast_audio(voice_segments)
    audio_output_path = "final_audio.mp3"
    with open(audio_output_path, "wb") as f:
        f.write(mixed_audio_bytes)
    logging.info(f"Final mixed audio saved to {audio_output_path}")

    # Step 2: Generate Video Frames (Images)
    logging.info("Generating video frames from script...")
    image_files = generate_images_from_script(script_data)
    if not image_files:
        logging.error("No images generated; aborting video creation.")
        return
    logging.info(f"Generated {len(image_files)} scene images.")

    # Step 3: Assemble Video
    logging.info("Assembling final video...")
    assemble_video_from_images_and_audio(image_files, audio_output_path, output_video_filename)
    logging.info(f"Podcast video created successfully: {output_video_filename}")

##########################################
# Example Usage
##########################################

if __name__ == "__main__":
    # Example podcast script (list of segments with speaker and text)
    podcast_script = [
        {'speaker': 'Play-by-play Announcer', 'text': "Welcome, everyone, to today's podcast! Today is February 17, 2025, and we're discussing the Royals' last game."},
        {'speaker': 'Color Commentator', 'text': "A tough loss for the Royals as the Twins shut them out."},
        {'speaker': 'Play-by-play Announcer', 'text': "Key moment: In the top of the first, Minnesota's Alex Kirilloff hit a two-run home run."},
        # Add additional segments as needed...
    ]

    # Example voice segments (each with 'speaker', 'audio' as bytes, and optionally 'text')
    # In practice, these would come from your voice recording processing
    voice_segments = [
        {'speaker': 'Play-by-play Announcer', 'audio': open("voice1.mp3", "rb").read(), 'text': "Welcome, everyone, to today's podcast!"},
        {'speaker': 'Color Commentator', 'audio': open("voice2.mp3", "rb").read(), 'text': "A tough loss for the Royals."},
        {'speaker': 'Play-by-play Announcer', 'audio': open("voice3.mp3", "rb").read(), 'text': "Key moment: Home run by Alex Kirilloff."},
        # Add additional segments as needed...
    ]

    # Initialize your MLBAudioMixer (adjust project_id and secret_name as needed)
    audio_mixer = MLBAudioMixer(project_id="gem-rush-007", secret_name="cloud-run-invoker")
    
    # Generate final video by combining audio and video generation pipelines
    generate_podcast_video(podcast_script, voice_segments, audio_mixer, output_video_filename="final_podcast_video.mp4")
