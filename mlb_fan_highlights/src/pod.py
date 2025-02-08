from google.cloud import texttospeech_v1beta1 as texttospeech
from google.api_core.client_options import ClientOptions
import os
import json
from typing import List, Dict
from surwater import generate_mlb_podcasts
import logging
from audio_mixer import MLBAudioMixer
logging.basicConfig(level=logging.DEBUG)
from gcs_handler import GCSHandler
import uuid
from google.cloud import logging as cloud_logging

# Configure cloud logging at the top of the script, before other imports
logging.basicConfig(level=logging.INFO)
log_client = cloud_logging.Client()
log_client.setup_logging()

project_id = "gem-rush-007"
secret_name = "cloud-run-invoker"


class MLBPodcastSynthesizer:
    def __init__(self, tts_location: str = "us"):
        self.tts_location = tts_location
        # Default to English
        self.set_language("English")
        self.tts_client = texttospeech.TextToSpeechClient(
            client_options=ClientOptions(
                api_endpoint=f"{self.tts_location}-texttospeech.googleapis.com"
            )
        )

    def set_language(self, language: str):
        """Configure language-specific settings."""
        language_settings = {
            "English": {
                "code": "en-US",
                "voices": {
                    "Play-by-play Announcer": "en-US-News-N",
                    "Color Commentator": "en-US-Wavenet-H",
                    "Player Quotes": "en-US-Wavenet-D"
                }
            },
            "Spanish": {
                "code": "es-ES",
                "voices": {
                    "Play-by-play Announcer": "es-ES-Standard-C",
                    "Color Commentator": "es-ES-Standard-E",
                    "Player Quotes": "es-ES-Studio-F"
                }
            },
            "Japanese": {
                "code": "ja-JP",
                "voices": {
                    "Play-by-play Announcer": "ja-JP-Neural2-B",
                    "Color Commentator": "ja-JP-Neural2-C",
                    "Player Quotes": "ja-JP-Neural2-D"
                }
            }
        }
        
        if language not in language_settings:
            raise ValueError(f"Unsupported language: {language}")
            
        settings = language_settings[language]
        self.language_code = settings["code"]
        self.voices = settings["voices"]

    def synthesize_speech(self, text: str, voice_name: str) -> bytes:
        """Synthesizes speech from text using specified voice."""
        logging.debug(f"Synthesizing speech with language code: {self.language_code}, voice name: {voice_name}")
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=self.language_code,
            name=voice_name
        )
        
        # Adjust audio configuration based on language
        speaking_rate = 1.0
        if self.language_code.startswith("ja"):
            # Japanese typically needs a slightly slower rate
            speaking_rate = 0.9
        elif self.language_code.startswith("es"):
            # Spanish can handle a slightly faster rate
            speaking_rate = 1.1
            
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate
        )
        
        try:
             response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
             logging.debug(f"Response from TTS: {response}")
             return response.audio_content
        except Exception as e:
             logging.error(f"Error when synthesizing audio for language: {self.language_code} and voice : {voice_name}, error was : {e}")
             return b''

    def create_podcast(self, script: List[Dict[str, str]], output_filename: str) -> str:
       """Creates a multi-speaker podcast from the generated script with enhanced audio."""
       voice_segments = []
    
       for segment in script:
           speaker = segment["speaker"]
           text = segment["text"]
        
           if speaker in self.voices:
               voice_name = self.voices[speaker]
               audio_content = self.synthesize_speech(text, voice_name)
            
               voice_segments.append({
                "speaker": speaker,
                "text": text,
                "audio": audio_content
                })
        # Initialize audio mixer
       mixer = MLBAudioMixer(project_id, secret_name)
    # Mix the audio with effects and background
       audio_bytes = mixer.mix_podcast_audio(voice_segments)
        
        # Upload using the new GCS handler
        
       logging.info("Uploading audio to GCS")
       gcs_handler = GCSHandler(secret_id=secret_name) 
       url = gcs_handler.upload_audio(audio_bytes, f"podcast-{uuid.uuid4()}.mp3")
       return url    

def generate_mlb_podcast_with_audio(contents: str, language: str = "English", output_filename: str = "mlb_podcast.mp3") -> str:
    """
    Main function to generate and synthesize MLB podcast with language support
    """
    try:
        # Ensure output filename is absolute and has no empty directory components
        output_filename = os.path.abspath(output_filename)
        # Generate the podcast script
        script_json = generate_mlb_podcasts(contents)
        print(script_json)
        
        # Check for errors in script generation
        if isinstance(script_json, dict) and "error" in script_json:
            raise Exception(f"Script generation error: {script_json['error']}")
        
        # Handle Spanish podcasts differently
        if language == "Spanish":
            from spanish_handler import create_spanish_podcast
            return create_spanish_podcast(script_json, output_filename)
        
        # Initialize the synthesizer with the selected language
        synthesizer = MLBPodcastSynthesizer()
        synthesizer.set_language(language)
        
        # Create the audio podcast
        audio_file = synthesizer.create_podcast(script_json, output_filename)
        
        return audio_file
        
    except Exception as e:
        raise Exception(f"Failed to generate podcast: {str(e)}")