from google.cloud import texttospeech_v1beta1 as texttospeech
import json
import os
from surfire import generate_mlb_podcasts
from typing import List, Dict
import uuid
from google.cloud import storage
from google.cloud import videointelligence_v1 as videointelligence
import time
import logging
from google.cloud import logging as cloud_logging
from google.oauth2 import service_account
from datetime import timedelta
from google.protobuf import duration_pb2
from google.protobuf import timestamp_pb2
from google.cloud import secretmanager_v1

# Configure cloud logging at the top of the script, before other imports
logging.basicConfig(level=logging.INFO)
log_client = cloud_logging.Client()
log_client.setup_logging()

project_id = "gem-rush-007"
secret_name = "cloud-run-invoker"


class MediaProcessor:
    def __init__(self, project_id, bucket_name, service_account_json):
        """Initializes the MediaProcessor"""
        self.project_id = project_id
        self.bucket_name = bucket_name
        # Initialize google cloud services
        self.storage_client = storage.Client.from_service_account_info(
             json.loads(service_account_json),
        )
        self.video_client = videointelligence.VideoIntelligenceServiceClient.from_service_account_info(
             json.loads(service_account_json),
        )
        
    def _upload_audio_to_gcs(self, audio_content: bytes, file_name: str) -> str:
        """Uploads audio to GCS and returns a signed URL."""
        try:
          # Get the bucket
          bucket = self.storage_client.bucket(self.bucket_name)
          # Upload the file
          blob = bucket.blob(file_name)
          blob.upload_from_string(audio_content, content_type="audio/mp3")
          #Generate the signed URL
          url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=15),
            method="GET"
          )
          return url
        except Exception as e:
            raise Exception(f"An error occurred while uploading audio to GCS: {e}")

    def mix_audio(self, audio_segments: List[Dict[str, bytes]], output_filename: str) -> str:
      """Mixes audio using Google Cloud Media Processing API"""
      try:
        # Upload the audio segments to the bucket
        gcs_uris = []
        for segment in audio_segments:
              file_name = f"segment-{uuid.uuid4()}.mp3"
              gcs_uris.append(self._upload_audio_to_gcs(segment["audio"], file_name))



        # Create the audio stream using the api
        inputs = [videointelligence.types.InputUri(uri=uri) for uri in gcs_uris]
        audio_stream = videointelligence.types.AudioStream(
           streams=inputs,
        )
    
        #Define audio format
        audio_format = videointelligence.types.AudioFormat(
                encoding="mp3",
                sample_rate_hertz=44100,
                channel_count=2,
             )


        #Set an output
        output_uri =  self._upload_audio_to_gcs(b'', output_filename) #empty bytes since we are not uploading anything here, just creating a uri
        output_config = videointelligence.types.OutputConfig(uri = output_uri, audio_format=audio_format)

        # Create processing config
        processing_config = videointelligence.types.ProcessingConfig(
           audio_stream=audio_stream,
           output_config=output_config
        )
        
        # Create request for media processing
        request = videointelligence.types.ProcessRequest(
                project_id = self.project_id,
                 location_id="us-central1",
                processing_config = processing_config
             )


        #Start processing the audio using the API
        operation = self.video_client.process(request=request)
        logging.info(f"Waiting for operation to complete: {operation.operation.name}")
        
        while not operation.done():
             time.sleep(10)
             logging.info(f"Operation {operation.operation.name} is still processing")
        
        if operation.result():
            logging.info(f"Operation {operation.operation.name} has completed successfully")
            return output_uri
        
        else:
             raise Exception(f"Operation {operation.operation.name} has failed with error: {operation.operation.error}")

      except Exception as e:
        logging.error(f"Error while mixing audio with google cloud api: {e}")
        raise Exception(f"Failed to mix audio using Google Cloud Media Processing API: {e}")


class SpanishMLBAudioMixer:
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
            "intro": "assets/music/ranchero.mp3",
            "highlight": "assets/music/highlight.mp3",
            "outro": "assets/music/ranchero-outro.mp3"
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
           # Create a google cloud client
            storage_client = storage.Client()
            # Get the bucket
            bucket = storage_client.bucket(self.bucket_name)
            # Get the blob
            blob = bucket.blob(blob_path)
            # Download bytes
            audio_bytes = blob.download_as_bytes()
             # Convert to AudioSegment
            return audio_bytes
        except Exception as e:
            logging.error(f"Error loading audio from {blob_path}: {e}")
            # Fallback to silent audio if loading fails
            return b''

    def _compress_audio(self, audio: bytes) -> bytes:
        return audio # We no longer use `pydub` for this purpose

    def _fade_effect(self, effect: bytes) -> bytes:
        return effect # We no longer use `pydub` for this purpose

    def _process_voice_segment(self, voice_audio: bytes) -> bytes:
         return voice_audio # We no longer use `pydub` for this purpose

    def mix_podcast_audio(self, voice_segments: List[Dict[str, bytes]]) -> bytes:
         return voice_segments # We no longer use `pydub` for this purpose

    def _add_pause(self, audio: bytes, duration: int) -> bytes:
         return audio # We no longer use `pydub` for this purpose

    def _detect_spanish_event_triggers(self, text: str) -> List[str]:
        triggers = []
        events = {
            "crowd_cheer": [
                "jonrón", "cuadrangular", "anota", "carrera", "victoria", 
                "gana", "walk-off", "remonta", "celebra"
            ],
            "bat_hit": [
                "hit", "sencillo", "doble", "triple", "batazo", 
                "conecta", "línea", "imparable"
            ],
            "crowd_tension": [
                "cuenta llena", "bases llenas", "última entrada", 
                "novena entrada", "momento crucial", "presión"
            ],
        }
        
        text_lower = text.lower()
        for effect, keywords in events.items():
            if any(keyword in text_lower for keyword in keywords):
                triggers.append(effect)
                
        return triggers

    def to_bytes(self, mixed_audio: bytes) -> bytes:
        """Converts the mixed AudioSegment to bytes."""
        return mixed_audio
    
def generate_spanish_audio(contents: str, language: str, output_filename: str = "spanish_mlb_podcast.mp3") -> str:
    """
    Generates a Spanish MLB podcast with sound effects and music mixing.
    
    Args:
        contents: Input text/data for generating the podcast script
        output_filename: Desired name for the output audio file
        
    Returns:
        str: Path to the generated podcast file
    """
    try:
        logging.info("genearating spanish audio")
        # Speaker configurations for Spanish voices
        speaker_configs = {
            "Narrador de jugada por jugada": {
                "voice": "es-ES-Neural2-B",
                "gender": "MALE",
                "speed": 1.1
            },
            "Comentarista de color": {
                "voice": "es-ES-Neural2-C",
                "gender": "FEMALE",
                "speed": 1.0
            },
            "Citas de Jugadores": {
                "voice": "es-ES-Neural2-D",
                "gender": "FEMALE",
                "speed": 0.95
            }
        }

        # Initialize TTS client
        client = texttospeech.TextToSpeechClient()
        
        # Generate podcast script
        script_json = generate_mlb_podcasts(contents)
        
        if isinstance(script_json, dict) and "error" in script_json:
            raise Exception(f"Script generation error: {script_json['error']}")
        
        # Generate voice segments
        voice_segments = []
        for segment in script_json:
            speaker = segment['speaker']
            text = segment['text'].strip()
            
            if speaker in speaker_configs and text:
                print(f"Processing segment: {speaker}")
                
                # Create input for text-to-speech
                input_text = texttospeech.SynthesisInput(text=text)
                
                voice = texttospeech.VoiceSelectionParams(
                    language_code="es-ES",
                    name=speaker_configs[speaker]["voice"],
                    ssml_gender=texttospeech.SsmlVoiceGender[speaker_configs[speaker]["gender"]]
                )
                
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3,
                    speaking_rate=speaker_configs[speaker]["speed"]
                )
                
                # Generate audio for segment
                response = client.synthesize_speech(
                    request={"input": input_text, "voice": voice, "audio_config": audio_config}
                )
                
                voice_segments.append({
                    "audio": response.audio_content,
                    "text": text,
                    "speaker": speaker
                })
        
         # Initialize the audio mixer
        mixer = MediaProcessor(project_id, "mlb-audio-assets", _get_secret(secret_name, project_id))
        
        # Mix the audio with effects and background
        audio_url = mixer.mix_audio(voice_segments, output_filename)
      
        
        print(f"Successfully generated spanish mlb podcast and saved to GCS: {audio_url}")
        return audio_url
        
    except Exception as e:
        logging.error(f"Failed to generate Spanish MLB podcast: {str(e)}")
        raise Exception(f"Failed to generate Spanish MLB podcast: {str(e)}")

def _get_secret(secret_name, project_id):
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