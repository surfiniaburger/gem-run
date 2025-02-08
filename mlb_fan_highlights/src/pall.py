#from google.cloud import texttospeech
from google.cloud import texttospeech_v1beta1 as texttospeech
import json
import os
from surwater_s import generate_mlb_podcasts
from spanish_audio_mixer import SpanishMLBAudioMixer
import uuid
from gcs_handler import GCSHandler
import logging
from google.cloud import logging as cloud_logging

# Configure cloud logging at the top of the script, before other imports
logging.basicConfig(level=logging.INFO)
log_client = cloud_logging.Client()
log_client.setup_logging()

project_id = "gem-rush-007"
secret_name = "cloud-run-invoker"

def create_audio_for_speaker(text, speaker_config):
    """Creates audio for a single piece of dialogue."""
    try:
        logging.info("Creating audio from a single piece of dialogue")
        client = texttospeech.TextToSpeechClient()
    
        input_text = texttospeech.SynthesisInput(text=text)
    
        voice = texttospeech.VoiceSelectionParams(
            language_code="es-ES",  # Spanish
            name=speaker_config["voice"],
            ssml_gender=texttospeech.SsmlVoiceGender[speaker_config["gender"]]
        )
    
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaker_config["speed"]
        )
    
        response = client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )
    
        return response.audio_content
    except Exception as e:      
     logging.error(f"Error Generating audio: {str(e)}")
     raise


def create_podcast(script_data, output_filename):
    """Process the entire podcast script with multiple speakers."""
    
    # Speaker configurations
    speaker_configs = {
        "Narrador de jugada por jugada": {
            "voice": "es-ES-Neural2-B",  # Male voice for play-by-play
            "gender": "MALE",
            "speed": 1.1  # Slightly faster for exciting moments
        },
        "Comentarista de color": {
            "voice": "es-ES-Neural2-C",  
            "gender": "FEMALE",
            "speed": 1.0
        },
        "Citas de Jugadores": {
            "voice": "es-ES-Neural2-D",  # Different voice for player quotes
            "gender": "FEMALE",
            "speed": 0.95  # Slightly slower for quotes
        }
    }
    
    combined_audio = b""
    
    for i, segment in enumerate(script_data):
        speaker = segment['speaker']
        text = segment['text'].strip()
        
        if speaker in speaker_configs and text:
            print(f"Processing segment {i}: {speaker}")
            
            # Generate audio for the segment
            audio_content = create_audio_for_speaker(text, speaker_configs[speaker])
            combined_audio += audio_content
            
            # Add a short pause between segments (0.5 second of silence)
            pause = create_audio_for_speaker(" ", speaker_configs[speaker])
            combined_audio += pause
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    # Write the final combined podcast
    with open(output_filename, "wb") as out:
        out.write(combined_audio)
        print(f'Full podcast written to file "{output_filename}"')
    
    return output_filename

def list_available_voices():
    client = texttospeech.TextToSpeechClient()
    voices = client.list_voices(language_code="es-ES")
    for voice in voices.voices:
        print(f"Name: {voice.name}")
        print(f"Gender: {voice.ssml_gender}")
        print(f"Language codes: {voice.language_codes}\n")

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
        # Ensure output filename is absolute and has no empty directory components
        output_filename = os.path.abspath(output_filename)        
        # Generate podcast script
        script_json = generate_mlb_podcasts(contents)
        print(script_json)
        
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
        mixer = SpanishMLBAudioMixer(project_id, secret_name)
        
        # Mix the audio with effects and background
        audio_bytes = mixer.mix_podcast_audio(voice_segments)
        
        # Upload using the new GCS handler
        logging.info("Uploading audio to GCS")
        gcs_handler = GCSHandler(secret_id=secret_name)

        
        url = gcs_handler.upload_audio(audio_bytes, f"podcast-{uuid.uuid4()}.mp3")
        logging.info(f"Successfully generated spanish mlb podcast and saved to GCS: {url}")

        if gcs_handler.is_url_expired(url):
            url = gcs_handler.refresh_signed_url(url)
            
        return url        
        
    except Exception as e:
        logging.error(f"Failed to generate Spanish MLB podcast: {str(e)}")
        raise Exception(f"Failed to generate Spanish MLB podcast: {str(e)}")
