#from google.cloud import texttospeech
import traceback
from google.cloud import  texttospeech
import json
import os
from surwater_j import generate_mlb_podcasts
from japanese_audio_mixer import JapaneseMLBAudioMixer
from gcs_handler import GCSHandler
import uuid
import logging
from google.cloud import logging as cloud_logging
from vid import CloudVideoGenerator


project_id = "gem-rush-007"
secret_name = "cloud-run-invoker"

# Configure cloud logging at the top of the script, before other imports
logging.basicConfig(level=logging.INFO)
log_client = cloud_logging.Client()
log_client.setup_logging()

def create_audio_for_speaker(text, speaker_config):
    """Creates audio for a single piece of dialogue."""
    client = texttospeech.TextToSpeechClient()
    
    input_text = texttospeech.SynthesisInput(text=text)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code="ja-JP",  # Japanese
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

def create_podcast(script_data, output_filename):
    """Process the entire podcast script with multiple speakers."""
    
    # Speaker configurations
    speaker_configs = {
       "実況アナウンサー": {
           "voice": "ja-JP-Neural2-B",  # Male voice for play-by-play
           "gender": "FEMALE",
           "speed": 1.1  # Slightly faster for exciting moments
        },
       "解説者": {
           "voice": "ja-JP-Neural2-C",  
           "gender": "MALE",
           "speed": 1.0  # Normal speed for analysis
        },
       "選手の声": {
           "voice": "ja-JP-Neural2-D",  # Different voice for player quotes
           "gender": "MALE",
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
    voices = client.list_voices(language_code="ja-JP")
    for voice in voices.voices:
        print(f"Name: {voice.name}")
        print(f"Gender: {voice.ssml_gender}")
        print(f"Language codes: {voice.language_codes}\n")


def generate_japanese_audio(contents: str, language: str, output_filename: str = "japanese_mlb_podcast.mp3") -> str:
    logging.info("Generating Japanese audio")
    """
    Generates a Japanese MLB podcast with sound effects and music mixing.
    
    Args:
        contents: Input text/data for generating the podcast script
        output_filename: Desired name for the output audio file
        
    Returns:
        str: Path to the generated podcast file
    """
    try:
        logging.info("generating spanish audio")
        logging.info(f"Content been seent to generate script: {contents}")

        speaker_configs = {
            "実況アナウンサー": {
              "voice": "ja-JP-Neural2-B",  # Male voice for play-by-play
              "gender": "FEMALE",
              "speed": 1.1  # Slightly faster for exciting moments
            },
           "解説者": {
              "voice": "ja-JP-Neural2-C",  
              "gender": "MALE",
              "speed": 1.0  # Normal speed for analysis
            },
           "選手の声": {
              "voice": "ja-JP-Neural2-D",  # Different voice for player quotes
              "gender": "MALE",
              "speed": 0.95  # Slightly slower for quotes
            }
        }

        # Initialize TTS client
        client = texttospeech.TextToSpeechClient()
        logging.info("TTS client initialized")
        # Generate podcast script
        script_json = generate_mlb_podcasts(contents)
        print(script_json)
        print(type(generate_mlb_podcasts))
        print(generate_mlb_podcasts)
        
        if isinstance(script_json, dict) and "error" in script_json:
            raise Exception(f"Script generation error: {script_json['error']}")
        
        # Generate voice segments
        voice_segments = []
        for segment in script_json:
            try:
             speaker = segment['speaker']
             text = segment['text'].strip()
            
             if speaker in speaker_configs and text:
                print(f"Processing segment: {speaker}")
                
                # Create input for text-to-speech
                input_text = texttospeech.SynthesisInput(text=text)
                
                voice = texttospeech.VoiceSelectionParams(
                    language_code="ja-JP",
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
            except Exception as segment_error:
                logging.error(f"Error processing segment: {segment_error}")
                raise        
        # Initialize the audio mixer
        mixer = JapaneseMLBAudioMixer(project_id, secret_name)
        
        # Mix the audio with effects and background
        audio_bytes = mixer.mix_podcast_audio(voice_segments)
        
        # Upload using the new GCS handler
        logging.info("Uploading audio to GCS")
        gcs_handler = GCSHandler(secret_id=secret_name)
        
        url = gcs_handler.upload_audio(audio_bytes, f"podcast-{uuid.uuid4()}.mp3")
        logging.info(f"Successfully generated Japanese mlb podcast and saved to GCS: {url}")
        #video_gen = CloudVideoGenerator(gcs_handler)
        #gcs_uri = gcs_handler.signed_url_to_gcs_uri(url)
        #video_url = video_gen.create_video(gcs_uri, script_json)
        #print(video_url)
        return url        
        
    except Exception as e:
        logging.error(f"Failed to generate Japanese MLB podcast: {str(e)}")
        logging.error(f"Detailed error: {type(e)}")
        logging.error(f"Error details: {str(e)}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        raise
