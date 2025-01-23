#from google.cloud import texttospeech
from google.cloud import  texttospeech
import json
import os
from surfire import generate_mlb_podcasts
from japanese_audio_mixer import JapaneseMLBAudioMixer
from gcs_handler import GCSHandler
import uuid

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


def generate_japanese_audio(contents: str, language: str, output_filename: str = "mlb_podcast.mp3") -> str:
    """
    Main function to generate and synthesize MLB podcast with language support
    """
    try:
        output_filename = os.path.abspath(output_filename)
        script_json = generate_mlb_podcasts(contents)
        print(script_json)
        
        if isinstance(script_json, dict) and "error" in script_json:
            raise Exception(f"Script generation error: {script_json['error']}")
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
            
        # First create the individual voice segments
        voice_segments = []
        for segment in script_json:
            speaker = segment['speaker']
            text = segment['text'].strip()

            
            
            if speaker in speaker_configs and text:
                print(f"Processing segment: {speaker}")
                audio_content = create_audio_for_speaker(text, speaker_configs[speaker])
                voice_segments.append({
                    "audio": audio_content,
                    "text": text,
                    "speaker": speaker
                })
        
        # Initialize the audio mixer
        mixer = JapaneseMLBAudioMixer()
        
        
        # Mix the audio with effects and background
        audio_bytes = mixer.mix_podcast_audio(voice_segments)
        
        # Upload using the new GCS handler
        
        key_file_path = "./gem-rush-007-a9765f2ada0e.json"  # Same path as your working command line example
        gcs_handler = GCSHandler(key_file_path=key_file_path)
        
        url = gcs_handler.upload_audio(audio_bytes, f"podcast-{uuid.uuid4()}.mp3")
        return url       
        
    except Exception as e:
        raise Exception(f"Failed to generate script: {str(e)}")


